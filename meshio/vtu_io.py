# -*- coding: utf-8 -*-
#
'''
I/O for VTU.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import base64
# lxml cannot parse large files and instead throws the exception
#
# lxml.etree.XMLSyntaxError: xmlSAX2Characters: huge text node, [...]
#
# Use Python's native xml parser to avoid this error.
import xml.etree.ElementTree as ET
import zlib

import numpy

from .vtk_io import vtk_to_meshio_type, cell_data_from_raw
from .gmsh_io import num_nodes_per_cell


def num_bytes_to_num_base64_chars(num_bytes):
    if num_bytes % 3 == 0:
        num_chars = num_bytes // 3 * 4
    elif num_bytes % 3 == 1:
        num_chars = (num_bytes+2) // 3 * 4
    else:
        assert num_bytes % 3 == 2
        num_chars = (num_bytes+1) // 3 * 4
    return num_chars


def _cells_from_data(connectivity, offsets, types):
    # Translate it into the cells dictionary.
    # `connectivity` is a one-dimensional vector with
    # (p0, p1, ... ,pk, p10, p11, ..., p1k, ...

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

    assert len(offsets) == len(types)

    cells = {}
    for tpe, b in bins.items():
        meshio_type = vtk_to_meshio_type[tpe]
        n = num_nodes_per_cell[meshio_type]
        indices = numpy.array([
            # The offsets point to the _end_ of the indices
            numpy.arange(n) + o - n for o in offsets[b]
            ])
        cells[meshio_type] = connectivity[indices]

    return cells


# pylint: disable=too-many-instance-attributes
class VtuReader(object):
    '''Helper class for reading VTU files. Some properties are global to the
    file (e.g., byte_order), and instead of passing around these parameters,
    make them properties of this class.
    '''
    def __init__(self, filename):
        points = None
        point_data = {}
        cell_data_raw = {}
        cells = {}
        field_data = {}

        self.vtu_to_numpy_type = {
            'Float32': numpy.float32,
            'Float64': numpy.float64,
            'Int8': numpy.int8,
            'Int16': numpy.int16,
            'Int32': numpy.int32,
            'Int64': numpy.int64,
            'UInt8': numpy.uint8,
            'UInt16': numpy.uint16,
            'UInt32': numpy.uint32,
            'UInt64': numpy.uint64,
            }

        tree = ET.parse(filename)
        root = tree.getroot()

        assert root.tag == 'VTKFile'
        assert root.attrib['type'] == 'UnstructuredGrid'
        assert root.attrib['version'] in ['0.1', '1.0'], \
            'Unknown VTU file version \'{}\'.'.format(root.attrib['version'])

        try:
            assert root.attrib['compressor'] == 'vtkZLibDataCompressor'
        except KeyError:
            pass

        self.header_type = (
            root.attrib['header_type'] if 'header_type' in root.attrib
            else 'UInt32'
            )

        try:
            self.byte_order = root.attrib['byte_order']
            assert self.byte_order in ['LittleEndian', 'BigEndian'], \
                'Unknown byte order \'{}\'.'.format(self.byte_order)
        except KeyError:
            self.byte_order = None

        grid = None
        self.appended_data = None
        for c in root:
            if c.tag == 'UnstructuredGrid':
                assert grid is None, 'More than one UnstructuredGrid found.'
                grid = c
            else:
                assert c.tag == 'AppendedData', \
                    'Unknown main tag \'{}\'.'.format(c.tag)
                assert self.appended_data is None, \
                    'More than one AppendedData found.'
                assert c.attrib['encoding'] == 'base64'
                self.appended_data = c.text.strip()
                # The appended data always begins with a (meaningless)
                # underscore.
                assert self.appended_data[0] == '_'
                self.appended_data = self.appended_data[1:]

        assert grid is not None, 'No UnstructuredGrid found.'

        piece = None
        for c in grid:
            if c.tag == 'Piece':
                assert piece is None, 'More than one Piece found.'
                piece = c
            else:
                assert c.tag == 'FieldData', \
                    'Unknown grid subtag \'{}\'.'.format(c.tag)
                # TODO test field data
                for data_array in c:
                    field_data[data_array.attrib['Name']] = \
                        self.read_data(data_array)

        assert piece is not None, 'No Piece found.'

        num_points = int(piece.attrib['NumberOfPoints'])
        num_cells = int(piece.attrib['NumberOfCells'])

        for child in piece:
            if child.tag == 'Points':
                data_arrays = list(child)
                assert len(data_arrays) == 1
                data_array = data_arrays[0]

                assert data_array.tag == 'DataArray'

                points = self.read_data(data_array)

                num_components = int(data_array.attrib['NumberOfComponents'])
                points = points.reshape(num_points, num_components)

            elif child.tag == 'Cells':
                for data_array in child:
                    assert data_array.tag == 'DataArray'
                    cells[data_array.attrib['Name']] = \
                        self.read_data(data_array)

                assert len(cells['offsets']) == num_cells
                assert len(cells['types']) == num_cells

            elif child.tag == 'PointData':
                for c in child:
                    assert c.tag == 'DataArray'
                    point_data[c.attrib['Name']] = self.read_data(c)

            else:
                assert child.tag == 'CellData', \
                    'Unknown tag \'{}\'.'.format(child.tag)

                for c in child:
                    assert c.tag == 'DataArray'
                    cell_data_raw[c.attrib['Name']] = self.read_data(c)

        assert points is not None
        assert 'connectivity' in cells
        assert 'offsets' in cells
        assert 'types' in cells

        cells = _cells_from_data(
                cells['connectivity'], cells['offsets'], cells['types']
                )

        # get cell data in shape
        cell_data = cell_data_from_raw(cells, cell_data_raw)

        self.points = points
        self.cells = cells
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data
        return

    def read_binary(self, data, data_type):
        # first read the the block size; it determines the size of the header
        dtype = self.vtu_to_numpy_type[self.header_type]
        num_bytes_per_item = numpy.dtype(dtype).itemsize
        num_chars = num_bytes_to_num_base64_chars(num_bytes_per_item)
        byte_string = base64.b64decode(data[:num_chars])[:num_bytes_per_item]
        num_blocks = numpy.fromstring(byte_string, dtype)[0]

        # read the entire header
        num_header_items = 3 + num_blocks
        num_header_bytes = num_bytes_per_item * num_header_items
        num_header_chars = num_bytes_to_num_base64_chars(num_header_bytes)
        byte_string = base64.b64decode(data[:num_header_chars])
        header = numpy.fromstring(byte_string, dtype)

        # num_blocks = header[0]
        # max_uncompressed_block_size = header[1]
        # last_compressed_block_size = header[2]
        block_sizes = header[3:]

        # Read the block data
        byte_array = base64.b64decode(data[num_header_chars:])
        dtype = self.vtu_to_numpy_type[data_type]
        num_bytes_per_item = numpy.dtype(dtype).itemsize

        byte_offsets = numpy.concatenate(
                [[0], numpy.cumsum(block_sizes, dtype=block_sizes.dtype)]
                )
        # https://github.com/numpy/numpy/issues/10135
        byte_offsets = byte_offsets.astype(numpy.int64)

        # process the compressed data
        block_data = numpy.concatenate([
                numpy.fromstring(zlib.decompress(
                    byte_array[byte_offsets[k]:byte_offsets[k+1]]
                    ), dtype=dtype)
                for k in range(num_blocks)
                ])

        return block_data

    def read_data(self, c):
        if c.attrib['format'] == 'ascii':
            # ascii
            data = numpy.array(
                c.text.split(),
                dtype=self.vtu_to_numpy_type[c.attrib['type']]
                )
        elif c.attrib['format'] == 'binary':
            data = self.read_binary(c.text.strip(), c.attrib['type'])
        else:
            # appended data
            assert c.attrib['format'] == 'appended', \
                'Unknown data format \'{}\'.'.format(c.attrib['format'])

            offset = int(c.attrib['offset'])
            data = self.read_binary(
                    self.appended_data[offset:], c.attrib['type']
                    )

        if 'NumberOfComponents' in c.attrib:
            data = data.reshape(-1, int(c.attrib['NumberOfComponents']))
        return data


def read(filename):
    reader = VtuReader(filename)
    return (
        reader.points, reader.cells, reader.point_data, reader.cell_data,
        reader.field_data
        )


def write(filetype,
          filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    # pylint: disable=import-error
    from .vtk_io import write as vtk_write
    return vtk_write(
        filetype, filename, points, cells,
        point_data=point_data, cell_data=cell_data, field_data=field_data
        )
