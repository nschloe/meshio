# -*- coding: utf-8 -*-
#
'''
I/O for VTU.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import base64
import struct
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
    print(max(offsets), len(offsets))
    print(len(connectivity))
    print((types == 10).all())

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


class VtuReader(object):
    '''Helper class for reading VTU files. Some properties are global to the
    file (e.g., byte_order), and instead of passing around these parameters,
    make them properties of this class.
    '''
    def __init__(self, filename):
        from lxml import etree as ET

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

        self.vtu_to_struct_type = {
            'Float32': ('f', 4),
            'Float64': ('d', 8),
            'Int32': ('i', 4),
            'Int64': ('q', 8),
            'UInt8': ('B', 1),
            'UInt32': ('I', 4),
            'UInt64': ('Q', 8),
            }

        tree = ET.parse(filename)
        root = tree.getroot()

        assert root.tag == 'VTKFile'
        assert root.attrib['type'] == 'UnstructuredGrid'
        assert root.attrib['version'] in ['0.1', '1.0'], \
            'Unknown VTU file version \'{}\'.'.format(root.attrib['version'])
        assert root.attrib['compressor'] == 'vtkZLibDataCompressor'

        self.header_type = (
            root.attrib['header_type'] if 'header_type' in root.attrib
            else 'UInt32'
            )

        self.byte_order = root.attrib['byte_order']
        assert self.byte_order in ['LittleEndian', 'BigEndian'], \
            'Unknown byte order \'{}\'.'.format(self.byte_order)

        grid = None
        self.appended_data = None
        for c in root.getchildren():
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
        for c in grid.getchildren():
            if c.tag == 'Piece':
                assert piece is None, 'More than one Piece found.'
                piece = c
            else:
                assert c.tag == 'FieldData', \
                    'Unknown grid subtag \'{}\'.'.format(c.tag)
                # TODO test field data
                data_arrays = c.getchildren()
                assert len(data_arrays) == 1
                data_array = data_arrays[0]
                field_data[data_array.attrib['Name']] = \
                    self.read_data(data_array)

        assert piece is not None, 'No Piece found.'

        num_points = int(piece.attrib['NumberOfPoints'])
        num_cells = int(piece.attrib['NumberOfCells'])

        for child in piece.getchildren():
            if child.tag == 'Points':
                data_arrays = child.getchildren()
                assert len(data_arrays) == 1
                data_array = data_arrays[0]

                assert data_array.tag == 'DataArray'
                assert data_array.attrib['Name'] == 'Points'

                points = self.read_data(data_array)

                num_components = int(data_array.attrib['NumberOfComponents'])
                points = points.reshape(num_points, num_components)

            elif child.tag == 'Cells':
                for data_array in child.getchildren():
                    assert data_array.tag == 'DataArray'
                    cells[data_array.attrib['Name']] = \
                        self.read_data(data_array)

                assert len(cells['offsets']) == num_cells
                assert len(cells['types']) == num_cells

            elif child.tag == 'PointData':
                for c in child.getchildren():
                    assert c.tag == 'DataArray'
                    point_data[c.attrib['Name']] = self.read_data(c)

            else:
                assert child.tag == 'CellData', \
                    'Unknown tag \'{}\'.'.format(child.tag)
                for c in child.getchildren():
                    assert c.tag == 'DataArray'
                    cell_data_raw[c.attrib['Name']] = self.read_data(c)

        assert points is not None
        assert 'connectivity' in cells
        assert 'offsets' in cells
        assert 'types' in cells

        print(points.shape)
        print(cells)
        print(cells['connectivity'].shape)
        print(cells['offsets'].shape)
        print(cells['types'].shape)

        cells = _cells_from_data(
                cells['connectivity'], cells['offsets'], cells['types']
                )

        # get point data in shape
        for key in point_data:
            if len(point_data[key]) != len(points):
                point_data[key] = point_data[key].reshape(len(points), -1)

        # get cell data in shape
        cell_data = cell_data_from_raw(cells, cell_data_raw)

        self.points = points
        self.cells = cells
        self.point_data = point_data
        self.cell_data = cell_data
        self.field_data = field_data
        return

    def read_binary(self, data, data_type):
        # https://docs.python.org/2/library/struct.html

        # process the header
        byte_string = base64.b64decode(data)
        bo = '<' if self.byte_order == 'LittleEndian' else '>'
        symbol, num_bytes = self.vtu_to_struct_type[self.header_type]
        num_blocks = int(struct.unpack(
            bo + symbol, byte_string[0:num_bytes]
            )[0])
        # Not needed:
        # uncompressed_size = max uncompressed block size
        # uncompressed_size = int(struct.unpack(
        #     bo + symbol, byte_string[num_bytes:2*num_bytes]
        #     )[0])
        # block size after compression:
        last_block_size = int(struct.unpack(
            bo + symbol, byte_string[2*num_bytes:3*num_bytes]
            )[0])

        # TODO numpy
        block_sizes = [
            int(struct.unpack(
                bo + symbol,
                byte_string[(3+k)*num_bytes:(4+k)*num_bytes]
                )[0])
            for k in range(num_blocks)
            ]

        print('    ', num_blocks, block_sizes, last_block_size)
        # print('     uncompressed: ', uncompressed_size)

        # Check how many characters the header occupies. This is determined
        # according to base64 encoding.
        header_num_bytes = (3 + num_blocks) * num_bytes
        char_offset = num_bytes_to_num_base64_chars(header_num_bytes)

        block_data = []
        for k in range(num_blocks):
            block_num_bytes = block_sizes[k]
            block_num_chars = num_bytes_to_num_base64_chars(block_num_bytes)

            print('     =========')
            print('     data type', data_type)
            print('     block num bytes', block_num_bytes)
            print('     block num chars', block_num_chars)
            print('     char_offset', char_offset)

            # process the compressed data
            compressed_data = data[char_offset:char_offset + block_num_chars]
            # print(compressed_data)
            print('     len(compressed data)', len(compressed_data))
            decoded = base64.b64decode(compressed_data)
            print('     decoded bytes', len(decoded))
            decompressed = zlib.decompress(decoded)
            print('     decompressed bytes', len(decompressed))

            char_offset += block_num_chars

            struct_type, num_bytes = self.vtu_to_struct_type[data_type]

            assert len(decompressed) % num_bytes == 0

            # TODO numpy function
            block_data.append(numpy.array([
                struct.unpack(
                    struct_type,
                    decompressed[num_bytes*k:num_bytes*(k+1)]
                    )[0]
                for k in range(len(decompressed) // num_bytes)
                ]))

        return numpy.concatenate(block_data)

    def read_appended(self, offset, dtype):
        data = self.appended_data[offset:]
        return self.read_binary(data, dtype)

    def read_data(self, c):
        if c.attrib['format'] == 'ascii':
            # ascii
            return numpy.array(
                c.text.split(),
                dtype=self.vtu_to_numpy_type[c.attrib['type']]
                )
        elif c.attrib['format'] == 'binary':
            return self.read_binary(c.text.strip(), c.attrib['type'])

        print(c.attrib['Name'])

        # appended
        assert c.attrib['format'] == 'appended', \
            'Unknown data format \'{}\'.'.format(c.attrib['format'])

        print('     offset ', int(c.attrib['offset']))
        return self.read_appended(int(c.attrib['offset']), c.attrib['type'])


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
