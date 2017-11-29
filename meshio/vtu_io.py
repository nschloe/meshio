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


def _cells_from_data(connectivity, offsets, types):
    # Translate it into the cells dictionary.
    # `connectivity` is a one-dimensional vector with
    # (p0, p1, ... ,pk, p10, p11, ..., p1k, ...

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

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


def _read_binary(data, byte_order, header_type, data_type):
    # https://docs.python.org/2/library/struct.html
    vtu_to_struct_type = {
        'Float32': ('f', 4),
        'Float64': ('d', 8),
        'Int32': ('i', 4),
        'Int64': ('q', 8),
        'UInt8': ('B', 1),
        'UInt32': ('I', 4),
        'UInt64': ('Q', 8),
        }

    # process the header
    byte_string = base64.b64decode(data)
    bo = '<' if byte_order == 'LittleEndian' else '>'
    symbol, num_bytes = vtu_to_struct_type[header_type]
    num_blocks = int(struct.unpack(
        bo + symbol, byte_string[0:num_bytes]
        )[0])
    # Not needed:
    # uncompressed_size = int(struct.unpack(
    #     bo + symbol, byte_string[num_bytes:2*num_bytes]
    #     )[0])
    # last_block_size = int(struct.unpack(
    #     bo + symbol, byte_string[2*num_bytes:3*num_bytes]
    #     )[0])
    # block_sizes = [
    #     int(struct.unpack(
    #         bo + symbol,
    #         byte_string[(3+k)*num_bytes:(4+k)*num_bytes]
    #         )[0])
    #     for k in range(num_blocks)
    #     ]

    # Check how many characters the header occupies. This is
    # determined according to base64 encoding.
    header_num_bytes = (3 + num_blocks) * num_bytes
    if header_num_bytes % 3 == 0:
        header_num_chars = header_num_bytes // 3 * 4
    elif header_num_bytes % 3 == 1:
        header_num_chars = (header_num_bytes+2) // 3 * 4
    else:
        assert header_num_bytes % 3 == 2
        header_num_chars = (header_num_bytes+1) // 3 * 4

    # process the compressed data
    compressed_data = data[header_num_chars:]
    decompressed = \
        zlib.decompress(base64.b64decode(compressed_data))

    struct_type, num_bytes = vtu_to_struct_type[data_type]

    assert len(decompressed) % num_bytes == 0

    out = numpy.array([
        struct.unpack(
            struct_type,
            decompressed[num_bytes*k:num_bytes*(k+1)]
            )[0]
        for k in range(len(decompressed) // num_bytes)
        ])

    return out


def read(filename):
    from lxml import etree as ET

    tree = ET.parse(filename)
    root = tree.getroot()

    assert root.tag == 'VTKFile'
    assert root.attrib['type'] == 'UnstructuredGrid'
    assert root.attrib['version'] in ['0.1', '1.0'], \
        'Unknown VTU file version \'{}\'.'.format(root.attrib['version'])
    assert root.attrib['compressor'] == 'vtkZLibDataCompressor'

    header_type = (
        root.attrib['header_type'] if 'header_type' in root.attrib
        else 'UInt32'
        )

    byte_order = root.attrib['byte_order']
    assert byte_order in ['LittleEndian', 'BigEndian'], \
        'Unknown byte order \'{}\'.'.format(byte_order)

    grid = None
    appended_data = None
    for c in root.getchildren():
        if c.tag == 'UnstructuredGrid':
            assert grid is None, \
                'More than one UnstructuredGrid found.'
            grid = c
        else:
            assert c.tag == 'AppendedData', \
                'Unknown main tag \'{}\'.'.format(c.tag)
            assert appended_data is None, \
                'More than one AppendedData found.'
            appended_data = c

    assert grid is not None, \
        'No UnstructuredGrid found.'

    pieces = grid.getchildren()
    assert len(pieces) == 1

    piece = pieces[0]

    num_points = int(piece.attrib['NumberOfPoints'])

    points = None
    point_data = {}
    cell_data_raw = {}
    field_data = {}

    cells = {}

    vtu_to_numpy_type = {
        'Float64': numpy.float64,
        'Int64': numpy.int64,
        'UInt8': numpy.uint8,
        'UInt32': numpy.uint32,
        }

    for child in piece.getchildren():
        if child.tag == 'Points':
            c = child.getchildren()
            assert len(c) == 1
            c = c[0]
            assert c.tag == 'DataArray'
            assert c.attrib['Name'] == 'Points'

            if c.attrib['format'] == 'ascii':
                points = numpy.array(
                    c.text.split(),
                    dtype=vtu_to_numpy_type[c.attrib['type']]
                    )
            else:
                assert c.attrib['format'] == 'binary', \
                    'Unknown data format \'{}\'.'.format(c.attrib['format'])
                points = _read_binary(
                    c.text.strip(), byte_order, header_type, c.attrib['type']
                    )

            num_components = int(c.attrib['NumberOfComponents'])
            points = points.reshape(num_points, num_components)

        elif child.tag == 'Cells':
            for data in child.getchildren():
                assert data.tag == 'DataArray'

                if data.attrib['format'] == 'ascii':
                    cells[data.attrib['Name']] = numpy.array(
                        data.text.split(),
                        dtype=vtu_to_numpy_type[data.attrib['type']]
                        )
                else:
                    assert data.attrib['format'] == 'binary'
                    cells[data.attrib['Name']] = _read_binary(
                        data.text.strip(),
                        byte_order, header_type, data.attrib['type']
                        )

        elif child.tag == 'PointData':
            for c in child.getchildren():
                assert c.tag == 'DataArray'
                if c.attrib['format'] == 'ascii':
                    point_data[c.attrib['Name']] = numpy.array(
                        c.text.split(),
                        dtype=vtu_to_numpy_type[c.attrib['type']]
                        )
                else:
                    assert c.attrib['format'] == 'binary'
                    point_data[c.attrib['Name']] = _read_binary(
                        c.text.strip(),
                        byte_order, header_type, c.attrib['type']
                        )

        else:
            assert child.tag == 'CellData', \
                'Unknown tag \'{}\'.'.format(child.tag)
            for c in child.getchildren():
                assert c.tag == 'DataArray'
                if c.attrib['format'] == 'ascii':
                    cell_data_raw[c.attrib['Name']] = numpy.array(
                        c.text.split(),
                        dtype=vtu_to_numpy_type[c.attrib['type']]
                        )
                else:
                    assert c.attrib['format'] == 'binary'
                    cell_data_raw[c.attrib['Name']] = _read_binary(
                        c.text.strip(),
                        byte_order, header_type, c.attrib['type']
                        )

    assert points is not None
    assert 'connectivity' in cells
    assert 'offsets' in cells
    assert 'types' in cells

    cells = _cells_from_data(
            cells['connectivity'], cells['offsets'], cells['types']
            )

    # get point data in shape
    for key in point_data:
        if len(point_data[key]) != len(points):
            point_data[key] = point_data[key].reshape(len(points), -1)

    # get cell data in shape
    cell_data = cell_data_from_raw(cells, cell_data_raw)

    return points, cells, point_data, cell_data, field_data


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
