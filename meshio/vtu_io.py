# -*- coding: utf-8 -*-
#
'''
I/O for VTK, VTU, Exodus etc.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy

from .gmsh_io import num_nodes_per_cell


def _cells_from_data(connectivity, offsets, types):
    # create cells
    vtk_to_meshio_type = {
        1: 'vertex',
        3: 'line',
        5: 'triangle',
        9: 'quad',
        10: 'tetra',
        12: 'hexahedron',
        13: 'wedge',
        14: 'pyramid',
        21: 'line3',
        22: 'triangle6',
        23: 'quad8',
        24: 'tetra10',
        25: 'hexahedron20',
        }

    # assert (types == types[0]).all()

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


def read(filename):
    from lxml import etree as ET

    tree = ET.parse(filename)
    root = tree.getroot()

    assert root.tag == 'VTKFile'
    assert root.attrib['type'] == 'UnstructuredGrid'
    assert root.attrib['version'] == '0.1'
    assert root.attrib['byte_order'] == 'LittleEndian'
    assert root.attrib['header_type'] == 'UInt32'
    assert root.attrib['compressor'] == 'vtkZLibDataCompressor'

    children = root.getchildren()
    assert len(children) == 1
    grid = children[0]
    assert grid.tag == 'UnstructuredGrid'

    pieces = grid.getchildren()
    assert len(pieces) == 1

    piece = pieces[0]

    points = None
    point_data = {}
    cell_data = {}
    field_data = {}

    connectivity = None
    offsets = None
    types = None

    vtu_type_to_numpy_type = {
        'Float64': numpy.float64,
        'Int64': numpy.int64,
        'UInt8': numpy.uint8,
        }

    for child in piece.getchildren():
        if child.tag == 'PointData':
            c = child.getchildren()
            if c:
                assert len(c) == 1
                c = c[0]
                assert c.tag == 'DataArray'
                assert c.attrib['format'] == 'ascii'

                point_data[c.attrib['Name']] = numpy.array(
                    c.text.split(),
                    dtype=vtu_type_to_numpy_type[c.attrib['type']]
                    )

        elif child.tag == 'CellData':
            pass
        elif child.tag == 'Points':
            c = child.getchildren()
            assert len(c) == 1
            c = c[0]
            assert c.tag == 'DataArray'
            assert c.attrib['Name'] == 'Points'
            assert c.attrib['format'] == 'ascii'

            points = numpy.array(
                c.text.split(),
                dtype=vtu_type_to_numpy_type[c.attrib['type']]
                ).reshape(-1, int(c.attrib['NumberOfComponents']))

        else:
            assert child.tag == 'Cells', \
                'Unknown tag \'{}\'.'.format(child.tag)

            for data in child.getchildren():
                assert data.tag == 'DataArray'
                assert data.attrib['format'] == 'ascii'

                if data.attrib['Name'] == 'connectivity':
                    connectivity = numpy.array(
                        data.text.split(),
                        dtype=vtu_type_to_numpy_type[data.attrib['type']]
                        )
                elif data.attrib['Name'] == 'offsets':
                    offsets = numpy.array(
                        data.text.split(),
                        dtype=vtu_type_to_numpy_type[data.attrib['type']]
                        )
                else:
                    assert data.attrib['Name'] == 'types', \
                        'Unknown array \'{}\'.'.format(data.attrib['Name'])
                    types = numpy.array(
                        data.text.split(),
                        dtype=vtu_type_to_numpy_type[data.attrib['type']]
                        )

    assert points is not None
    assert connectivity is not None
    assert offsets is not None
    assert types is not None

    # get point_data in shape
    for key in point_data:
        if len(point_data[key]) != len(points):
            point_data[key] = point_data[key].reshape(len(points), -1)

    cells = _cells_from_data(connectivity, offsets, types)

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
