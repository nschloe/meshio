# -*- coding: utf-8 -*-
#
'''
I/O for XDMF.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import xml.etree.ElementTree as ET

import numpy

from .vtk_io import cell_data_from_raw


def read(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    assert root.tag == 'Xdmf'

    if root.attrib['Version'] == '2.2':
        return _read_xdmf2(root)

    assert root.attrib['Version'] == '3.0', \
        'Unknown XDMF version {}.'.format(root.attrib['Version'])

    return _read_xdmf3(root)


def _read_xdmf2(root):
    def read_data_item(data_item):
        dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
        data_type = data_item.attrib['NumberType']
        precision = data_item.attrib['Precision']
        assert data_item.attrib['Format'] == 'XML'

        return numpy.array(
            data_item.text.split(),
            dtype=_xdmf_to_numpy_type(data_type, precision)
            ).reshape(dims)

    domains = list(root)
    assert len(domains) == 1
    domain = domains[0]
    assert domain.tag == 'Domain'

    grids = list(domain)
    assert len(grids) == 1
    grid = grids[0]
    assert grid.tag == 'Grid'
    assert grid.attrib['GridType'] == 'Uniform'

    points = None
    cells = {}
    point_data = {}
    cell_data_raw = {}
    field_data = {}

    xdmf_to_meshio_type = {
        'Hexahedron': 'hexahedron',
        'Quadrilateral': 'quad',
        'Tetrahedron': 'tetra',
        'Triangle': 'triangle',
        }

    for c in grid:
        if c.tag == 'Topology':
            data_items = list(c)
            assert len(data_items) == 1
            meshio_type = xdmf_to_meshio_type[c.attrib['TopologyType']]
            cells[meshio_type] = read_data_item(data_items[0])

        elif c.tag == 'Geometry':
            assert c.attrib['GeometryType'] == 'XYZ'
            data_items = list(c)
            assert len(data_items) == 1
            points = read_data_item(data_items[0])

        else:
            assert c.tag == 'Attribute', \
                'Unknown section \'{}\'.'.format(c.tag)

            assert c.attrib['Active'] == '1'
            assert c.attrib['AttributeType'] == 'None'

            data_items = list(c)
            assert len(data_items) == 1

            data = read_data_item(data_items[0])

            name = c.attrib['Name']
            if c.attrib['Center'] == 'Node':
                point_data[name] = data
            else:
                assert c.attrib['Center'] == 'Cell'
                cell_data_raw[name] = data

    cell_data = cell_data_from_raw(cells, cell_data_raw)

    return points, cells, point_data, cell_data, field_data


# XDMF 2 writer
# def write(filetype,
#           filename,
#           points,
#           cells,
#           point_data=None,
#           cell_data=None,
#           field_data=None
#           ):
#     # pylint: disable=import-error
#     from .vtk_io import write as vtk_write
#     return vtk_write(
#         filetype, filename, points, cells,
#         point_data=point_data, cell_data=cell_data, field_data=field_data
#         )


def _xdmf_to_numpy_type(data_type, precision):
    if data_type == 'Int' and precision == '8':
        return numpy.int64

    assert data_type == 'Float' and precision == '8', \
        'Unknown XDMF type ({}, {}).'.format(data_type, precision)
    return numpy.float64


def translate_mixed_cells(data):
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (cell_type1, p0, p1, ... ,pk, cell_type2, p10, p11, ..., p1k, ...

    # http://www.xdmf.org/index.php/XDMF_Model_and_Format#Topology
    xdmf_idx_to_num_nodes = {
        1: 1,
        4: 3,
        5: 4,
        6: 4,
        7: 5,
        8: 6,
        9: 8,
        }

    xdmf_idx_to_meshio_type = {
        1: 'vertex',
        4: 'triangle',
        5: 'quad',
        6: 'tetra',
        7: 'pyramid',
        8: 'wedge',
        9: 'hexahedron',
        }

    # collect types and offsets
    types = []
    offsets = []
    r = 0
    while r < len(data):
        types.append(data[r])
        offsets.append(r)
        r += xdmf_idx_to_num_nodes[data[r]] + 1

    offsets = numpy.array(offsets)

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

    cells = {}
    for tpe, b in bins.items():
        meshio_type = xdmf_idx_to_meshio_type[tpe]
        assert (data[offsets[b]] == tpe).all()
        n = xdmf_idx_to_num_nodes[tpe]
        indices = numpy.array([
            numpy.arange(1, n+1) + o for o in offsets[b]
            ])
        cells[meshio_type] = data[indices]

    return cells


def _read_xdmf3(root):
    def read_data_item(data_item):
        dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
        data_type = data_item.attrib['DataType']
        precision = data_item.attrib['Precision']
        assert data_item.attrib['Format'] == 'XML'

        return numpy.array(
            data_item.text.split(),
            dtype=_xdmf_to_numpy_type(data_type, precision)
            ).reshape(dims)

    domains = list(root)
    assert len(domains) == 1
    domain = domains[0]
    assert domain.tag == 'Domain'

    grids = list(domain)
    assert len(grids) == 1
    grid = grids[0]
    assert grid.tag == 'Grid'

    points = None
    cells = {}
    point_data = {}
    cell_data_raw = {}
    field_data = {}

    xdmf_to_meshio_type = {
        'Polyvertex': 'vertex',
        'Triangle': 'triangle',
        'Quadrilateral': 'quad',
        'Tetrahedron': 'tetra',
        'Pyramid': 'pyramid',
        'Wedge': 'wedge',
        'Hexahedron': 'hexahedron',
        'Edge_3': 'line3',
        'Tri_6': 'triangle6',
        'Quad_8': 'quad8',
        'Tet_10': 'tetra10',
        'Pyramid_13': 'pyramid13',
        'Wedge_15': 'wedge15',
        'Hex_20': 'hexahedron20',
        }

    for c in grid:
        if c.tag == 'Topology':
            data_items = list(c)
            assert len(data_items) == 1
            data_item = data_items[0]

            data = read_data_item(data_item)

            if c.attrib['Type'] == 'Mixed':
                cells = translate_mixed_cells(data)
            else:
                meshio_type = xdmf_to_meshio_type[c.attrib['Type']]
                cells[meshio_type] = data

        elif c.tag == 'Geometry':
            assert c.attrib['Type'] == 'XYZ'
            data_items = list(c)
            assert len(data_items) == 1
            data_item = data_items[0]
            points = read_data_item(data_item)

        else:
            assert c.tag == 'Attribute', \
                'Unknown section \'{}\'.'.format(c.tag)

            assert c.attrib['Type'] == 'None'

            data_items = list(c)
            assert len(data_items) == 1
            data_item = data_items[0]

            data = read_data_item(data_item)

            name = c.attrib['Name']
            if c.attrib['Center'] == 'Node':
                point_data[name] = data
            else:
                assert c.attrib['Center'] == 'Cell'
                cell_data_raw[name] = data

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
