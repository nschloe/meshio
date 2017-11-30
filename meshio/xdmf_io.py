# -*- coding: utf-8 -*-
#
'''
I/O for XDMF.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import xml.etree.ElementTree as ET

import numpy

from .vtk_io import cell_data_from_raw
from .xdmf3_io import xdmf_to_numpy_type


def read_data_item(data_item):
    dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
    data_type = data_item.attrib['NumberType']
    precision = data_item.attrib['Precision']
    assert data_item.attrib['Format'] == 'XML'

    return numpy.array(
        data_item.text.split(),
        dtype=xdmf_to_numpy_type(data_type, precision)
        ).reshape(dims)


def read(filetype, filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    assert root.tag == 'Xdmf'
    assert root.attrib['Version'] == '2.2'

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
