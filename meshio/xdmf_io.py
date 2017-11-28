# -*- coding: utf-8 -*-
#
'''
I/O for XDMF.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy

from .vtk_io import cell_data_from_raw


def read(filetype, filename):
    from lxml import etree as ET

    tree = ET.parse(filename)
    root = tree.getroot()

    assert root.tag == 'Xdmf'
    assert root.attrib['Version'] == '2.2'

    domains = root.getchildren()
    assert len(domains) == 1
    domain = domains[0]
    assert domain.tag == 'Domain'

    grids = domain.getchildren()
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

    def xdmf_to_numpy_type(number_type, precision):
        if number_type == 'Int' and precision == '8':
            return numpy.int64

        assert number_type == 'Float' and precision == '8', \
            'Unknown XDMF type ({}, {}).'.format(number_type, precision)
        return numpy.float64

    for c in grid.getchildren():
        if c.tag == 'Topology':
            meshio_type = xdmf_to_meshio_type[c.attrib['TopologyType']]

            data_items = c.getchildren()
            assert len(data_items) == 1
            data_item = data_items[0]

            dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
            number_type = data_item.attrib['NumberType']
            precision = data_item.attrib['Precision']
            assert data_item.attrib['Format'] == 'XML'

            cells[meshio_type] = numpy.array(
                data_item.text.split(),
                dtype=xdmf_to_numpy_type(number_type, precision)
                ).reshape(dims)

        elif c.tag == 'Geometry':
            assert c.attrib['GeometryType'] == 'XYZ'

            data_items = c.getchildren()
            assert len(data_items) == 1
            data_item = data_items[0]

            dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
            number_type = data_item.attrib['NumberType']
            precision = data_item.attrib['Precision']
            assert data_item.attrib['Format'] == 'XML'

            points = numpy.array(
                data_item.text.split(),
                dtype=xdmf_to_numpy_type(number_type, precision)
                ).reshape(dims)

        else:
            assert c.tag == 'Attribute', \
                'Unknown section \'{}\'.'.format(c.tag)

            assert c.attrib['Active'] == '1'
            assert c.attrib['AttributeType'] == 'None'

            data_items = c.getchildren()
            assert len(data_items) == 1
            data_item = data_items[0]

            dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
            number_type = data_item.attrib['NumberType']
            precision = data_item.attrib['Precision']
            assert data_item.attrib['Format'] == 'XML'

            data = numpy.array(
                data_item.text.split(),
                dtype=xdmf_to_numpy_type(number_type, precision)
                ).reshape(dims)

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
