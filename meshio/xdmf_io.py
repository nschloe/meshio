# -*- coding: utf-8 -*-
#
'''
I/O for XDMF.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy

from .vtk_io import translate_cells

# Make explicit copies of the data; some (all?) of it is quite volatile and
# contains garbage once the vtk_mesh goes out of scopy.


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

    xdmf_to_meshio_type = {
        'Triangle': 'triangle'
        }

    def xdmf_to_numpy_type(number_type, precision):
        if number_type == 'Int' and precision == '8':
            return numpy.int64
        return None

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

        else:
            assert c.tag == 'Geometry', \
                'Unknown section '

            exit(1)

    point_data = None
    cell_data = None
    field_data = None
    exit(1)

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
