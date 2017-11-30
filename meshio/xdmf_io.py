# -*- coding: utf-8 -*-
#
'''
I/O for XDMF.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import os
import xml.etree.ElementTree as ET

import h5py
import numpy

from .vtk_io import cell_data_from_raw


def read(filename):
    return XdmfReader(filename).read()


def _xdmf_to_numpy_type(data_type, precision):
    if data_type == 'Int' and precision == '4':
        return numpy.int32
    elif data_type == 'Int' and precision == '8':
        return numpy.int64
    elif data_type == 'Float' and precision == '4':
        return numpy.float32

    assert data_type == 'Float' and precision == '8', \
        'Unknown XDMF type ({}, {}).'.format(data_type, precision)
    return numpy.float64


def _translate_mixed_cells(data):
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


class XdmfReader(object):
    def __init__(self, filename):
        self.filename = filename
        self.xdmf_to_meshio_type = {
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

        return

    def read(self):
        tree = ET.parse(self.filename)
        root = tree.getroot()

        assert root.tag == 'Xdmf'

        version = root.attrib['Version']

        if version.split('.')[0] == '2':
            return self.read_xdmf2(root)

        assert version.split('.')[0] == '3', \
            'Unknown XDMF version {}.'.format(version)

        return self.read_xdmf3(root)

    def read_data_item(self, data_item, dt_key='DataType'):
        dims = [int(d) for d in data_item.attrib['Dimensions'].split()]
        data_type = data_item.attrib[dt_key]
        precision = data_item.attrib['Precision']

        if data_item.attrib['Format'] == 'XML':
            return numpy.array(
                data_item.text.split(),
                dtype=_xdmf_to_numpy_type(data_type, precision)
                ).reshape(dims)

        assert data_item.attrib['Format'] == 'HDF', \
            'Unknown XDMF Format \'{}\'.'.format(
                    data_item.attrib['Format']
                    )

        info = data_item.text.strip()
        filename, h5path = info.split(':')

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        full_hdf5_path = os.path.join(
                os.path.dirname(self.filename),
                filename
                )

        f = h5py.File(full_hdf5_path, 'r')
        assert h5path[0] == '/'

        for key in h5path[1:].split('/'):
            f = f[key]
        # `[()]` gives a numpy.ndarray
        return f[()]

    def read_xdmf2(self, root):
        domains = list(root)
        assert len(domains) == 1
        domain = domains[0]
        assert domain.tag == 'Domain'

        grids = list(domain)
        assert len(grids) == 1, \
            'XDMF reader: Only supports one grid right now.'
        grid = grids[0]
        assert grid.tag == 'Grid'
        assert grid.attrib['GridType'] == 'Uniform'

        points = None
        cells = {}
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
            if c.tag == 'Topology':
                data_items = list(c)
                assert len(data_items) == 1
                meshio_type = \
                    self.xdmf_to_meshio_type[c.attrib['TopologyType']]
                cells[meshio_type] = self.read_data_item(
                    data_items[0], dt_key='NumberType'
                    )

            elif c.tag == 'Geometry':
                assert c.attrib['GeometryType'] == 'XYZ'
                data_items = list(c)
                assert len(data_items) == 1
                points = self.read_data_item(
                        data_items[0], dt_key='NumberType'
                        )

            else:
                assert c.tag == 'Attribute', \
                    'Unknown section \'{}\'.'.format(c.tag)

                # assert c.attrib['Active'] == '1'
                # assert c.attrib['AttributeType'] == 'None'

                data_items = list(c)
                assert len(data_items) == 1

                data = self.read_data_item(data_items[0], dt_key='NumberType')

                name = c.attrib['Name']
                if c.attrib['Center'] == 'Node':
                    point_data[name] = data
                elif c.attrib['Center'] == 'Cell':
                    cell_data_raw[name] = data
                else:
                    # TODO
                    assert c.attrib['Center'] == 'Grid'

        cell_data = cell_data_from_raw(cells, cell_data_raw)

        return points, cells, point_data, cell_data, field_data

    def read_xdmf3(self, root):
        domains = list(root)
        assert len(domains) == 1
        domain = domains[0]
        assert domain.tag == 'Domain'

        grids = list(domain)
        assert len(grids) == 1, \
            'XDMF reader: Only supports one grid right now.'
        grid = grids[0]
        assert grid.tag == 'Grid'

        points = None
        cells = {}
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
            if c.tag == 'Topology':
                data_items = list(c)
                assert len(data_items) == 1
                data_item = data_items[0]

                data = self.read_data_item(data_item)

                if c.attrib['Type'] == 'Mixed':
                    cells = _translate_mixed_cells(data)
                else:
                    meshio_type = self.xdmf_to_meshio_type[c.attrib['Type']]
                    cells[meshio_type] = data

            elif c.tag == 'Geometry':
                assert c.attrib['Type'] == 'XYZ'
                data_items = list(c)
                assert len(data_items) == 1
                data_item = data_items[0]
                points = self.read_data_item(data_item)

            else:
                assert c.tag == 'Attribute', \
                    'Unknown section \'{}\'.'.format(c.tag)

                assert c.attrib['Type'] == 'None'

                data_items = list(c)
                assert len(data_items) == 1
                data_item = data_items[0]

                data = self.read_data_item(data_item)

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


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    # pylint: disable=import-error
    from .vtk_io import write as vtk_write
    return vtk_write(
        'xdmf3', filename, points, cells,
        point_data=point_data, cell_data=cell_data, field_data=field_data
        )
