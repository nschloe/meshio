# -*- coding: utf-8 -*-
#
'''
I/O for XDMF.
http://www.xdmf.org/index.php/XDMF_Model_and_Format

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import os
try:
    from StringIO import cStringIO as BytesIO
except ImportError:
    from io import BytesIO
import xml.etree.cElementTree as ET

import numpy

from .gmsh_io import cell_data_from_raw
from .vtk_io import raw_from_cell_data
from .vtu_io import write_xml


def read(filename):
    return XdmfReader(filename).read()


numpy_to_xdmf_dtype = {
    numpy.dtype(numpy.int32): ('Int', '4'),
    numpy.dtype(numpy.int64): ('Int', '8'),
    numpy.dtype(numpy.uint32): ('UInt', '4'),
    numpy.dtype(numpy.uint64): ('UInt', '8'),
    numpy.dtype(numpy.float32): ('Float', '4'),
    numpy.dtype(numpy.float64): ('Float', '8'),
    }
xdmf_to_numpy_type = {v: k for k, v in numpy_to_xdmf_dtype.items()}


dtype_to_format_string = {
    numpy.dtype(numpy.int32): '%d',
    numpy.dtype(numpy.int64): '%d',
    numpy.dtype(numpy.uint32): '%d',
    numpy.dtype(numpy.uint64): '%d',
    numpy.dtype(numpy.float32): '%.7e',
    numpy.dtype(numpy.float64): '%.15e',
    }


# Check out
# <https://gitlab.kitware.com/xdmf/xdmf/blob/master/XdmfTopologyType.cpp>
# for the list of indices.
xdmf_idx_to_meshio_type = {
    0x1: 'vertex',
    0x2: 'line',
    0x4: 'triangle',
    0x5: 'quad',
    0x6: 'tetra',
    0x7: 'pyramid',
    0x8: 'wedge',
    0x9: 'hexahedron',
    0x22: 'line3',
    0x23: 'quad9',
    0x24: 'triangle6',
    0x25: 'quad8',
    0x26: 'tetra10',
    0x27: 'pyramid13',
    0x28: 'wedge15',
    0x29: 'wedge18',
    0x30: 'hexahedron20',
    0x31: 'hexahedron24',
    0x32: 'hexahedron27',
    0x33: 'hexahedron64',
    0x34: 'hexahedron125',
    0x35: 'hexahedron216',
    0x36: 'hexahedron343',
    0x37: 'hexahedron512',
    0x38: 'hexahedron729',
    0x39: 'hexahedron1000',
    0x40: 'hexahedron1331',
    # 0x41: 'hexahedron_spectral_64',
    # 0x42: 'hexahedron_spectral_125',
    # 0x43: 'hexahedron_spectral_216',
    # 0x44: 'hexahedron_spectral_343',
    # 0x45: 'hexahedron_spectral_512',
    # 0x46: 'hexahedron_spectral_729',
    # 0x47: 'hexahedron_spectral_1000',
    # 0x48: 'hexahedron_spectral_1331',
    }
meshio_type_to_xdmf_index = {v: k for k, v in xdmf_idx_to_meshio_type.items()}

# See
# <http://www.xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
# for XDMF types.
# There appears to be no particular consistency, so allow for different
# alternatives as well.
meshio_to_xdmf_type = {
    'vertex': ['Polyvertex'],
    'line': ['Polyline'],
    'triangle': ['Triangle'],
    'quad': ['Quadrilateral'],
    'tetra': ['Tetrahedron'],
    'pyramid': ['Pyramid'],
    'wedge': ['Wedge'],
    'hexahedron': ['Hexahedron'],
    'line3': ['Edge_3'],
    'triangle6': ['Triangle_6', 'Tri_6'],
    'quad8': ['Quadrilateral_8', 'Quad_8'],
    'tetra10': ['Tetrahedron_10', 'Tet_10'],
    'pyramid13': ['Pyramid_13'],
    'wedge15': ['Wedge_15'],
    'hexahedron20': ['Hexahedron_20', 'Hex_20'],
    }
xdmf_to_meshio_type = {
    v: k
    for k, vals in meshio_to_xdmf_type.items()
    for v in vals
    }


def _translate_mixed_cells(data):
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (cell_type1, p0, p1, ... ,pk, cell_type2, p10, p11, ..., p1k, ...

    # http://www.xdmf.org/index.php/XDMF_Model_and_Format#Topology
    # https://gitlab.kitware.com/xdmf/xdmf/blob/master/XdmfTopologyType.hpp#L394
    xdmf_idx_to_num_nodes = {
        1: 1,  # vertex
        4: 3,  # triangle
        5: 4,  # quad
        6: 4,  # tet
        7: 5,  # pyramid
        8: 6,  # wedge
        9: 8,  # hex
        11: 6,  # triangle6
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

    def read_data_item(self, data_item):
        import h5py
        dims = [int(d) for d in data_item.attrib['Dimensions'].split()]

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files
        # out there use both keys interchangeably.
        if 'DataType' in data_item.attrib:
            assert 'NumberType' not in data_item.attrib
            data_type = data_item.attrib['DataType']
        elif 'NumberType' in data_item.attrib:
            assert 'DataType' not in data_item.attrib
            data_type = data_item.attrib['NumberType']
        else:
            # Default, see
            # <http://www.xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
            data_type = 'Float'

        try:
            precision = data_item.attrib['Precision']
        except KeyError:
            precision = '4'

        if data_item.attrib['Format'] == 'XML':
            return numpy.array(
                data_item.text.split(),
                dtype=xdmf_to_numpy_type[(data_type, precision)]
                ).reshape(dims)
        elif data_item.attrib['Format'] == 'Binary':
            return numpy.fromfile(
                data_item.text.strip(),
                dtype=xdmf_to_numpy_type[(data_type, precision)]
                ).reshape(dims)

        assert data_item.attrib['Format'] == 'HDF', \
            'Unknown XDMF Format \'{}\'.'.format(data_item.attrib['Format'])

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

        try:
            assert grid.attrib['GridType'] == 'Uniform'
        except KeyError:
            # The default is 'Uniform'
            pass

        points = None
        cells = {}
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
            if c.tag == 'Topology':
                data_items = list(c)
                assert len(data_items) == 1
                meshio_type = xdmf_to_meshio_type[c.attrib['TopologyType']]
                cells[meshio_type] = self.read_data_item(data_items[0])

            elif c.tag == 'Geometry':
                try:
                    assert c.attrib['GeometryType'] == 'XYZ'
                except KeyError:
                    # The default is 'XYZ'
                    pass
                data_items = list(c)
                assert len(data_items) == 1
                points = self.read_data_item(data_items[0])

            else:
                assert c.tag == 'Attribute', \
                    'Unknown section \'{}\'.'.format(c.tag)

                # assert c.attrib['Active'] == '1'
                # assert c.attrib['AttributeType'] == 'None'

                data_items = list(c)
                assert len(data_items) == 1

                data = self.read_data_item(data_items[0])

                name = c.attrib['Name']
                if c.attrib['Center'] == 'Node':
                    point_data[name] = data
                elif c.attrib['Center'] == 'Cell':
                    cell_data_raw[name] = data
                else:
                    # TODO field data?
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

                # The XDMF2 key is `TopologyType`, just `Type` for XDMF3.
                # Allow both.
                if 'Type' in c.attrib:
                    assert 'TopologyType' not in c.attrib
                    cell_type = c.attrib['Type']
                else:
                    cell_type = c.attrib['TopologyType']

                if cell_type == 'Mixed':
                    cells = _translate_mixed_cells(data)
                else:
                    meshio_type = xdmf_to_meshio_type[cell_type]
                    cells[meshio_type] = data

            elif c.tag == 'Geometry':
                try:
                    geometry_type = c.attrib['GeometryType']
                except KeyError:
                    geometry_type = 'XYZ'

                data_items = list(c)
                assert len(data_items) == 1
                data_item = data_items[0]
                points = self.read_data_item(data_item)

                if geometry_type == 'XY':
                    points = numpy.column_stack([
                        points, numpy.zeros(len(points))
                        ])

            else:
                assert c.tag == 'Attribute', \
                    'Unknown section \'{}\'.'.format(c.tag)

                # Don't be too struct here: FEniCS, for example, calls this
                # 'AttributeType'.
                # assert c.attrib['Type'] == 'None'

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


class XdmfWriter(object):
    def __init__(self,
                 filename,
                 points,
                 cells,
                 point_data=None,
                 cell_data=None,
                 field_data=None,
                 pretty_xml=True,
                 data_format='HDF'):
        assert data_format in ['XML', 'Binary', 'HDF'], (
            'Unknown XDMF data format '
            '\'{}\' (use \'XML\', \'Binary\', or \'HDF\'.)'.format(data_format)
            )

        point_data = {} if point_data is None else point_data
        cell_data = {} if cell_data is None else cell_data
        field_data = {} if field_data is None else field_data

        self.filename = filename
        self.data_format = data_format
        self.data_counter = 0

        if data_format == 'HDF':
            import h5py
            self.h5_filename = os.path.splitext(self.filename)[0] + '.h5'
            self.h5_file = h5py.File(self.h5_filename, 'w')

        xdmf_file = ET.Element('Xdmf', Version='3.0')

        domain = ET.SubElement(xdmf_file, 'Domain')
        grid = ET.SubElement(domain, 'Grid', Name='Grid')

        self.points(grid, points)
        self.cells(cells, grid)
        self.point_data(point_data, grid)
        self.cell_data(cell_data, grid)

        ET.register_namespace('xi', 'https://www.w3.org/2001/XInclude/')

        write_xml(filename, xdmf_file, pretty_xml, indent=2)
        return

    def numpy_to_xml_string(self, data):
        if self.data_format == 'XML':
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype]
            numpy.savetxt(s, data.flatten(), fmt)
            return s.getvalue().decode()
        elif self.data_format == 'Binary':
            bin_filename = '{}{}.bin'.format(
                os.path.splitext(self.filename)[0],
                self.data_counter,
                )
            self.data_counter += 1
            # write binary data to file
            with open(bin_filename, 'wb') as f:
                data.tofile(f)
            return bin_filename

        assert self.data_format == 'HDF'
        name = 'data{}'.format(self.data_counter)
        self.data_counter += 1
        self.h5_file.create_dataset(name, data=data)
        return self.h5_filename + ':/' + name

    def points(self, grid, points):
        geo = ET.SubElement(grid, 'Geometry', Type='XYZ')
        dt, prec = numpy_to_xdmf_dtype[points.dtype]
        dim = '{} {}'.format(*points.shape)
        data_item = ET.SubElement(
            geo, 'DataItem',
            DataType=dt, Dimensions=dim,
            Format=self.data_format, Precision=prec
            )
        data_item.text = self.numpy_to_xml_string(points)
        return

    def cells(self, cells, grid):
        if len(cells) == 1:
            meshio_type = list(cells.keys())[0]
            xdmf_type = meshio_to_xdmf_type[meshio_type][0]
            topo = ET.SubElement(grid, 'Topology', Type=xdmf_type)
            dt, prec = numpy_to_xdmf_dtype[cells[meshio_type].dtype]
            dim = '{} {}'.format(*cells[meshio_type].shape)
            data_item = ET.SubElement(
                topo, 'DataItem',
                DataType=dt, Dimensions=dim,
                Format=self.data_format, Precision=prec
                )
            data_item.text = \
                self.numpy_to_xml_string(cells[meshio_type])
        elif len(cells) > 1:
            topo = ET.SubElement(grid, 'Topology', Type='Mixed')
            total_num_cells = sum(c.shape[0] for c in cells.values())
            total_num_cell_items = \
                sum(numpy.prod(c.shape) for c in cells.values())
            dim = str(total_num_cell_items + total_num_cells)
            # Lines translate to Polylines, and one needs to specify the exact
            # number of nodes. Hence, prepend 2.
            if 'line' in cells:
                cells['line'] = numpy.insert(cells['line'], 0, 2, axis=1)
            cd = numpy.concatenate([
                # prepend column with xdmf type index
                numpy.insert(
                    value, 0, meshio_type_to_xdmf_index[key], axis=1
                    ).flatten()
                for key, value in cells.items()
                ])
            dt, prec = numpy_to_xdmf_dtype[cd.dtype]
            data_item = ET.SubElement(
                topo, 'DataItem',
                DataType=dt, Dimensions=dim,
                Format=self.data_format, Precision=prec
                )
            data_item.text = self.numpy_to_xml_string(cd)
        return

    def point_data(self, point_data, grid):
        for name, data in point_data.items():
            att = ET.SubElement(
                grid, 'Attribute',
                Name=name, Type='None', Center='Node'
                )
            dt, prec = numpy_to_xdmf_dtype[data.dtype]
            dim = ' '.join([str(s) for s in data.shape])
            data_item = ET.SubElement(
                att, 'DataItem',
                DataType=dt, Dimensions=dim,
                Format=self.data_format, Precision=prec
                )
            data_item.text = self.numpy_to_xml_string(data)
        return

    def cell_data(self, cell_data, grid):
        raw = raw_from_cell_data(cell_data)
        for name, data in raw.items():
            att = ET.SubElement(
                grid, 'Attribute',
                Name=name, Type='None', Center='Cell'
                )
            dt, prec = numpy_to_xdmf_dtype[data.dtype]
            dim = ' '.join([str(s) for s in data.shape])
            data_item = ET.SubElement(
                att, 'DataItem',
                DataType=dt, Dimensions=dim,
                Format=self.data_format, Precision=prec
                )
            data_item.text = self.numpy_to_xml_string(data)
        return


def write(*args, **kwargs):
    XdmfWriter(*args, **kwargs)
    return
