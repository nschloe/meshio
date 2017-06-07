# -*- coding: utf-8 -*-
#
'''
I/O for DOLFIN's XML format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/dolfin_xml/dolfin_xml.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging
import os
import re

import numpy


def _read_mesh(filename):
    from lxml import etree as ET

    tree = ET.parse(filename)
    root = tree.getroot()
    mesh = root.getchildren()[0]
    assert mesh.tag == 'mesh'

    dolfin_to_meshio_type = {
        'triangle': ('triangle', 3),
        'tetrahedron': ('tetra', 4),
        }

    cell_type, npc = dolfin_to_meshio_type[mesh.attrib['celltype']]

    is_2d = mesh.attrib['dim'] == '2'
    if not is_2d:
        assert mesh.attrib['dim'] == '3'

    points = None
    cells = {
        cell_type: None
        }

    for child in mesh.getchildren():
        if child.tag == 'vertices':
            num_verts = int(child.attrib['size'])
            points = numpy.empty((num_verts, 3))
            for vert in child.getchildren():
                assert vert.tag == 'vertex'
                idx = int(vert.attrib['index'])
                points[idx, 0] = vert.attrib['x']
                points[idx, 1] = vert.attrib['y']
                if is_2d:
                    points[idx, 2] = 0.0
                else:
                    points[idx, 2] = vert.attrib['z']

        else:
            assert child.tag == 'cells', \
                'Unknown entry \'{}\'.'.format(child.tag)
            num_cells = int(child.attrib['size'])
            cells[cell_type] = numpy.empty((num_cells, npc), dtype=int)
            for cell in child.getchildren():
                assert dolfin_to_meshio_type[cell.tag][0] == cell_type
                idx = int(cell.attrib['index'])
                for k in range(npc):
                    cells[cell_type][idx, k] = cell.attrib['v{}'.format(k)]

    return points, cells, cell_type


def _read_cell_data(filename, cell_type):
    from lxml import etree as ET
    dolfin_type_to_numpy_type = {
        'int': numpy.dtype('int'),
        'float': numpy.dtype('float'),
        'uint': numpy.dtype('uint'),
        }

    cell_data = {cell_type: {}}
    dir_name = os.path.dirname(filename)
    if not os.path.dirname(filename):
        dir_name = os.getcwd()

    # Loop over all files in the same directory as `filename`.
    basename = os.path.splitext(os.path.basename(filename))[0]
    for f in os.listdir(dir_name):
        # Check if there are files by the name "<filename>_*.xml"; if yes,
        # extract the * pattern and make it the name of the data set.
        out = re.match('{}_([^\\.]+)\\.xml'.format(basename), f)
        if not out:
            continue
        name = out.group(1)
        tree = ET.parse(f)
        root = tree.getroot()
        mesh_function = root.getchildren()[0]
        assert mesh_function.tag == 'mesh_function'
        size = int(mesh_function.attrib['size'])
        dtype = dolfin_type_to_numpy_type[mesh_function.attrib['type']]
        data = numpy.empty(size, dtype=dtype)
        for child in mesh_function.getchildren():
            assert child.tag == 'entity'
            idx = int(child.attrib['index'])
            data[idx] = child.attrib['value']

        cell_data[cell_type][name] = data
    return cell_data


def read(filename):
    points, cells, cell_type = _read_mesh(filename)
    point_data = {}
    cell_data = _read_cell_data(filename, cell_type)
    field_data = {}
    return points, cells, point_data, cell_data, field_data


def _write_mesh(
        filename,
        points,
        cell_type,
        cells
        ):
    from lxml import etree as ET

    stripped_cells = {cell_type: cells[cell_type]}

    dolfin = ET.Element(
        'dolfin',
        nsmap={'dolfin': 'http://fenicsproject.org/'}
        )

    meshio_to_dolfin_type = {
            'triangle': 'triangle',
            'tetra': 'tetrahedron',
            }

    if len(cells) > 1:
        discarded_cells = cells.keys()
        discarded_cells.remove(cell_type)
        logging.warning(
          'DOLFIN XML can only handle one cell type at a time. '
          'Using %s, discarding %s.',
          cell_type, ', '.join(discarded_cells)
          )

    dim = 2 if all(points[:, 2] == 0) else 3

    mesh = ET.SubElement(
            dolfin,
            'mesh',
            celltype=meshio_to_dolfin_type[cell_type],
            dim=str(dim)
            )
    vertices = ET.SubElement(mesh, 'vertices', size=str(len(points)))

    for k, point in enumerate(points):
        ET.SubElement(
            vertices,
            'vertex',
            index=str(k),
            x=repr(point[0]),
            y=repr(point[1]),
            z=repr(point[2])
            )

    num_cells = 0
    for cls in stripped_cells.values():
        num_cells += len(cls)

    xcells = ET.SubElement(mesh, 'cells', size=str(num_cells))
    idx = 0
    for ct, cls in stripped_cells.items():
        for cell in cls:
            cell_entry = ET.SubElement(
                xcells,
                meshio_to_dolfin_type[ct],
                index=str(idx)
                )
            for k, c in enumerate(cell):
                cell_entry.attrib['v{}'.format(k)] = str(c)
            idx += 1

    tree = ET.ElementTree(dolfin)
    tree.write(filename, pretty_print=True)
    return


def _numpy_type_to_dolfin_type(dtype):
    types = ['int', 'uint', 'float']
    for t in types:
        # issubtype handles all of int8, int16, float64 etc.
        if numpy.issubdtype(dtype, numpy.dtype(t)):
            return t
    return


def _write_cell_data(
        filename,
        dim,
        cell_data
        ):
    from lxml import etree as ET

    dolfin = ET.Element(
        'dolfin',
        nsmap={'dolfin': 'http://fenicsproject.org/'}
        )

    mesh_function = ET.SubElement(
            dolfin,
            'mesh_function',
            type=_numpy_type_to_dolfin_type(cell_data.dtype),
            dim=str(dim),
            size=str(len(cell_data))
            )

    for k, value in enumerate(cell_data):
        ET.SubElement(
            mesh_function,
            'entity',
            index=str(k),
            value=repr(value),
            )

    tree = ET.ElementTree(dolfin)
    tree.write(filename, pretty_print=True)
    return


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    if 'tetra' in cells:
        cell_type = 'tetra'
    else:
        assert 'triangle' in cells
        cell_type = 'triangle'

    _write_mesh(filename, points, cell_type, cells)

    if cell_type in cell_data:
        for key, data in cell_data[cell_type].items():
            cell_data_filename = \
                '{}_{}.xml'.format(os.path.splitext(filename)[0], key)
            dim = 2 if all(points[:, 2] == 0) else 3
            _write_cell_data(cell_data_filename, dim, numpy.array(data))
    return
