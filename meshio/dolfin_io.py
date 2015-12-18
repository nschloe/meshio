# -*- coding: utf-8 -*-
#
'''
I/O for DOLFIN's XML format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/dolfin_xml/dolfin_xml.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy
# import xml.etree.ElementTree as ET
from lxml import etree as ET


def read(filename):

    tree = ET.parse(filename)
    root = tree.getroot()
    mesh = root.getchildren()[0]
    assert(mesh.tag == 'mesh')

    cell_type = mesh.attrib['celltype']
    nodes_per_cell = {
        'triangle': 3,
        'quad': 4
        }
    npc = nodes_per_cell[cell_type]

    points = None
    cells = {
        cell_type: None
        }
    print(cells)
    for child in mesh.getchildren():
        if child.tag == 'vertices':
            num_verts = int(child.attrib['size'])
            points = numpy.empty((num_verts, 3))
            for vert in child.getchildren():
                assert(vert.tag == 'vertex')
                idx = int(vert.attrib['index'])
                points[idx, 0] = vert.attrib['x']
                points[idx, 1] = vert.attrib['y']
                points[idx, 2] = vert.attrib['z']

        elif child.tag == 'cells':
            num_cells = int(child.attrib['size'])
            cells[cell_type] = numpy.empty((num_cells, npc), dtype=int)
            for cell in child.getchildren():
                assert(cell.tag == cell_type)
                idx = int(cell.attrib['index'])
                for k in range(npc):
                    cells[cell_type][idx, k] = cell.attrib['v%s' % k]

        else:
            raise RuntimeError('Unknown entry \'%s\'.' % child.tag)

    print(cells)
    point_data = []
    cell_data = []
    field_data = []
    return points, cells, point_data, cell_data, field_data


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    if point_data is None:
        point_data = {}
    if cell_data is None:
        cell_data = {}
    if field_data is None:
        field_data = {}

    dolfin = ET.Element('dolfin')
    dolfin.attrib['xmlns_dolfin'] = 'http://www.fenics.org/dolfin/'

    if len(cells) > 1:
        raise RuntimeError(
          'Dolfin XML can only deal with one cell type at a time.'
          )

    cell_type = cells.keys()[0]

    if all(points[:, 2] == 0):
        dim = '2'
    else:
        dim = '3'

    mesh = ET.SubElement(dolfin, 'mesh', celltype=cell_type, dim=dim)
    vertices = ET.SubElement(mesh, 'vertices', size=str(len(points)))
    for k, point in enumerate(points):
        ET.SubElement(
            vertices,
            'vertex',
            index=str(k),
            x=str(point[0]), y=str(point[1]), z=str(point[2])
            )

    num_cells = 0
    for cls in cells.values():
        num_cells += len(cls)

    xcells = ET.SubElement(mesh, 'cells', size=str(num_cells))
    idx = 0
    for cell_type, cls in cells.iteritems():
        for cell in cls:
            cell_entry = ET.SubElement(
                xcells,
                cell_type,
                index=str(idx)
                )
            for k, c in enumerate(cell):
                cell_entry.attrib['v%d' % k] = str(c)
            idx += 1

    tree = ET.ElementTree(dolfin)
    tree.write(filename, pretty_print=True)
    return
