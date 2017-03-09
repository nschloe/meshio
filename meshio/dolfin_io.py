# -*- coding: utf-8 -*-
#
'''
I/O for DOLFIN's XML format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/dolfin_xml/dolfin_xml.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy
import warnings


def read(filename):
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

        elif child.tag == 'cells':
            num_cells = int(child.attrib['size'])
            cells[cell_type] = numpy.empty((num_cells, npc), dtype=int)
            for cell in child.getchildren():
                assert(dolfin_to_meshio_type[cell.tag][0] == cell_type)
                idx = int(cell.attrib['index'])
                for k in range(npc):
                    cells[cell_type][idx, k] = cell.attrib['v%s' % k]

        else:
            raise RuntimeError('Unknown entry \'%s\'.' % child.tag)

    point_data = {}
    cell_data = {}
    field_data = {}
    return points, cells, point_data, cell_data, field_data


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    from lxml import etree as ET

    if point_data is None:
        point_data = {}
    if cell_data is None:
        cell_data = {}
    if field_data is None:
        field_data = {}

    dolfin = ET.Element(
        'dolfin',
        nsmap={'dolfin': 'http://fenicsproject.org/'}
        )

    meshio_to_dolfin_type = {
            'triangle': 'triangle',
            'tetra': 'tetrahedron',
            }

    if 'tetra' in cells:
        stripped_cells = {'tetra': cells['tetra']}
        cell_type = 'tetra'
    elif 'triangle' in cells:
        stripped_cells = {'triangle': cells['triangle']}
        cell_type = 'triangle'
    else:
        raise RuntimeError(
          'Dolfin XML can only deal with triangle or tetra. '
          'The input data contains only ' + ', '.join(cells.keys()) + '.'
          )

    if len(cells) > 1:
        discarded_cells = cells.keys()
        discarded_cells.remove(cell_type)
        warnings.warn(
          'DOLFIN XML can only handle one cell type at a time. '
          'Using ' + cell_type +
          ', discarding ' + ', '.join(discarded_cells) +
          '.'
          )

    if all(points[:, 2] == 0):
        dim = '2'
    else:
        dim = '3'

    mesh = ET.SubElement(
            dolfin,
            'mesh',
            celltype=meshio_to_dolfin_type[cell_type],
            dim=dim
            )
    vertices = ET.SubElement(mesh, 'vertices', size=str(len(points)))

    for k, point in enumerate(points):
        ET.SubElement(
            vertices,
            'vertex',
            index=str(k),
            x='%r' % point[0],
            y='%r' % point[1],
            z='%r' % point[2]
            )

    num_cells = 0
    for cls in stripped_cells.values():
        num_cells += len(cls)

    xcells = ET.SubElement(mesh, 'cells', size=str(num_cells))
    idx = 0
    for cell_type, cls in stripped_cells.items():
        for cell in cls:
            cell_entry = ET.SubElement(
                xcells,
                meshio_to_dolfin_type[cell_type],
                index=str(idx)
                )
            for k, c in enumerate(cell):
                cell_entry.attrib['v%d' % k] = str(c)
            idx += 1

    tree = ET.ElementTree(dolfin)
    tree.write(filename, pretty_print=True)
    return
