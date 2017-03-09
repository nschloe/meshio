# -*- coding: utf-8 -*-
#
'''
I/O for Gmsh's msh format, cf.
<http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from itertools import islice
import numpy
import warnings


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename) as f:
        points, cells, point_data, cell_data, field_data = read_buffer(f)

    return points, cells, point_data, cell_data, field_data


def read_buffer(f):
    # The format is specified at
    # <http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format>.

    # Initialize the data optional data fields
    field_data = {}
    cell_data = {}
    point_data = {}

    has_additional_tag_data = False
    while True:
        try:
            line = next(islice(f, 1))
        except StopIteration:
            break
        assert line[0] == '$'
        environ = line[1:].strip()
        if environ == 'MeshFormat':
            line = next(islice(f, 1))
            # 2.2 0 8
            line = next(islice(f, 1))
            assert line.strip() == '$EndMeshFormat'
        elif environ == 'PhysicalNames':
            line = next(islice(f, 1))
            num_phys_names = int(line)
            for k, line in enumerate(islice(f, num_phys_names)):
                key = line.split(' ')[2].replace('"', '').replace('\n', '')
                phys_group = int(line.split(' ')[1])
                field_data[key] = phys_group
            line = next(islice(f, 1))
            assert line.strip() == '$EndPhysicalNames'
        elif environ == 'Nodes':
            # The first line is the number of nodes
            # TODO speed up with <http://stackoverflow.com/a/41642053/353337>
            line = next(islice(f, 1))
            num_nodes = int(line)
            points = numpy.empty((num_nodes, 3))
            for k, line in enumerate(islice(f, num_nodes)):
                # Throw away the index immediately
                points[k, :] = numpy.array(line.split(), dtype=float)[1:]
            line = next(islice(f, 1))
            assert line.strip() == '$EndNodes'
        elif environ == 'Elements':
            # The first line is the number of elements
            line = next(islice(f, 1))
            num_cells = int(line)
            cells = {}
            gmsh_to_meshio_type = {
                    15: ('vertex', 1),
                    1: ('line', 2),
                    2: ('triangle', 3),
                    3: ('quad', 4),
                    4: ('tetra', 4),
                    5: ('hexahedron', 8),
                    6: ('wedge', 6),
                    7: ('pyramid', 5),
                    8: ('line3', 3),
                    9: ('triangle6', 6),
                    10: ('quad9', 9),
                    11: ('tetra10', 10),
                    12: ('hexahedron27', 27),
                    13: ('prism18', 18),
                    14: ('pyramid14', 14),
                    26: ('line4', 4),
                    36: ('quad16', 16)
                    }
            # For each cell, there are at least two tags: the physical entity
            # and the elementary geometrical entity the cell belongs to (see
            # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>).
            for k, line in enumerate(islice(f, num_cells)):
                # Throw away the index (data[0]) immediately;
                data = numpy.array(line.split(), dtype=int)
                t = gmsh_to_meshio_type[data[1]]

                if t[0] not in cells:
                    cells[t[0]] = []

                # Subtract one to account for the fact that python indices are
                # 0-based.
                cells[t[0]].append(data[-t[1]:] - 1)

                # data[2] gives the number of tags. The gmsh manual
                # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>
                # says:
                # >>>
                # By default, the first tag is the number of the physical
                # entity to which the element belongs; the second is the number
                # of the elementary geometrical entity to which the element
                # belongs; the third is the number of mesh partitions to which
                # the element belongs, followed by the partition ids (negative
                # partition ids indicate ghost cells). A zero tag is equivalent
                # to no tag. Gmsh and most codes using the MSH 2 format require
                # at least the first two tags (physical and elementary tags).
                # <<<

                # Just remember if there's any more tag data so we can warn. No
                # idea how to store the data in an array yet since it can be
                # "sparse", i.e., not all cells need to have the same number of
                # tags.
                if data[2] >= 2:
                    if data[2] > 2:
                        has_additional_tag_data = True

                    if t[0] not in cell_data:
                        cell_data[t[0]] = {
                            'physical': [],
                            'geometrical': [],
                            }
                    cell_data[t[0]]['physical'].append(data[3])
                    cell_data[t[0]]['geometrical'].append(data[4])
                else:
                    # TODO handle data[2] == 1
                    assert data[2] == 0

            line = next(islice(f, 1))
            assert line.strip() == '$EndElements'

            for key in cells:
                cells[key] = numpy.vstack(cells[key])
        else:
            raise RuntimeError('Unknown environment \'%s\'.' % environ)

    if has_additional_tag_data:
        warnings.warn(
            'The file contains tag data that couldn\'t be processed.'
            )

    return points, cells, point_data, cell_data, field_data


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    '''Writes msh files, cf.
    http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
    '''
    if point_data is None:
        point_data = {}
    if cell_data is None:
        cell_data = {}
    if field_data is None:
        field_data = {}

    with open(filename, 'w') as fh:
        fh.write('$MeshFormat\n2 0 8\n$EndMeshFormat\n')

        # Write nodes
        fh.write('$Nodes\n')
        fh.write('%d\n' % len(points))
        for k, x in enumerate(points):
            fh.write('%d %r %r %r\n' % (k+1, x[0], x[1], x[2]))
        fh.write('$EndNodes\n')

        # Translate meshio types to gmsh codes
        # http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
        meshio_to_gmsh_type = {
                'vertex': 15,
                'line': 1,
                'triangle': 2,
                'quad': 3,
                'tetra': 4,
                'hexahedron': 5,
                'wedge': 6,
                }
        fh.write('$Elements\n')
        # count all cells
        num_cells = 0
        for key, data in cells.items():
            num_cells += data.shape[0]
        fh.write('%d\n' % num_cells)

        consecutive_index = 0
        for cell_type, node_idcs in cells.items():
            if cell_type in cell_data and len(cell_data[cell_type]) > 0:
                for key in cell_data[cell_type]:
                    # assert data consistency
                    assert len(cell_data[cell_type][key]) == len(node_idcs)
                    # TODO assert that the data type is int

                # if a tag is present, make sure that there are 'physical' and
                # 'geometrical' as well.
                if 'physical' not in cell_data[cell_type]:
                    cell_data[cell_type]['physical'] = \
                        numpy.ones(len(node_idcs))
                if 'geometrical' not in cell_data[cell_type]:
                    cell_data[cell_type]['geometrical'] = \
                        numpy.ones(len(node_idcs))

                # 'physical' and 'geometrical' go first; this is what the gmsh
                # file format prescribes
                keywords = cell_data[cell_type].keys()
                keywords.remove('physical')
                keywords.remove('geometrical')
                sorted_keywords = ['physical', 'geometrical'] + keywords
                fcd = numpy.column_stack([
                        cell_data[cell_type][key] for key in sorted_keywords
                        ])

                num_nodes_per_cell = node_idcs.shape[1]
                form = '%d ' + '%d' % meshio_to_gmsh_type[cell_type] \
                    + ' %d' % len(cell_data) + ' %d' * len(cell_data) \
                    + ' ' + ' '.join(num_nodes_per_cell * ['%d']) \
                    + '\n'
                for k, c in enumerate(node_idcs):
                    fh.write(
                        form % (
                            (consecutive_index + k + 1,) +
                            tuple(fcd[k]) +
                            tuple(c + 1)
                            )
                        )
            else:  # no cell data
                num_nodes_per_cell = node_idcs.shape[1]

                form = '%d ' + '%d' % meshio_to_gmsh_type[cell_type] \
                    + ' 0 ' \
                    + ' ' + ' '.join(num_nodes_per_cell * ['%d']) \
                    + '\n'
                for k, c in enumerate(node_idcs):
                    fh.write(
                        form % (
                            (consecutive_index + k + 1,) +
                            tuple(c + 1)
                            )
                        )
            consecutive_index += data.shape[0]
        fh.write('$EndElements')

    return
