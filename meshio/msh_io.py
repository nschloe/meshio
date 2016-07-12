# -*- coding: utf-8 -*-
#
'''
I/O for Gmsh's msh format, cf.
<http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from itertools import islice
import numpy


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    # The format is specified at
    # <http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    with open(filename) as f:
        while True:
            try:
                line = next(islice(f, 1))
            except StopIteration:
                break
            assert(line[0] == '$')
            environ = line[1:].strip()
            if environ == 'MeshFormat':
                line = next(islice(f, 1))
                # 2.2 0 8
                line = next(islice(f, 1))
                assert(line.strip() == '$EndMeshFormat')
            elif environ == 'Nodes':
                # The first line is the number of nodes
                line = next(islice(f, 1))
                num_nodes = int(line)
                points = numpy.empty((num_nodes, 3))
                for k, line in enumerate(islice(f, num_nodes)):
                    # Throw away the index immediately
                    points[k, :] = numpy.array(line.split(), dtype=float)[1:]
                line = next(islice(f, 1))
                assert(line.strip() == '$EndNodes')
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
                        6: ('wedge', 6)
                        }
                for k, line in enumerate(islice(f, num_cells)):
                    # Throw away the index immediately;
                    data = numpy.array(line.split(), dtype=int)
                    t = gmsh_to_meshio_type[data[1]]
                    # Subtract one to account for the fact that python indices
                    # are 0-based.
                    if t[0] in cells:
                        cells[t[0]].append(data[-t[1]:] - 1)
                    else:
                        cells[t[0]] = [data[-t[1]:] - 1]

                line = next(islice(f, 1))
                assert(line.strip() == '$EndElements')
            elif environ == 'PhysicalNames':
                line = next(islice(f, 1))
                num_phys_names = int(line)
                for k, line in enumerate(islice(f, num_phys_names)):
                    pass
                line = next(islice(f, 1))
                assert(line.strip() == '$EndPhysicalNames')
            else:
                raise RuntimeError('Unknown environment \'%s\'.' % environ)

    for key in cells:
        cells[key] = numpy.vstack(cells[key])

    return points, cells, {}, {}, {}


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
            fh.write('%d %f %f %f\n' % (k+1, x[0], x[1], x[2]))
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
        num_cells = 0
        for key, data in cells.iteritems():
            num_cells += data.shape[0]
        fh.write('%d\n' % num_cells)
        num_cells = 0
        for key, data in cells.iteritems():
            n = data.shape[1]
            form = '%d ' + '%d' % meshio_to_gmsh_type[key] + ' 0 ' + \
                ' '.join(n * ['%d']) + '\n'
            for k, c in enumerate(data):
                fh.write(form % ((num_cells+k+1,) + tuple(c + 1)))
            num_cells += data.shape[0]    
        fh.write('$EndElements')

    return
