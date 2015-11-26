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
                line = islice(f, 1).next()
            except StopIteration:
                break
            assert(line[0] == '$')
            environ = line[1:].strip()
            if environ == 'MeshFormat':
                line = islice(f, 1).next()
                # 2.2 0 8
                line = islice(f, 1).next()
                assert(line.strip() == '$EndMeshFormat')
            elif environ == 'Nodes':
                # The first line is the number of nodes
                line = islice(f, 1).next()
                num_nodes = int(line)
                points = numpy.empty((num_nodes, 3))
                for k, line in enumerate(islice(f, num_nodes)):
                    # Throw away the index immediately
                    points[k, :] = numpy.array(line.split(), dtype=float)[1:]
                line = islice(f, 1).next()
                assert(line.strip() == '$EndNodes')
            elif environ == 'Elements':
                # The first line is the number of elements
                line = islice(f, 1).next()
                num_elems = int(line)
                elems = {
                    'points': [],
                    'lines': [],
                    'triangles': [],
                    'tetrahedra': [],
                    'prism': [],
                    'hexahedra': []
                    }
                for k, line in enumerate(islice(f, num_elems)):
                    # Throw away the index immediately
                    data = numpy.array(line.split(), dtype=int)
                    if data[1] == 15:
                        elems['points'].append(data[-1:])
                    elif data[1] == 1:
                        elems['lines'].append(data[-2:])
                    elif data[1] == 2:
                        elems['triangles'].append(data[-3:])
                    elif data[1] == 3:
                        pass
                    elif data[1] == 4:
                        elems['tetrahedra'].append(data[-4:])
                    elif data[1] == 5:
                        elems['hexahedra'].append(data[-8:])
                    elif data[1] == 6:
                        elems['prism'].append(data[-6:])
                    else:
                        raise RuntimeError('Unknown element type')
                for key in elems:
                    # Subtract one to account for the fact that python indices
                    # are 0-based.
                    elems[key] = numpy.array(elems[key], dtype=int) - 1
                line = islice(f, 1).next()
                assert(line.strip() == '$EndElements')
            elif environ == 'PhysicalNames':
                line = islice(f, 1).next()
                num_phys_names = int(line)
                for k, line in enumerate(islice(f, num_phys_names)):
                    pass
                line = islice(f, 1).next()
                assert(line.strip() == '$EndPhysicalNames')
            else:
                raise RuntimeError('Unknown environment \'%s\'.' % environ)

    if len(elems['tetrahedra']) > 0:
        cells = elems['tetrahedra']
    elif len(elems['hexahedra']) > 0:
        cells = elems['hexahedra']
    elif len(elems['triangles']) > 0:
        cells = elems['triangles']
    else:
        raise RuntimeError('Expected at least triangles.')

    return points, cells


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
        for key, data in cells.iteritems():
            n = data.shape[1]
            form = '%d ' + '%d' % meshio_to_gmsh_type[key] + ' 0 ' + \
                ' '.join(n * ['%d']) + '\n'
            for k, c in enumerate(data):
                fh.write(form % ((k,) + tuple(c + 1)))
        fh.write('$EndElements')

    return
