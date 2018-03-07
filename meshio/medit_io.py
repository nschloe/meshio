# -*- coding: utf-8 -*-
#
'''
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
for something like a specification.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from itertools import islice
import logging
import numpy


def read(filename):
    with open(filename) as f:
        points, cells = read_buffer(f)

    return points, cells, {}, {}, {}


def read_buffer(f):
    dim = 0
    cells = {}

    while True:
        try:
            line = next(islice(f, 1))
        except StopIteration:
            break

        stripped = line.strip()

        # skip comments and empty lines
        if not stripped or stripped[0] == '#':
            continue

        assert stripped[0].isalpha()
        keyword = stripped.split(' ')[0]

        meshio_from_medit = {
            'Edges': ('line', 2),
            'Triangles': ('triangle', 3),
            'Quadrilaterals': ('quad', 4),
            'Tetrahedra': ('tetra', 4),
            'Hexahedra': ('hexahedra', 8)
            }

        if keyword == 'MeshVersionFormatted':
            assert stripped[-1] == '1'
        elif keyword == 'Dimension':
            dim = int(stripped[-1])
        elif keyword == 'Vertices':
            assert dim > 0
            # The first line is the number of nodes
            line = next(islice(f, 1))
            num_verts = int(line)
            points = numpy.empty((num_verts, dim), dtype=float)
            for k, line in enumerate(islice(f, num_verts)):
                # Throw away the label immediately
                points[k] = numpy.array(line.split(), dtype=float)[:-1]
        elif keyword in meshio_from_medit:
            meshio_name, num = meshio_from_medit[keyword]
            # The first line is the number of elements
            line = next(islice(f, 1))
            num_cells = int(line)
            cell_data = numpy.empty((num_cells, num), dtype=int)
            for k, line in enumerate(islice(f, num_cells)):
                data = numpy.array(line.split(), dtype=int)
                # Throw away the label
                cell_data[k] = data[:-1]

            # adapt 0-base
            cells[meshio_name] = cell_data - 1
        else:
            assert keyword == 'End', 'Unknown keyword \'{}\'.'.format(keyword)

    return points, cells


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None):
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    with open(filename, 'wb') as fh:
        fh.write(b'MeshVersionFormatted 1\n')
        fh.write(b'# Created by meshio\n')

        # Dimension info
        d = '\nDimension {}\n'.format(points.shape[1])
        fh.write(d.encode('utf-8'))

        # vertices
        fh.write(b'\nVertices\n')
        fh.write('{}\n'.format(len(points)).encode('utf-8'))
        labels = numpy.ones(len(points), dtype=int)
        data = numpy.c_[points, labels]
        fmt = ' '.join(['%r'] * points.shape[1]) + ' %d'
        numpy.savetxt(fh, data, fmt)

        medit_from_meshio = {
            'line': ('Edges', 2),
            'triangle': ('Triangles', 3),
            'quad': ('Quadrilaterals', 4),
            'tetra': ('Tetrahedra', 4),
            'hexahedra': ('Hexahedra', 8)
            }

        for key, data in cells.items():
            try:
                medit_name, num = medit_from_meshio[key]
            except KeyError:
                msg = (
                    'MEDIT\'s mesh format doesn\'t know {} cells. Skipping.'
                    ).format(key)
                logging.warning(msg)
                continue
            fh.write(b'\n')
            fh.write('{}\n'.format(medit_name).encode('utf-8'))
            fh.write('{}\n'.format(len(data)).encode('utf-8'))
            labels = numpy.ones(len(data), dtype=int)
            # adapt 1-base
            data_with_label = numpy.c_[data + 1, labels]
            fmt = ' '.join(['%d'] * (num + 1))
            numpy.savetxt(fh, data_with_label, fmt)

        fh.write(b'\nEnd\n')

    return
