# -*- coding: utf-8 -*-
#
'''
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy


def read(filename):
    '''Reads a Medit file.
    '''
    with open(filename) as f:
        points, cells = read_buffer(f)

    return points, cells, {}, {}, {}


def read_buffer(f):
    return


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

    with open(filename, 'w') as fh:
        fh.write('MeshVersionFormatted 1\n')
        fh.write('# Created by meshio\n')

        # Dimension info
        fh.write('\n')
        fh.write('Dimension\n')
        fh.write('%d\n' % points.shape[1])

        # vertices
        fh.write('\n')
        fh.write('Vertices\n')
        fh.write('%d\n' % len(points))
        labels = numpy.ones(len(points), dtype=int)
        data = numpy.c_[points, labels]
        fmt = ' '.join(['%e'] * points.shape[1]) + ' %d'
        numpy.savetxt(fh, data, fmt)

        # edges
        if 'line' in cells:
            fh.write('\n')
            fh.write('Edges\n')
            fh.write('%d\n' % len(cells['line']))
            labels = numpy.ones(len(cells['line']), dtype=int)
            data = numpy.c_[cells['line'], labels]
            numpy.savetxt(fh, data, '%d %d %d')

        # triangles
        if 'triangle' in cells:
            fh.write('\n')
            fh.write('Triangles\n')
            fh.write('%d\n' % len(cells['triangle']))
            labels = numpy.ones(len(cells['triangle']), dtype=int)
            data = numpy.c_[cells['triangle'], labels]
            numpy.savetxt(fh, data, '%d %d %d %d')

        # quadrilaterals
        if 'quad' in cells:
            fh.write('\n')
            fh.write('Quadrilaterals\n')
            fh.write('%d\n' % len(cells['quad']))
            labels = numpy.ones(len(cells['quad']), dtype=int)
            data = numpy.c_[cells['quad'], labels]
            numpy.savetxt(fh, data, '%d %d %d %d %d')

        # tetrahedra
        if 'tetra' in cells:
            fh.write('\n')
            fh.write('Tetrahedra\n')
            fh.write('%d\n' % len(cells['tetra']))
            labels = numpy.ones(len(cells['tetra']), dtype=int)
            data = numpy.c_[cells['tetra'], labels]
            numpy.savetxt(fh, data, '%d %d %d %d %d')

        # hexahedra
        if 'hexahedra' in cells:
            fh.write('\n')
            fh.write('Hexahedra\n')
            fh.write('%d\n' % len(cells['hexahedra']))
            labels = numpy.ones(len(cells['hexahedra']), dtype=int)
            data = numpy.c_[cells['hexahedra'], labels]
            numpy.savetxt(fh, data, '%d %d %d %d %d %d %d %d %d')

        fh.write('\nEnd\n')

    return
