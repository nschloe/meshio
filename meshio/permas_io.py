# -*- coding: utf-8 -*-
#
'''
I/O for PERMAS dat format, cf.

.. moduleauthor:: Nils Wagner
'''

import gzip
import numpy
import re


def read(filename):
    '''Reads a PERMAS dato, post file.
    '''
    # The format is specified at
    # <http://www.intes.de #PERMAS-ASCII-file-format>.
    if filename.endswith('post.gz'):
        f = gzip.open(filename, 'r')
    elif filename.endswith('dato.gz'):
        f = gzip.open(filename, 'r')
    elif filename.endswith('dato'):
        f = open(filename, 'r')
    elif filename.endswith('post'):
        f = open(filename, 'r')
    else:
        print 'Unsupported file format'

    while True:
        line = f.readline()
        if not line:
            break
        if re.search('\$END STRUCTURE', line):
            break
        if re.search('\$ELEMENT TYPE = TRIA3', line):
            elems = {
                'points': [],
                'lines': [],
                'triangles': [],
                'tetrahedra': []
                }

            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                elems['triangles'].append(data[-3:])

        if re.search('\$ELEMENT TYPE = TET4', line):
            elems = {
                'points': [],
                'lines': [],
                'triangles': [],
                'tetrahedra': []
                }

            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                elems['tetrahedra'].append(data[-4:])

        if re.search('\$COOR', line):
            points = []
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                for r in numpy.array(line.split(), dtype=float)[1:]:
                    points.append(r)
    points = numpy.reshape(points, newshape=(len(points)/3, 3))

    for key in elems:
        # Subtract one to account for the fact that python indices
        # are 0-based.
        elems[key] = numpy.array(elems[key], dtype=int) - 1

    if len(elems['tetrahedra']) > 0:
        cells = elems['tetrahedra']
    elif len(elems['triangles']) > 0:
        cells = elems['triangles']
    else:
        raise RuntimeError('Expected at least triangles.')
    return points, cells, {}, {}, {}


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    '''Writes PERMAS dat files, cf.
    http://www.intes.de # PERMAS-ASCII-file-format
    '''
    with open(filename, 'w') as fh:
        fh.write('$ENTER COMPONENT NAME = DFLT_COMP DOFTYPE = DISP MATH\n')
        fh.write('! \n')
        fh.write('    $STRUCTURE\n')
        fh.write('! \n')

        # Write nodes
        fh.write('        $COOR NSET = ALL_NODES\n')
        for k, x in enumerate(points):
            fh.write(
                '        %8d %+.15f %+.15f %+.15f\n' %
                (k+1, x[0], x[1], x[2])
                )

        # Write elements.
        # translate the number of points per element (aka the type) to a PERMAS
        # enum.
        type_to_number = {
            1: 15,  # point
            2: 1,  # line
            3: 2,  # triangle
            4: 4,  # tetrahedron
            }
        for k, c in enumerate(cells):
            n = len(c)
            form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
            if n == 2:
                if k == 0:
                    fh.write('! \n')
                    fh.write('        $ELEMENT TYPE = PLOTL2 ESET = PLOTL2\n')
                fh.write(form % (k+1, c[0], c[1]))
            elif n == 3:
                if k == 0:
                    fh.write('! \n')
                    fh.write('        $ELEMENT TYPE = TRIA3 ESET = TRIA3\n')
                fh.write(form % (k+1, c[0]+1, c[1]+1, c[2]+1))
            elif n == 4:
                if k == 0:
                    fh.write('!\n')
                    fh.write('        $ELEMENT TYPE = TET4 ESET = TET4\n')
                fh.write(form % (k+1, c[0]+1, c[1]+1, c[2]+1, c[3]+1))
        fh.write('!\n')
        fh.write('    $END STRUCTURE\n')
        fh.write('!\n')
        fh.write('$EXIT COMPONENT\n')
        fh.write('!\n')
        fh.write('$FIN\n')
    return
