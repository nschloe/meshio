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
    '''Reads a (compressed) PERMAS dato or post file.
    '''
    # The format is specified at
    # <http://www.intes.de>.
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

    cells = {}

    while True:
        line = f.readline()
        if not line:
            break
        if re.search('\$END STRUCTURE', line):
            break
        if re.search('\$ELEMENT TYPE = TRIA3', line):
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                if 'triangle' in cells:
                    cells['triangle'].append(data[-3:])
                else:
                    cells['triangle'] = [data[-3:]]

        if re.search('\$ELEMENT TYPE = TET4', line):
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                if 'tetra' in cells:
                    cells['tetra'].append(data[-4:])
                else:
                    cells['tetra'] = [data[-4:]]

        if re.search('\$ELEMENT TYPE = PENTA6', line):
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                if 'wedge' in cells:
                    cells['wedge'].append(data[-6:])
                else:
                    cells['wedge'] = [data[-6:]]

        if re.search('\$ELEMENT TYPE = HEXE8', line):
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                if 'hexahedron' in cells:
                    cells['hexahedron'].append(data[-8:])
                else:
                    cells['hexahedron'] = [data[-8:]]

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

    for key in cells:
        # Subtract one to account for the fact that python indices
        # are 0-based.
        cells[key] = numpy.array(cells[key], dtype=int) - 1

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

        #
        # Avoid non-unique element numbers in case of multiple element types by num_ele !!!
        #
        num_ele = 0
        for eltyp in cells.keys(): # Loop over available element types
            
            if eltyp == 'line':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = PLOTL2 ESET = PLOTL2\n')
                cell = cells[eltyp]
                for k, c in enumerate(cell):
                    n = len(c)
                    form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
                    fh.write(form % (k+1, c[0]+1, c[1]+1))
                num_ele += len(cell)

            elif eltyp == 'quad':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = QUAD4 ESET = QUAD4\n')
                cell = cells[eltyp]
                for k, c in enumerate(cell):
                    n = len(c)
                    form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
                    fh.write(form % (num_ele+k+1, c[0]+1, c[1]+1, c[2]+1, c[3]+1))
                num_ele += len(cell)
                
            elif eltyp == 'triangle':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = TRIA3 ESET = TRIA3\n')
                cell = cells[eltyp]
                for k, c in enumerate(cell):
                    n = len(c)
                    form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
                    fh.write(form % (num_ele+k+1, c[0]+1, c[1]+1, c[2]+1))
                num_ele += len(cell)
                
            elif eltyp == 'tetrahedron':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = TET4 ESET = TET4\n')
                cell = cells[eltyp]
                for k, c in enumerate(cell):
                    n = len(c)
                    form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
                    fh.write(form % (num_ele+k+1, c[0]+1, c[1]+1, c[2]+1, c[3]+1))
                num_ele += len(cell)

            elif eltyp == 'wedge':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = PENTA6 ESET = TET4\n')
                cell = cells[eltyp]
                for k, c in enumerate(cell):
                    n = len(c)
                    form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
                    fh.write(form % (
                        num_ele+k+1, c[0]+1, c[1]+1, c[2]+1, c[3]+1,
                        c[4]+1,c[5]+1))
                num_ele += len(cell)

            elif eltyp == 'hexahedron':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = HEXE8 ESET = HEXE8\n')
                cell = cells[eltyp]
                for k, c in enumerate(cell):
                    n = len(c)
                    form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
                    fh.write(form % (
                        num_ele+k+1, c[0]+1, c[1]+1, c[2]+1, c[3]+1,
                        c[4]+1, c[5]+1, c[6]+1, c[7]+1
                        ))
                num_ele += len(cell)
            else:
                print 'Unknown element type'
                
        fh.write('!\n')
        fh.write('    $END STRUCTURE\n')
        fh.write('!\n')
        fh.write('$EXIT COMPONENT\n')
        fh.write('!\n')
        fh.write('$FIN\n')
    return
