# -*- coding: utf-8 -*-
#
'''
I/O for PERMAS dat format, cf.

.. moduleauthor:: Nils Wagner
'''

import gzip
import numpy
import re
from .__about__ import __version__, __website__, __author__


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
        print('Unsupported file format')

    cells = {}
    meshio_to_permas_type = {
        'line': (2, 'PLOTL2'),
        'triangle': (3, 'TRIA3'),
        'quad': (4, 'QUAD4'),
        'tetra': (4, 'TET4'),
        'hexahedron': (8, 'HEXE8'),
        'wedge': (6, 'PENTA6'),
        'pyramid': (5, 'PYRA5')
        }

    while True:
        line = f.readline()
        if not line or re.search('\$END STRUCTURE', line):
            break
        for meshio_type, permas_ele in meshio_to_permas_type.items():
            num_nodes = permas_ele[0]
            permas_type = permas_ele[1]

            if re.search('\$ELEMENT TYPE = %s' % permas_type, line):
                while True:
                    line = f.readline()
                    if not line or line.startswith('!'):
                        break
                    data = numpy.array(line.split(), dtype=int)
                    if meshio_type in cells:
                        cells[meshio_type].append(data[-num_nodes:])
                    else:
                        cells[meshio_type] = [data[-num_nodes:]]

        if re.search('\$COOR', line):
            points = []
            while True:
                line = f.readline()
                if not line or line.startswith('!'):
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
    if point_data is None:
        point_data = {}
    if cell_data is None:
        cell_data = {}
    if field_data is None:
        field_data = {}

    with open(filename, 'w') as fh:
        fh.write('!\n')
        fh.write('! Author of meshio: %s\n' % __author__)
        fh.write('! File written by meshio version %s\n' % __version__)
        fh.write('! Further information available at %s\n' % __website__)
        fh.write('!\n')
        fh.write('$ENTER COMPONENT NAME = DFLT_COMP DOFTYPE = DISP MATH\n')
        fh.write('! \n')
        fh.write('    $SITUATION NAME = REAL_MODES\n')
        fh.write('        DFLT_COMP SYSTEM = NSV CONSTRAINTS = SPCVAR_1 ! LOADING = LOADVAR_1\n')
        fh.write('    $END SITUATION\n')
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

        meshio_to_permas_type = {
            'line': (2, 'PLOTL2'),
            'triangle': (3, 'TRIA3'),
            'quad': (4, 'QUAD4'),
            'tetra': (4, 'TET4'),
            'hexahedron': (8, 'HEXE8'),
            'wedge': (6, 'PENTA6'),
            'pyramid': (5, 'PYRA5')
            }

        #
        # Avoid non-unique element numbers in case of multiple element types by
        # num_ele !!!
        #
        num_ele = 0

        for meshio_type, cell in cells.items():
            numcells, num_local_nodes = cell.shape
            permas_type = meshio_to_permas_type[meshio_type]
            fh.write('!\n')
            fh.write(
                '        $ELEMENT TYPE = %s ESET = %s\n' %
                (permas_type[1], permas_type[1])
                )
            for k, c in enumerate(cell):
                form = '        %8d ' + \
                    ' '.join(num_local_nodes * ['%8d']) + \
                    '\n'
                fh.write(form % ((k+num_ele+1,) + tuple(c + 1)))
            num_ele += numcells

        fh.write('!\n')
        fh.write('    $END STRUCTURE\n')
        fh.write('!\n')
        elem_3D = ['HEXE8', 'TET4', 'PENTA6', 'PYRA5']
        elem_2D = ['TRIA3', 'QUAD4']
        fh.write('    $SYSTEM NAME = NSV\n')
        fh.write('!\n')
        fh.write('        $ELPROP\n')
        for meshio_type, cell in cells.items():
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type[1] in elem_3D:
                fh.write(
                    '            %s MATERIAL = DUMMY_MATERIAL\n' %
                    permas_type[1]
                    )
            elif permas_type[1] in elem_2D:
                fh.write('            %s GEODAT = GD_%s MATERIAL = DUMMY_MATERIAL\n' %(permas_type[1],permas_type[1]))
            else:
                pass
        fh.write('!\n')
        fh.write('        $GEODAT SHELL  CONT = THICK  NODES = ALL\n')
        for meshio_type, cell in cells.items():
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type[1] in elem_2D:
                fh.write('            GD_%s 1.0\n' % permas_type[1])
        fh.write('!\n')
        fh.write('    $END SYSTEM\n')
        fh.write('!\n')
        fh.write('    $CONSTRAINTS NAME = SPCVAR_1\n')
        fh.write('    $END CONSTRAINTS\n')
        fh.write('!\n')
        fh.write('    $LOADING NAME = LOADVAR_1\n')
        fh.write('    $END LOADING\n')
        fh.write('!\n')
        fh.write('$EXIT COMPONENT\n')
        fh.write('!\n')
        fh.write('$ENTER MATERIAL\n')
        fh.write('!\n')
        fh.write('    $MATERIAL NAME = DUMMY_MATERIAL TYPE = ISO\n')
        fh.write('!\n')
        fh.write('        $ELASTIC  GENERAL  INPUT = DATA\n')
        fh.write('            0.0 0.0\n')
        fh.write('!\n')
        fh.write('        $DENSITY  GENERAL  INPUT = DATA\n')
        fh.write('            0.0\n')
        fh.write('!\n')
        fh.write('        $THERMEXP  GENERAL  INPUT = DATA\n')
        fh.write('            0.0\n')
        fh.write('!\n')
        fh.write('    $END MATERIAL\n')
        fh.write('!\n')
        fh.write('$EXIT MATERIAL\n')
        fh.write('!\n')
        fh.write('$FIN\n')
    return
