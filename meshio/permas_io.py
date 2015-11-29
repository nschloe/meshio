# -*- coding: utf-8 -*-
#
'''
I/O for PERMAS dat format, cf.

.. moduleauthor:: Nils Wagner
'''

import gzip
import numpy
import re
from meta import __version__,__website__

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

        if re.search('\$ELEMENT TYPE = QUAD4', line):
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith('!'):
                    break
                data = numpy.array(line.split(), dtype=int)
                if 'quad' in cells:
                    cells['quad'].append(data[-4:])
                else:
                    cells['quad'] = [data[-4:]]

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
        print 'key', key
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
        fh.write('!\n')
        fh.write('! file written by meshio version %s\n' %__version__)
        fh.write('! Further information available at %s\n' %__website__)
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
            'vertex': 'NA',
            'line': 'PLOTL2',
            'triangle': 'TRIA3',
            'quad': 'QUAD4',
            'tetra': 'TET4',
            'hexahedron': 'HEXE8',
            'wedge': 'PENTA6',
            'pyramid': 'PYRA5'
            }
        #
        # Avoid non-unique element numbers in case of multiple element types by num_ele !!!
        #
        num_ele = 0
        
        for meshio_type, cell in cells.iteritems():
            numcells, num_local_nodes = cell.shape
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type != 'NA':
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = %s ESET = %s\n' % (permas_type, permas_type))
                for k, c in enumerate(cell):
                    form = '        %8d ' + ' '.join(num_local_nodes * ['%8d']) + '\n'
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
        for meshio_type, cell in cells.iteritems():
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type in elem_3D:
                fh.write('            %s MATERIAL = STEEL\n' %permas_type)
            elif permas_type in elem_2D:
                fh.write('            %s GEODAT = GD_%s MATERIAL = STEEL\n' %(permas_type,permas_type))
            else:
                pass
        fh.write('!\n')
        fh.write('        $GEODAT SHELL  CONT = THICK  NODES = ALL\n')
        for meshio_type, cell in cells.iteritems():
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type in elem_2D:
                fh.write('            GD_%s 1.0\n' %permas_type)
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
        fh.write('    $MATERIAL NAME = STEEL TYPE = ISO\n')
        fh.write('!\n')
        fh.write('        $ELASTIC  GENERAL  INPUT = DATA\n')
        fh.write('            2.1E+05 0.3\n')
        fh.write('!\n')
        fh.write('        $DENSITY  GENERAL  INPUT = DATA\n')
        fh.write('            7.85E-09\n')
        fh.write('!\n')
        fh.write('        $THERMEXP  GENERAL  INPUT = DATA\n')
        fh.write('            1.200000E-05\n')
        fh.write('!\n')
        fh.write('    $END MATERIAL\n')
        fh.write('!\n')
        fh.write('$EXIT MATERIAL\n')
        fh.write('!\n')
        fh.write('$FIN\n')
    return
