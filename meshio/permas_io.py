# -*- coding: utf-8 -*-
#
'''
I/O for PERMAS dat format.

.. moduleauthor:: Nils Wagner
'''


def write(
        filename,
        points,
        cells
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
        fh.write('! Number of nodes = %d\n' % len(points))
        fh.write('! \n')
        fh.write('        $COOR NSET = ALL_NODES\n')
        fh.write('! \n')
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
        fh.write('!\n')
        fh.write('! Number of elements = %d\n' % len(cells))
        fh.write('!\n')
        for k, c in enumerate(cells):
            n = len(c)
            form = '        %8d ' + ' '.join(n * ['%8d']) + '\n'
#
#           EID : element ID > 0 !!!
#           Ni  : Node ID
#
            if n == 2:
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = PLOTL2 ESET = PLOTL2\n')
                fh.write('!\n')
                fh.write(form % (k+1, c[0]+1, c[1]+1))
            elif n == 3:
                fh.write('!\n')
                fh.write('        $ELEMENT TYPE = TRIA3 ESET = TRIA3\n')
                fh.write('!\n')
                fh.write(form % (k+1, c[0]+1, c[1]+1, c[2]+1))
            elif n == 4:
                if k == 0:
                    fh.write('!\n')
                    fh.write('        $ELEMENT TYPE = TET4 ESET = TET4\n')
                    fh.write('!\n')
                fh.write(form % (k+1, c[0]+1, c[1]+1, c[2]+1, c[3]+1))
        fh.write('!\n')
        fh.write('    $END STRUCTURE\n')
        fh.write('!\n')
        fh.write('$EXIT COMPONENT\n')
        fh.write('!\n')
        fh.write('$FIN\n')
    return
