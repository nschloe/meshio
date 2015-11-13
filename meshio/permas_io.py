# -*- coding: utf-8 -*-
#
'''
I/O for PERMAS dat format, cf.

.. moduleauthor:: 
'''

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
        fh.write('    $STRUCTURE\n') 

        # Write nodes
        fh.write('        $COOR NSET = ALL_NODES\n')
        for k, x in enumerate(points):
            fh.write('%d %f %f %f\n' % (k+1, x[0], x[1], x[2]))

        # Write elements.
        # translate the number of points per element (aka the type) to a PERMAS
        # enum.
        type_to_number = {
            1: 15,  # point
            2: 1,  # line
            3: 2,  # triangle
            4: 4,  # tetrahedron
            }
        fh.write('! Number of elements %d\n' % len(cells))
        for k, c in enumerate(cells):
            n = len(c)
            form = '%d ' + ' '.join(n * ['%d']) + '\n'
#
#           EID : element ID > 0 !!!
#           N1  : Node ID
#
            if n==2:
                fh.write('$ELEMENT TYPE = PLOTL2\n')
#                  EID N1 N2  
            elif n==3:
                fh.write('$ELEMENT TYPE = TRIA3\n')
#                  EID N1 N2 N3 
            elif n==4:
                fh.write('$ELEMENT TYPE = TET4\n')
#                  EID N1 N2 N3 N4
        fh.write('    $END STRUCTURE\n')
        fh.write('$EXIT COMPONENT\n')
        fh.write('$FIN\n')
    return
