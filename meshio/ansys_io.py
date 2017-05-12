# -*- coding: utf-8 -*-
#
'''
I/O for Ansys's msh format, cf.
<https://www.sharcnet.ca/Software/TGrid/pdf/ug/appb.pdf>.
'''
from . __about__ import __version__

import numpy


def read(filename):
    return


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    with open(filename, 'wb') as fh:
        # header
        fh.write(('(1 "meshio %s")\n' % __version__).encode('utf8'))

        # dimension
        dim = 2 if all(points[:, 2] == 0.0) else 3
        fh.write(('(2 %d)"\n' % dim).encode('utf8'))

        # total number of nodes
        fh.write(('(10 (0 1 %x 0))"\n' % len(points)).encode('utf8'))

        # total number of cells
        total_num_cells = sum([len(c) for c in cells])
        fh.write(('(12 (0 1 %x 0))"\n' % total_num_cells).encode('utf8'))

        # Write nodes
        fh.write(('(10 (1 1 %x 1))(\n' % len(points)).encode('utf8'))
        numpy.savetxt(fh, points)
        fh.write(('))\n').encode('utf8'))

        # Write cells
        meshio_to_ansys_type = {
                'triangle': 1,
                'tetra': 2,
                'quad': 3,
                'hex': 4,
                'pyra': 5,
                'wedge': 6,
                }
        first_index = 0
        for key, values in cells.items():
            last_index = first_index + len(values) - 1
            fh.write((
                '((12 (1 %x %x 1 %d))(\n' %
                (first_index, last_index, meshio_to_ansys_type[key])
                ).encode('utf8'))
            numpy.savetxt(fh, values, fmt='%x')
            fh.write(('))\n').encode('utf8'))
            first_index = last_index + 1

    return
