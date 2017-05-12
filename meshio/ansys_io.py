# -*- coding: utf-8 -*-
#
'''
I/O for Ansys's msh format, cf.
<https://www.sharcnet.ca/Software/TGrid/pdf/ug/appb.pdf>.
'''
from . __about__ import __version__

from itertools import islice
import numpy
import re
import warnings


def read(filename):
    # Initialize the data optional data fields
    field_data = {}
    cell_data = {}
    point_data = {}

    points = None
    cells = {}

    with open(filename) as f:
        while True:
            try:
                line = next(islice(f, 1))
            except StopIteration:
                break

            # expect the line to have the form
            #  (<index> [...]
            out = re.match('\s*\(\s*([0-9]+).*', line)
            assert out
            index = int(out.group(1))

            if index == 1:
                # header
                # (1 "<text>")
                pass
            elif index == 2:
                # dimensionality
                # (2 3)
                pass
            elif index == 10:
                # points
                # (10 (zone-id first-index last-index type ND)
                out = re.match('\s*\(\s*10\s+\(([^\)]+)\).*', line)
                a = [int(num, 16) for num in out.group(1).split()]
                zone_id = a[0]
                if zone_id == 0:
                    # a declaration of the total number of points
                    continue

                assert len(a) > 4
                first_index = a[1]
                last_index = a[2]
                num_points = last_index - first_index + 1
                dim = a[4]

                # read point data
                points = numpy.empty((num_points, dim))
                for k in range(num_points):
                    line = next(islice(f, 1))
                    dat = line.split()
                    assert len(dat) == dim
                    for d in range(dim):
                        points[k][d] = float(dat[d])

                # make sure that the point data set is properly closed
                line = next(islice(f, 1))
                assert re.match('\s*\)\s*\)\s*', line)
            elif index == 12:
                # cells
                # (12 (zone-id first-index last-index type element-type))
                out = re.match('\s*\(\s*12\s+\(([^\)]+)\).*', line)
                a = [int(num, 16) for num in out.group(1).split()]
                zone_id = a[0]
                if zone_id == 0:
                    # a declaration of the total number of cells
                    continue

                assert len(a) > 4
                first_index = a[1]
                last_index = a[2]
                num_cells = last_index - first_index + 1
                element_type = a[4]

                if element_type == 1:
                    key = 'triangle'
                    num_nodes_per_cell = 3
                elif element_type == 2:
                    key = 'tetra'
                    num_nodes_per_cell = 4
                elif element_type == 3:
                    key = 'quad'
                    num_nodes_per_cell = 4
                elif element_type == 4:
                    key = 'hexa'
                    num_nodes_per_cell = 8
                elif element_type == 5:
                    key = 'pyra'
                    num_nodes_per_cell = 5
                elif element_type == 6:
                    key = 'wedge'
                    num_nodes_per_cell = 6
                else:
                    raise ValueError('Unknown element type %d.' % element_type)

                # read cell data
                data = numpy.empty((num_cells, num_nodes_per_cell), dtype=int)
                for k in range(num_cells):
                    line = next(islice(f, 1))
                    dat = line.split()
                    assert len(dat) == num_nodes_per_cell
                    data[k] = [int(d, 16) for d in dat]

                cells[key] = data

                # make sure that the point data set is properly closed
                line = next(islice(f, 1))
                assert re.match('\s*\)\s*\)\s*', line)
            else:
                warnings.warn('Unknown index \'%d\'. Skipping.' % index)
                # Skipping ahead to the closing bracket. Assume that, if the
                # line ends with a closing bracket, that's the one. Otherwise
                # skip to the next "))" line.
                if line[-1] == ')':
                    continue
                while re.match('\s*\)\s*\)\s*', line) is None:
                    line = next(islice(f, 1))

    return points, cells, point_data, cell_data, field_data


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
        fh.write(('(2 %d)\n' % dim).encode('utf8'))

        # total number of nodes
        fh.write(('(10 (0 1 %x 0))\n' % len(points)).encode('utf8'))

        # total number of cells
        total_num_cells = sum([len(c) for c in cells])
        fh.write(('(12 (0 1 %x 0))\n' % total_num_cells).encode('utf8'))

        # Write nodes
        fh.write(('(10 (1 1 %x 1 %d))(\n' % points.shape).encode('utf8'))
        numpy.savetxt(fh, points, fmt='%.15e')
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
                '(12 (1 %x %x 1 %d)(\n' %
                (first_index, last_index, meshio_to_ansys_type[key])
                ).encode('utf8'))
            numpy.savetxt(fh, values, fmt='%x')
            fh.write(('))\n').encode('utf8'))
            first_index = last_index + 1

    return
