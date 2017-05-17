# -*- coding: utf-8 -*-
#
'''
I/O for Ansys's msh format, cf.
<http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf>.
'''
from . __about__ import __version__

from itertools import islice
import numpy
import re
import warnings


def _skip_till_two_closing_brackets(f):
    line = next(islice(f, 1))
    while line.strip() == '':
        line = next(islice(f, 1))

    if re.match('\s*\)\s*\)\s*$', line):
        # Two closing brackets: alright, continue.
        return

    # One closing bracket: skip ahead for the other one.
    assert line.strip() == ')'
    line = next(islice(f, 1))
    while line.strip() == '':
        line = next(islice(f, 1))
    assert line.strip() == ')'
    return


def read(filename):
    # Initialize the data optional data fields
    field_data = {}
    cell_data = {}
    point_data = {}

    points = []
    cells = {}

    first_point_index_overall = None
    last_point_index = None

    with open(filename) as f:
        while True:
            try:
                line = next(islice(f, 1))
            except StopIteration:  # EOF
                break

            if line.strip() == '':
                continue

            # expect the line to have the form
            #  (<index> [...]
            out = re.match('\s*\(\s*([0-9]+).*', line)
            assert out
            index = out.group(1)

            if index == '0':
                # Comment.
                # If the last character of the line isn't a closing bracket,
                # skip ahead to a line that consists of only a closing bracket.
                if line.strip()[-1] == ')':
                    continue
                while re.match('\s*\)\s*', line) is None:
                    line = next(islice(f, 1))
            elif index == '1':
                # header
                # (1 "<text>")
                pass
            elif index == '2':
                # dimensionality
                # (2 3)
                pass
            elif index == '10':
                # points
                # (10 (zone-id first-index last-index type ND)

                # If the line is self-contained, it is merely a declaration of
                # the total number of points.
                if re.match('^\s*\(\s*10\s*\([^\)]+\)\s*\)\s*$', line):
                    continue

                out = re.match('\s*\(\s*10\s*\(([^\)]*)\).*', line)
                a = [int(num, 16) for num in out.group(1).split()]

                assert len(a) > 4
                first_point_index = a[1]
                # store the very first point index
                if first_point_index_overall is None:
                    first_point_index_overall = first_point_index
                # make sure that point arrays are subsequent
                if last_point_index is not None:
                    assert last_point_index + 1 == first_point_index
                last_point_index = a[2]
                num_points = last_point_index - first_point_index + 1
                dim = a[4]

                # Skip ahead to the line that opens the data block (might be
                # the current line already).
                while line.strip()[-1] != '(':
                    line = next(islice(f, 1))

                # read point data
                pts = numpy.empty((num_points, dim))
                for k in range(num_points):
                    line = next(islice(f, 1))
                    dat = line.split()
                    assert len(dat) == dim
                    for d in range(dim):
                        pts[k][d] = float(dat[d])

                points.append(pts)

                # make sure that the data set is properly closed
                _skip_till_two_closing_brackets(f)
            elif index == '12':
                # cells
                # (12 (zone-id first-index last-index type element-type))

                # If the line is self-contained, it is merely a declaration of
                # the total number of points.
                if re.match('^\s*\(\s*12\s*\([^\)]+\)\s*\)\s*$', line):
                    continue

                out = re.match('\s*\(\s*12\s*\(([^\)]+)\).*', line)
                a = [int(num, 16) for num in out.group(1).split()]

                assert len(a) > 4
                first_index = a[1]
                last_index = a[2]
                num_cells = last_index - first_index + 1
                element_type = a[4]

                element_type_to_key_num_nodes = {
                        0: ('mixed', None),
                        1: ('triangle', 3),
                        2: ('tetra', 4),
                        3: ('quad', 4),
                        4: ('hexa', 8),
                        5: ('pyra', 5),
                        6: ('wedge', 6),
                        }

                key, num_nodes_per_cell = \
                    element_type_to_key_num_nodes[element_type]

                if key == 'mixed':
                    warnings.warn(
                        'Cannot deal with mixed element type. Skipping.'
                        )
                    # Skipping ahead to the next line with two closing
                    # brackets.
                    while re.search('[^\)]*\)\s*\)\s*$', line) is None:
                        line = next(islice(f, 1))
                    continue

                # Skip ahead to the line that opens the data block (might be
                # the current line already).
                while line.strip()[-1] != '(':
                    line = next(islice(f, 1))

                # read cell data
                data = numpy.empty((num_cells, num_nodes_per_cell), dtype=int)
                for k in range(num_cells):
                    line = next(islice(f, 1))
                    dat = line.split()
                    assert len(dat) == num_nodes_per_cell
                    data[k] = [int(d, 16) for d in dat]

                cells[key] = data

                # make sure that the data set is properly closed
                _skip_till_two_closing_brackets(f)
            elif index == '13':
                # cells
                # (13 (zone-id first-index last-index type element-type))

                # If the line is self-contained, it is merely a declaration of
                # the total number of points.
                if re.match('^\s*\(\s*13\s*\([^\)]+\)\s*\)\s*$', line):
                    continue

                out = re.match('\s*\(\s*13\s*\(([^\)]+)\).*', line)
                a = [int(num, 16) for num in out.group(1).split()]

                assert len(a) > 4
                first_index = a[1]
                last_index = a[2]
                num_cells = last_index - first_index + 1
                element_type = a[4]

                element_type_to_key_num_nodes = {
                        0: ('mixed', None),
                        2: ('linear', 2),
                        3: ('triangle', 3),
                        4: ('quad', 4)
                        }

                key, num_nodes_per_cell = \
                    element_type_to_key_num_nodes[element_type]

                # Skip ahead to the line that opens the data block (might be
                # the current line already).
                while line.strip()[-1] != '(':
                    line = next(islice(f, 1))

                if key == 'mixed':
                    # From
                    # <http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf>:
                    # > If the face zone is of mixed type (element-type = 0),
                    # > the body of the section will include the face type and
                    # > will appear as follows
                    # >
                    # > type v0 v1 v2 c0 c1
                    # >
                    data = {}
                    for k in range(num_cells):
                        line = next(islice(f, 1))
                        dat = line.split()
                        type_index = int(dat[0], 16)
                        assert type_index != 0
                        type_string, num_nodes_per_cell = \
                            element_type_to_key_num_nodes[type_index]
                        assert len(dat) == num_nodes_per_cell + 3

                        if type_string not in data:
                            data[type_string] = []

                        data[type_string].append([
                            int(d, 16) for d in dat[1:num_nodes_per_cell+1]
                            ])

                    data = {key: numpy.array(data[key]) for key in data}

                else:
                    # read cell data
                    data = numpy.empty(
                        (num_cells, num_nodes_per_cell), dtype=int
                        )
                    for k in range(num_cells):
                        line = next(islice(f, 1))
                        dat = line.split()
                        # The body of a regular face section contains the grid
                        # connectivity, and each line appears as follows:
                        #   n0 n1 n2 cr cl
                        # where n* are the defining nodes (vertices) of the
                        # face, and c* are the adjacent cells.
                        assert len(dat) == num_nodes_per_cell + 2
                        data[k] = [
                            int(d, 16) for d in dat[:num_nodes_per_cell]
                            ]
                    data = {key: data}

                for key in data:
                    if key in cells:
                        cells[key] = numpy.concatenate([cells[key], data[key]])
                    else:
                        cells[key] = data[key]

                # make sure that the data set is properly closed
                _skip_till_two_closing_brackets(f)
            else:
                warnings.warn('Unknown index \'%s\'. Skipping.' % index)
                # Skipping ahead to the next line with two closing brackets.
                while re.search('[^\)]*\)\s*\)\s*$', line) is None:
                    line = next(islice(f, 1))

    points = numpy.concatenate(points)

    # Gauge the cells with the first point_index.
    for key in cells:
        cells[key] -= first_point_index_overall

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
        first_node_index = 1
        fh.write((
            '(10 (0 %x %x 0))\n' % (first_node_index, len(points))
            ).encode('utf8'))

        # total number of cells
        total_num_cells = sum([len(c) for c in cells])
        fh.write(('(12 (0 1 %x 0))\n' % total_num_cells).encode('utf8'))

        # Write nodes
        fh.write((
            '(10 (1 %x %x 1 %x))(\n' %
            (first_node_index, points.shape[0], points.shape[1])
            ).encode('utf8'))
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
            numpy.savetxt(fh, values + first_node_index, fmt='%x')
            fh.write(('))\n').encode('utf8'))
            first_index = last_index + 1

    return
