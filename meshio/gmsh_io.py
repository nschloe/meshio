# -*- coding: utf-8 -*-
#
'''
I/O for Gmsh's msh format, cf.
<http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from itertools import islice
import logging
import numpy
import struct

from . import helpers


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        points, cells, point_data, cell_data, field_data = read_buffer(f)

    return points, cells, point_data, cell_data, field_data


def read_buffer(f):
    # The format is specified at
    # <http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format>.

    # Initialize the data optional data fields
    cells = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    has_additional_tag_data = False
    is_ascii = None
    int_size = 4
    data_size = None
    while True:
        try:
            line = next(islice(f, 1)).decode('utf-8')
        except StopIteration:
            break
        assert line[0] == '$'
        environ = line[1:].strip()
        if environ == 'MeshFormat':
            line = next(islice(f, 1)).decode('utf-8')
            # Split the line
            # 2.2 0 8
            # into its components.
            str_list = list(filter(None, line.split()))
            assert str_list[0][0] == '2', 'Need mesh format 2'
            assert str_list[1] in ['0', '1']
            is_ascii = str_list[1] == '0'
            data_size = int(str_list[2])
            if not is_ascii:
                # The next line is the integer one in bytes. Useful to check
                # endianness. Just assert that we get 1 here.
                one = f.read(int_size)
                assert struct.unpack('i', one)[0] == 1
                line = next(islice(f, 1)).decode('utf-8')
                assert line == '\n'
            line = next(islice(f, 1)).decode('utf-8')
            assert line.strip() == '$EndMeshFormat'
        elif environ == 'PhysicalNames':
            line = next(islice(f, 1))
            num_phys_names = int(line)
            for k, line in enumerate(islice(f, num_phys_names)):
                key = line.split(' ')[2].replace('"', '').replace('\n', '')
                phys_group = int(line.split(' ')[1])
                field_data[key] = phys_group
            line = next(islice(f, 1))
            assert line.strip() == '$EndPhysicalNames'
        elif environ == 'Nodes':
            # The first line is the number of nodes
            line = next(islice(f, 1))
            num_nodes = int(line)
            points = numpy.empty((num_nodes, 3))
            if is_ascii:
                # TODO speed up with http://stackoverflow.com/a/41642053/353337
                for k, line in enumerate(islice(f, num_nodes)):
                    # Throw away the index immediately
                    points[k, :] = numpy.array(line.split(), dtype=float)[1:]
            else:
                # binary
                num_bytes = num_nodes * (int_size + 3 * data_size)
                assert numpy.int32(0).nbytes == int_size
                assert numpy.float64(0.0).nbytes == data_size
                dtype = [('index', numpy.int32), ('x', numpy.float64, (3,))]
                data = numpy.fromstring(f.read(num_bytes), dtype=dtype)
                assert (data['index'] == range(1, num_nodes+1)).all()
                points = data['x']
                line = next(islice(f, 1)).decode('utf-8')
                assert line == '\n'

            line = next(islice(f, 1)).decode('utf-8')
            assert line.strip() == '$EndNodes'
        else:
            assert environ == 'Elements', \
                'Unknown environment \'%s\'.' % environ
            # The first line is the number of elements
            line = next(islice(f, 1)).decode('utf-8')
            total_num_cells = int(line)
            gmsh_to_meshio_type = {
                    15: 'vertex',
                    1: 'line',
                    2: 'triangle',
                    3: 'quad',
                    4: 'tetra',
                    5: 'hexahedron',
                    6: 'wedge',
                    7: 'pyramid',
                    8: 'line3',
                    9: 'triangle6',
                    10: 'quad9',
                    11: 'tetra10',
                    12: 'hexahedron27',
                    13: 'prism18',
                    14: 'pyramid14',
                    26: 'line4',
                    36: 'quad16',
                    }
            # For each cell, there are at least two tags: the physical entity
            # and the elementary geometrical entity the cell belongs to (see
            # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>).
            if is_ascii:
                for k, line in enumerate(islice(f, total_num_cells)):
                    data = [int[k] for k in filter(None, line.split())]
                    t = gmsh_to_meshio_type[data[1]][0]
                    num_nodes_per_elem = helpers.num_nodes_per_cell[t]

                    if t not in cells:
                        cells[t] = []
                    cells[t].append(data[num_nodes_per_elem:-1])

                    # data[2] gives the number of tags. The gmsh manual
                    # <http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format>
                    # says:
                    # >>>
                    # By default, the first tag is the number of the physical
                    # entity to which the element belongs; the second is the
                    # number of the elementary geometrical entity to which the
                    # element belongs; the third is the number of mesh
                    # partitions to which the element belongs, followed by the
                    # partition ids (negative partition ids indicate ghost
                    # cells). A zero tag is equivalent to no tag. Gmsh and most
                    # codes using the MSH 2 format require at least the first
                    # two tags (physical and elementary tags).
                    # <<<
                    num_tags = data[2]
                    if t not in cell_data:
                        cell_data[t] = []
                    cell_data[t].append(data[3:3+num_tags])

                # convert to numpy arrays
                for key in cells:
                    cells[key] = numpy.array(cells[key], dtype=int)
                for key in cell_data:
                    cell_data[key] = numpy.array(cell_data[key], dtype=int)
            else:
                # binary
                num_elems = 0
                while num_elems < total_num_cells:
                    # read element header
                    elem_type = struct.unpack('i', f.read(int_size))[0]
                    t = gmsh_to_meshio_type[elem_type]
                    num_nodes_per_elem = helpers.num_nodes_per_cell[t]
                    num_elems0 = struct.unpack('i', f.read(int_size))[0]
                    num_tags = struct.unpack('i', f.read(int_size))[0]
                    assert num_tags >= 2

                    # read element data
                    num_bytes = 4 * (
                        num_elems0 * (1 + num_tags + num_nodes_per_elem)
                        )
                    shape = \
                        (num_elems0, 1 + num_tags + num_nodes_per_elem)
                    data = numpy.fromstring(
                        f.read(num_bytes), dtype=numpy.int32
                        ).reshape(shape)

                    if t not in cells:
                        cells[t] = []
                    cells[t].append(data[:, -num_nodes_per_elem:])

                    if t not in cell_data:
                        cell_data[t] = []
                    cell_data[t].append(data[:, 1:num_tags+1])

                    num_elems += num_elems0

                # collect cells
                for key in cells:
                    cells[key] = numpy.vstack(cells[key])

                # collect cell data
                for key in cell_data:
                    cell_data[key] = numpy.vstack(cell_data[key])

                assert next(islice(f, 1)).decode('utf-8') == '\n'

            line = next(islice(f, 1)).decode('utf-8')
            assert line.strip() == '$EndElements'

            # Subtract one to account for the fact that python indices are
            # 0-based.
            for key in cells:
                cells[key] -= 1

            # restrict to the standard two data items
            for key in cell_data:
                if cell_data[key].shape[1] > 2:
                    has_additional_tag_data = True
                cell_data[key] = {
                        'physical': cell_data[key][:, 0],
                        'geometrical': cell_data[key][:, 1],
                        }

    if has_additional_tag_data:
        logging.warning(
            'The file contains tag data that couldn\'t be processed.'
            )

    return points, cells, point_data, cell_data, field_data


def write(
        mode,
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    '''Writes msh files, cf.
    http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
    '''
    if point_data is None:
        point_data = {}
    if cell_data is None:
        cell_data = {}
    if field_data is None:
        field_data = {}

    assert mode in ['ascii', 'binary']

    with open(filename, 'wb') as fh:
        mode_idx = 0 if mode == 'ascii' else 1
        size_of_double = 8
        fh.write((
            '$MeshFormat\n2.2 %d %d\n' % (mode_idx, size_of_double)
            ).encode('utf-8'))
        if mode == 'binary':
            fh.write(struct.pack('i', 1))
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndMeshFormat\n'.encode('utf-8'))

        # Write nodes
        fh.write('$Nodes\n'.encode('utf-8'))
        fh.write(('%d\n' % len(points)).encode('utf-8'))
        if mode == 'ascii':
            for k, x in enumerate(points):
                fh.write(
                    ('%d %r %r %r\n' % (k+1, x[0], x[1], x[2])).encode('utf-8')
                    )
        else:
            assert mode == 'binary'
            # TODO write at once
            for k, x in enumerate(points):
                fh.write(struct.pack('i', k+1))
                fh.write(struct.pack('d', x[0]))
                fh.write(struct.pack('d', x[1]))
                fh.write(struct.pack('d', x[2]))
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndNodes\n'.encode('utf-8'))

        # Translate meshio types to gmsh codes
        # http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
        meshio_to_gmsh_type = {
                'vertex': 15,
                'line': 1,
                'triangle': 2,
                'quad': 3,
                'tetra': 4,
                'hexahedron': 5,
                'wedge': 6,
                }
        fh.write('$Elements\n'.encode('utf-8'))
        # count all cells
        total_num_cells = sum([data.shape[0] for _, data in cells.items()])
        fh.write(('%d\n' % total_num_cells).encode('utf-8'))

        consecutive_index = 0
        for cell_type, node_idcs in cells.items():
            # handle cell data
            if cell_type in cell_data and cell_data[cell_type]:
                for key in cell_data[cell_type]:
                    # assert data consistency
                    assert len(cell_data[cell_type][key]) == len(node_idcs)
                    # TODO assert that the data type is int

                # if a tag is present, make sure that there are 'physical' and
                # 'geometrical' as well.
                if 'physical' not in cell_data[cell_type]:
                    cell_data[cell_type]['physical'] = \
                        numpy.ones(len(node_idcs))
                if 'geometrical' not in cell_data[cell_type]:
                    cell_data[cell_type]['geometrical'] = \
                        numpy.ones(len(node_idcs))

                # 'physical' and 'geometrical' go first; this is what the gmsh
                # file format prescribes
                keywords = list(cell_data[cell_type].keys())
                keywords.remove('physical')
                keywords.remove('geometrical')
                sorted_keywords = ['physical', 'geometrical'] + keywords
                fcd = numpy.column_stack([
                        cell_data[cell_type][key] for key in sorted_keywords
                        ])
            else:
                # no cell data
                fcd = numpy.empty([len(node_idcs), 0])

            num_nodes_per_cell = node_idcs.shape[1]
            if mode == 'ascii':
                form = '%d ' + '%d' % meshio_to_gmsh_type[cell_type] \
                    + ' %d' % fcd.shape[1] + ' %d' * fcd.shape[1] \
                    + ' ' + ' '.join(num_nodes_per_cell * ['%d']) \
                    + '\n'
                for k, c in enumerate(node_idcs):
                    fh.write((
                        form % (
                            (consecutive_index + k + 1,) +
                            tuple(fcd[k]) +
                            tuple(c + 1)
                            )
                        ).encode('utf-8'))
            else:
                assert mode == 'binary'
                # header
                fh.write(struct.pack('i', meshio_to_gmsh_type[cell_type]))
                fh.write(struct.pack('i', node_idcs.shape[0]))
                fh.write(struct.pack('i', fcd.shape[1]))
                # fh.write('\n'.encode('utf-8'))
                # actual data
                # TODO write at once
                for k, c in enumerate(node_idcs):
                    fh.write(struct.pack('i', consecutive_index + k + 1))
                    fh.write(fcd[k].tobytes())
                    fh.write((c + 1).tobytes())
                    break
                fh.write('\n'.encode('utf-8'))
            consecutive_index += len(node_idcs)
        fh.write('$EndElements'.encode('utf-8'))

    return
