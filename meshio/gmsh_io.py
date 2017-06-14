# -*- coding: utf-8 -*-
#
'''
I/O for Gmsh's msh format, cf.
<http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging
import struct

import numpy

num_nodes_per_cell = {
    'vertex': 1,
    'line': 2,
    'triangle': 3,
    'quad': 4,
    'tetra': 4,
    'hexahedron': 8,
    'wedge': 6,
    'pyramid': 5,
    #
    'line3': 3,
    'triangle6': 6,
    'quad9': 9,
    'tetra10': 10,
    'hexahedron27': 27,
    'prism18': 18,
    'pyramid14': 14,
    'line4': 4,
    'quad16': 16,
    }

# Translate meshio types to gmsh codes
# http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
_gmsh_to_meshio_type = {
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
_meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        out = read_buffer(f)
    return out


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
        line = f.readline().decode('utf-8')
        if not line:
            # EOF
            break
        assert line[0] == '$'
        environ = line[1:].strip()
        if environ == 'MeshFormat':
            line = f.readline().decode('utf-8')
            # Split the line
            # 2.2 0 8
            # into its components.
            str_list = list(filter(None, line.split()))
            assert str_list[0][0] == '2', 'Need mesh format 2'
            assert str_list[1] in ['0', '1']
            is_ascii = str_list[1] == '0'
            data_size = int(str_list[2])
            if not is_ascii:
                # The next line is the integer 1 in bytes. Useful to check
                # endianness. Just assert that we get 1 here.
                one = f.read(int_size)
                assert struct.unpack('i', one)[0] == 1
                line = f.readline().decode('utf-8')
                assert line == '\n'
            line = f.readline().decode('utf-8')
            assert line.strip() == '$EndMeshFormat'
        elif environ == 'PhysicalNames':
            line = f.readline().decode('utf-8')
            num_phys_names = int(line)
            for _ in range(num_phys_names):
                line = f.readline().decode('utf-8')
                key = line.split(' ')[2].replace('"', '').replace('\n', '')
                phys_group = int(line.split(' ')[1])
                field_data[key] = phys_group
            line = f.readline().decode('utf-8')
            assert line.strip() == '$EndPhysicalNames'
        elif environ == 'Nodes':
            # The first line is the number of nodes
            line = f.readline().decode('utf-8')
            num_nodes = int(line)
            if is_ascii:
                points = numpy.fromfile(
                    f, count=num_nodes*4, sep=' '
                    ).reshape((num_nodes, 4))
                # The first number is the index
                points = points[:, 1:]
            else:
                # binary
                num_bytes = num_nodes * (int_size + 3 * data_size)
                assert numpy.int32(0).nbytes == int_size
                assert numpy.float64(0.0).nbytes == data_size
                dtype = [('index', numpy.int32), ('x', numpy.float64, (3,))]
                data = numpy.fromstring(f.read(num_bytes), dtype=dtype)
                assert (data['index'] == range(1, num_nodes+1)).all()
                points = data['x']
                line = f.readline().decode('utf-8')
                assert line == '\n'

            line = f.readline().decode('utf-8')
            assert line.strip() == '$EndNodes'
        else:
            assert environ == 'Elements', \
                'Unknown environment \'{}\'.'.format(environ)
            # The first line is the number of elements
            line = f.readline().decode('utf-8')
            total_num_cells = int(line)
            if is_ascii:
                for _ in range(total_num_cells):
                    line = f.readline().decode('utf-8')
                    data = [int(k) for k in filter(None, line.split())]
                    t = _gmsh_to_meshio_type[data[1]]
                    num_nodes_per_elem = num_nodes_per_cell[t]

                    if t not in cells:
                        cells[t] = []
                    cells[t].append(data[-num_nodes_per_elem:])

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
                    t = _gmsh_to_meshio_type[elem_type]
                    num_nodes_per_elem = num_nodes_per_cell[t]
                    num_elems0 = struct.unpack('i', f.read(int_size))[0]
                    num_tags = struct.unpack('i', f.read(int_size))[0]
                    # assert num_tags >= 2

                    # read element data
                    num_bytes = 4 * (
                        num_elems0 * (1 + num_tags + num_nodes_per_elem)
                        )
                    shape = \
                        (num_elems0, 1 + num_tags + num_nodes_per_elem)
                    b = f.read(num_bytes)
                    data = numpy.fromstring(
                        b, dtype=numpy.int32
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

                line = f.readline().decode('utf-8')
                assert line == '\n'

            line = f.readline().decode('utf-8')
            assert line.strip() == '$EndElements'

            # Subtract one to account for the fact that python indices are
            # 0-based.
            for key in cells:
                cells[key] -= 1

            # restrict to the standard two data items
            output_cell_data = {}
            for key in cell_data:
                if cell_data[key].shape[1] > 2:
                    has_additional_tag_data = True
                output_cell_data[key] = {}
                if cell_data[key].shape[1] > 0:
                    output_cell_data[key]['physical'] = cell_data[key][:, 0]
                if cell_data[key].shape[1] > 1:
                    output_cell_data[key]['geometrical'] = cell_data[key][:, 0]
            cell_data = output_cell_data

    if has_additional_tag_data:
        logging.warning(
            'The file contains tag data that couldn\'t be processed.'
            )

    return points, cells, point_data, cell_data, field_data


def write(
        filename,
        points,
        cells,
        is_ascii=False,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    '''Writes msh files, cf.
    http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
    '''
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    if not is_ascii:
        for key in cells:
            if cells[key].dtype != numpy.int32:
                logging.warning(
                    'Binary Gmsh needs 32-bit integers. Converting.'
                    )
                cells[key] = numpy.array(cells[key], dtype=numpy.int32)

    with open(filename, 'wb') as fh:
        mode_idx = 0 if is_ascii else 1
        size_of_double = 8
        fh.write((
            '$MeshFormat\n2.2 {} {}\n'.format(mode_idx, size_of_double)
            ).encode('utf-8'))
        if not is_ascii:  # binary
            fh.write(struct.pack('i', 1))
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndMeshFormat\n'.encode('utf-8'))

        # Write nodes
        fh.write('$Nodes\n'.encode('utf-8'))
        fh.write('{}\n'.format(len(points)).encode('utf-8'))
        if is_ascii:
            for k, x in enumerate(points):
                fh.write(
                    '{} {!r} {!r} {!r}\n'.format(k+1, x[0], x[1], x[2])
                    .encode('utf-8')
                    )
        else:
            dtype = [('index', numpy.int32), ('x', numpy.float64, (3,))]
            tmp = numpy.empty(len(points), dtype=dtype)
            tmp['index'] = 1 + numpy.arange(len(points))
            tmp['x'] = points
            fh.write(tmp.tostring())
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndNodes\n'.encode('utf-8'))

        fh.write('$Elements\n'.encode('utf-8'))
        # count all cells
        total_num_cells = sum([data.shape[0] for _, data in cells.items()])
        fh.write('{}\n'.format(total_num_cells).encode('utf-8'))

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
                        numpy.ones(len(node_idcs), dtype=numpy.int32)
                if 'geometrical' not in cell_data[cell_type]:
                    cell_data[cell_type]['geometrical'] = \
                        numpy.ones(len(node_idcs), dtype=numpy.int32)

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
                fcd = numpy.empty([len(node_idcs), 0], dtype=numpy.int32)

            if is_ascii:
                form = '{} ' + str(_meshio_to_gmsh_type[cell_type]) \
                    + ' ' + str(fcd.shape[1]) \
                    + ' {} {}\n'
                for k, c in enumerate(node_idcs):
                    fh.write(
                        form.format(
                            consecutive_index + k + 1,
                            ' '.join([str(val) for val in fcd[k]]),
                            ' '.join([str(cc + 1) for cc in c])
                            ).encode('utf-8')
                        )
            else:
                # header
                fh.write(struct.pack('i', _meshio_to_gmsh_type[cell_type]))
                fh.write(struct.pack('i', node_idcs.shape[0]))
                fh.write(struct.pack('i', fcd.shape[1]))
                # actual data
                a = numpy.arange(
                    len(node_idcs), dtype=numpy.int32
                    )[:, numpy.newaxis]
                a += 1 + consecutive_index
                array = numpy.hstack([a, fcd, node_idcs + 1])
                fh.write(array.tostring())

            consecutive_index += len(node_idcs)
        if not is_ascii:
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndElements'.encode('utf-8'))

    return
