# -*- coding: utf-8 -*-
#
'''
I/O for Gmsh's msh format, cf.
<http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging
import struct

import numpy

from .vtk_io import raw_from_cell_data


num_nodes_per_cell = {
    'vertex': 1,
    'line': 2,
    'triangle': 3,
    'quad': 4,
    'quad8': 8,
    'tetra': 4,
    'hexahedron': 8,
    'hexahedron20': 20,
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
    #
    'line4': 4,
    'triangle10': 10,
    'quad16': 16,
    'tetra20': 20,
    'hexahedron64': 64,
    #
    'line5': 5,
    'triangle15': 15,
    'quad25': 25,
    'tetra35': 35,
    'hexahedron125': 125,
    #
    'line6': 6,
    'triangle21': 21,
    'quad36': 36,
    'tetra56': 56,
    'hexahedron216': 216,
    }

# Translate meshio types to gmsh codes
# http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format
_gmsh_to_meshio_type = {
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
    15: 'vertex',
    16: 'quad8',
    17: 'hexahedron20',
    21: 'triangle10',
    23: 'triangle15',
    25: 'triangle21',
    26: 'line4',
    27: 'line5',
    28: 'line6',
    29: 'tetra20',
    30: 'tetra35',
    31: 'tetra56',
    36: 'quad16',
    37: 'quad25',
    38: 'quad36',
    92: 'hexahedron64',
    93: 'hexahedron125',
    94: 'hexahedron216',
    }
_meshio_to_gmsh_type = {v: k for k, v in _gmsh_to_meshio_type.items()}


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        out = read_buffer(f)
    return out


def _read_header(f, int_size):
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
        # The next line is the integer 1 in bytes. Useful for checking
        # endianness. Just assert that we get 1 here.
        one = f.read(int_size)
        assert struct.unpack('i', one)[0] == 1
        line = f.readline().decode('utf-8')
        assert line == '\n'
    line = f.readline().decode('utf-8')
    assert line.strip() == '$EndMeshFormat'
    return data_size, is_ascii


def _read_physical_names(f, field_data):
    line = f.readline().decode('utf-8')
    num_phys_names = int(line)
    for _ in range(num_phys_names):
        line = f.readline().decode('utf-8')
        key = line.split(' ')[2].replace('"', '').replace('\n', '')
        phys_group = int(line.split(' ')[1])
        phys_dim = int(line.split(' ')[0])
        value = numpy.array([phys_group, phys_dim], dtype=int)
        field_data[key] = value
    line = f.readline().decode('utf-8')
    assert line.strip() == '$EndPhysicalNames'
    return


def _read_nodes(f, is_ascii, int_size, data_size):
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
        points = numpy.ascontiguousarray(data['x'])
        line = f.readline().decode('utf-8')
        assert line == '\n'

    line = f.readline().decode('utf-8')
    assert line.strip() == '$EndNodes'
    return points


def _read_cells(f, cells, int_size, is_ascii):
    # The first line is the number of elements
    line = f.readline().decode('utf-8')
    total_num_cells = int(line)
    has_additional_tag_data = False
    cell_tags = {}
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
            # By default, the first tag is the number of the physical entity to
            # which the element belongs; the second is the number of the
            # elementary geometrical entity to which the element belongs; the
            # third is the number of mesh partitions to which the element
            # belongs, followed by the partition ids (negative partition ids
            # indicate ghost cells). A zero tag is equivalent to no tag. Gmsh
            # and most codes using the MSH 2 format require at least the first
            # two tags (physical and elementary tags).
            # <<<
            num_tags = data[2]
            if t not in cell_tags:
                cell_tags[t] = []
            cell_tags[t].append(data[3:3+num_tags])

        # convert to numpy arrays
        for key in cells:
            cells[key] = numpy.array(cells[key], dtype=int)
        for key in cell_tags:
            cell_tags[key] = numpy.array(cell_tags[key], dtype=int)
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

            if t not in cell_tags:
                cell_tags[t] = []
            cell_tags[t].append(data[:, 1:num_tags+1])

            num_elems += num_elems0

        # collect cells
        for key in cells:
            cells[key] = numpy.vstack(cells[key])

        # collect cell tags
        for key in cell_tags:
            cell_tags[key] = numpy.vstack(cell_tags[key])

        line = f.readline().decode('utf-8')
        assert line == '\n'

    line = f.readline().decode('utf-8')
    assert line.strip() == '$EndElements'

    # Subtract one to account for the fact that python indices are
    # 0-based.
    for key in cells:
        cells[key] -= 1

    # restrict to the standard two data items (physical, geometrical)
    output_cell_tags = {}
    for key in cell_tags:
        if cell_tags[key].shape[1] > 2:
            has_additional_tag_data = True
        output_cell_tags[key] = {}
        if cell_tags[key].shape[1] > 0:
            output_cell_tags[key]['gmsh:physical'] = cell_tags[key][:, 0]
        if cell_tags[key].shape[1] > 1:
            output_cell_tags[key]['gmsh:geometrical'] = cell_tags[key][:, 1]
    return has_additional_tag_data, output_cell_tags


def _read_data(f, tag, data_dict, int_size, data_size, is_ascii):
    # Read string tags
    num_string_tags = int(f.readline().decode('utf-8'))
    string_tags = [
        f.readline().decode('utf-8').strip()
        for _ in range(num_string_tags)
        ]
    # The real tags typically only contain one value, the time.
    # Discard it.
    num_real_tags = int(f.readline().decode('utf-8'))
    for _ in range(num_real_tags):
        f.readline()
    num_integer_tags = int(f.readline().decode('utf-8'))
    integer_tags = [
        int(f.readline().decode('utf-8'))
        for _ in range(num_integer_tags)
        ]
    num_components = integer_tags[1]
    num_items = integer_tags[2]
    if is_ascii:
        data = numpy.fromfile(
            f, count=num_items*(1+num_components), sep=' '
            ).reshape((num_items, 1+num_components))
        # The first number is the index
        data = data[:, 1:]
    else:
        # binary
        num_bytes = num_items * (int_size + num_components * data_size)
        assert numpy.int32(0).nbytes == int_size
        assert numpy.float64(0.0).nbytes == data_size
        dtype = [
            ('index', numpy.int32),
            ('values', numpy.float64, (num_components,))
            ]
        data = numpy.fromstring(f.read(num_bytes), dtype=dtype)
        assert (data['index'] == range(1, num_items+1)).all()
        data = numpy.ascontiguousarray(data['values'])
        line = f.readline().decode('utf-8')
        assert line == '\n'

    line = f.readline().decode('utf-8')
    assert line.strip() == '$End{}'.format(tag)

    # The gmsh format cannot distingiush between data of shape (n,) and (n, 1).
    # If shape[1] == 1, cut it off.
    if data.shape[1] == 1:
        data = data[:, 0]

    data_dict[string_tags[0]] = data
    return


def read_buffer(f):
    # The format is specified at
    # <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.

    # Initialize the optional data fields
    points = []
    cells = {}
    field_data = {}
    cell_data_raw = {}
    cell_tags = {}
    point_data = {}

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
            data_size, is_ascii = _read_header(f, int_size)
        elif environ == 'PhysicalNames':
            _read_physical_names(f, field_data)
        elif environ == 'Nodes':
            points = _read_nodes(f, is_ascii, int_size, data_size)
        elif environ == 'Elements':
            has_additional_tag_data, cell_tags = \
                _read_cells(f, cells, int_size, is_ascii)
        elif environ == 'NodeData':
            _read_data(
                f, 'NodeData', point_data, int_size, data_size, is_ascii
                )
        else:
            assert environ == 'ElementData', \
                'Unknown environment \'{}\'.'.format(environ)
            _read_data(
                f, 'ElementData', cell_data_raw, int_size, data_size, is_ascii
                )

    if has_additional_tag_data:
        logging.warning(
            'The file contains tag data that couldn\'t be processed.'
            )

    cell_data = cell_data_from_raw(cells, cell_data_raw)

    # merge cell_tags into cell_data
    for key, tag_dict in cell_tags.items():
        if key not in cell_data:
            cell_data[key] = {}
        for name, item_list in tag_dict.items():
            assert name not in cell_data[key]
            cell_data[key][name] = item_list

    return points, cells, point_data, cell_data, field_data


def cell_data_from_raw(cells, cell_data_raw):
    cell_data = {k: {} for k in cells}
    for key in cell_data_raw:
        d = cell_data_raw[key]
        r = 0
        for k in cells:
            cell_data[k][key] = d[r:r+len(cells[k])]
            r += len(cells[k])

    return cell_data


def _write_physical_names(fh, field_data):
    # Write physical names
    entries = []
    for phys_name in field_data:
        try:
            phys_num, phys_dim = field_data[phys_name]
            phys_num, phys_dim = int(phys_num), int(phys_dim)
            entries.append((phys_dim, phys_num, phys_name))
        except (ValueError, TypeError):
            logging.warning(
                'Field data contains entry that cannot be processed.'
            )
    entries.sort()
    if entries:
        fh.write('$PhysicalNames\n'.encode('utf-8'))
        fh.write('{}\n'.format(len(entries)).encode('utf-8'))
        for entry in entries:
            fh.write('{} {} "{}"\n'.format(*entry).encode('utf-8'))
        fh.write('$EndPhysicalNames\n'.encode('utf-8'))
    return


def _write_nodes(fh, points, write_binary):
    fh.write('$Nodes\n'.encode('utf-8'))
    fh.write('{}\n'.format(len(points)).encode('utf-8'))
    if write_binary:
        dtype = [('index', numpy.int32), ('x', numpy.float64, (3,))]
        tmp = numpy.empty(len(points), dtype=dtype)
        tmp['index'] = 1 + numpy.arange(len(points))
        tmp['x'] = points
        fh.write(tmp.tostring())
        fh.write('\n'.encode('utf-8'))
    else:
        for k, x in enumerate(points):
            fh.write(
                '{} {!r} {!r} {!r}\n'.format(k+1, x[0], x[1], x[2])
                .encode('utf-8')
                )
    fh.write('$EndNodes\n'.encode('utf-8'))
    return


def _write_elements(fh, cells, tag_data, write_binary):
    # write elements
    fh.write('$Elements\n'.encode('utf-8'))
    # count all cells
    total_num_cells = sum([data.shape[0] for _, data in cells.items()])
    fh.write('{}\n'.format(total_num_cells).encode('utf-8'))

    consecutive_index = 0
    for cell_type, node_idcs in cells.items():
        tags = []
        for key in ['gmsh:physical', 'gmsh:geometrical']:
            try:
                tags.append(tag_data[cell_type][key])
            except KeyError:
                pass
        fcd = numpy.concatenate([tags]).T

        # pylint: disable=len-as-condition
        if len(fcd) == 0:
            fcd = numpy.empty((len(node_idcs), 0), dtype=numpy.int32)

        if write_binary:
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
        else:
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

        consecutive_index += len(node_idcs)
    if write_binary:
        fh.write('\n'.encode('utf-8'))
    fh.write('$EndElements\n'.encode('utf-8'))
    return


def _write_data(fh, tag, name, data, write_binary):
    fh.write('${}\n'.format(tag).encode('utf-8'))
    # <http://gmsh.info/doc/texinfo/gmsh.html>:
    # > Number of string tags.
    # > gives the number of string tags that follow. By default the first
    # > string-tag is interpreted as the name of the post-processing view and
    # > the second as the name of the interpolation scheme. The interpolation
    # > scheme is provided in the $InterpolationScheme section (see below).
    fh.write('{}\n'.format(1).encode('utf-8'))
    fh.write('{}\n'.format(name).encode('utf-8'))
    fh.write('{}\n'.format(1).encode('utf-8'))
    fh.write('{}\n'.format(0.0).encode('utf-8'))
    # three integer tags:
    fh.write('{}\n'.format(3).encode('utf-8'))
    # time step
    fh.write('{}\n'.format(0).encode('utf-8'))
    # number of components
    num_components = data.shape[1] if len(data.shape) > 1 else 1
    assert num_components in [1, 3, 9], \
        'Gmsh only permits 1, 3, or 9 components per data field.'
    fh.write('{}\n'.format(num_components).encode('utf-8'))
    # num data items
    fh.write('{}\n'.format(data.shape[0]).encode('utf-8'))
    # actually write the data
    if write_binary:
        dtype = [
            ('index', numpy.int32),
            ('data', numpy.float64, num_components)
            ]
        tmp = numpy.empty(len(data), dtype=dtype)
        tmp['index'] = 1 + numpy.arange(len(data))
        tmp['data'] = data
        fh.write(tmp.tostring())
        fh.write('\n'.encode('utf-8'))
    else:
        fmt = ' '.join(['{}'] + ['{!r}'] * num_components) + '\n'
        # TODO unify
        if num_components == 1:
            for k, x in enumerate(data):
                fh.write(fmt.format(k+1, x).encode('utf-8'))
        else:
            for k, x in enumerate(data):
                fh.write(fmt.format(k+1, *x).encode('utf-8'))

    fh.write('$End{}\n'.format(tag).encode('utf-8'))
    return


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None,
          write_binary=True):
    '''Writes msh files, cf.
    <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    '''
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    if write_binary:
        for key in cells:
            if cells[key].dtype != numpy.int32:
                logging.warning(
                    'Binary Gmsh needs 32-bit integers (got %s). Converting.',
                    cells[key].dtype
                    )
                cells[key] = numpy.array(cells[key], dtype=numpy.int32)

    with open(filename, 'wb') as fh:
        mode_idx = 1 if write_binary else 0
        size_of_double = 8
        fh.write((
            '$MeshFormat\n2.2 {} {}\n'.format(mode_idx, size_of_double)
            ).encode('utf-8'))
        if write_binary:
            fh.write(struct.pack('i', 1))
            fh.write('\n'.encode('utf-8'))
        fh.write('$EndMeshFormat\n'.encode('utf-8'))

        if field_data:
            _write_physical_names(fh, field_data)

        # Split the cell data: gmsh:physical and gmsh:geometrical are tags, the
        # rest is actual cell data.
        tag_data = {}
        other_data = {}
        for cell_type, a in cell_data.items():
            tag_data[cell_type] = {}
            other_data[cell_type] = {}
            for key, data in a.items():
                if key in ['gmsh:physical', 'gmsh:geometrical']:
                    tag_data[cell_type][key] = data
                else:
                    other_data[cell_type][key] = data

        _write_nodes(fh, points, write_binary)
        _write_elements(fh, cells, tag_data, write_binary)
        for name, dat in point_data.items():
            _write_data(fh, 'NodeData', name, dat, write_binary)
        cell_data_raw = raw_from_cell_data(other_data)
        for name, dat in cell_data_raw.items():
            _write_data(fh, 'ElementData', name, dat, write_binary)

    return
