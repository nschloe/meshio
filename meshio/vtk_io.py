# -*- coding: utf-8 -*-
#
'''
I/O for VTK <https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging
import numpy

from .__about__ import __version__


# https://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
vtk_to_meshio_type = {
    0: 'empty',
    1: 'vertex',
    # 2: 'poly_vertex',
    3: 'line',
    # 4: 'poly_line',
    5: 'triangle',
    # 6: 'triangle_strip',
    # 7: 'polygon',
    # 8: 'pixel',
    9: 'quad',
    10: 'tetra',
    # 11: 'voxel',
    12: 'hexahedron',
    13: 'wedge',
    14: 'pyramid',
    15: 'penta_prism',
    16: 'hexa_prism',
    21: 'line3',
    22: 'triangle6',
    23: 'quad8',
    24: 'tetra10',
    25: 'hexahedron20',
    26: 'wedge15',
    27: 'pyramid13',
    28: 'quad9',
    29: 'hexahedron27',
    30: 'quad6',
    31: 'wedge12',
    32: 'wedge18',
    33: 'hexahedron24',
    34: 'triangle7',
    35: 'line4',
    #
    # 60: VTK_HIGHER_ORDER_EDGE,
    # 61: VTK_HIGHER_ORDER_TRIANGLE,
    # 62: VTK_HIGHER_ORDER_QUAD,
    # 63: VTK_HIGHER_ORDER_POLYGON,
    # 64: VTK_HIGHER_ORDER_TETRAHEDRON,
    # 65: VTK_HIGHER_ORDER_WEDGE,
    # 66: VTK_HIGHER_ORDER_PYRAMID,
    # 67: VTK_HIGHER_ORDER_HEXAHEDRON,
    }
meshio_to_vtk_type = {v: k for k, v in vtk_to_meshio_type.items()}


# These are all VTK data types. One sometimes finds 'vtktypeint64', but
# this is ill-formed.
vtk_to_numpy_dtype = {
    'bit': numpy.dtype('bool'),
    'unsigned_char': numpy.dtype('uint8'),
    'char': numpy.dtype('int8'),
    'unsigned_short': numpy.dtype('uint16'),
    'short': numpy.dtype('int16'),
    'unsigned_int': numpy.dtype('uint32'),
    'int': numpy.dtype('int32'),
    'unsigned_long': numpy.dtype('int64'),
    'long': numpy.dtype('int64'),
    'float': numpy.dtype('float32'),
    'double': numpy.dtype('float64'),
    }

numpy_to_vtk_dtype = {v: k for k, v in vtk_to_numpy_dtype.items()}


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # initialize output data
    points = None
    field_data = {}
    cell_data_raw = {}
    point_data = {}

    # skip header and title
    f.readline()
    f.readline()

    data_type = f.readline().decode('utf-8').strip()
    assert data_type in ['ASCII', 'BINARY'], \
        'Unknown VTK data type \'{}\'.'.format(data_type)
    is_ascii = data_type == 'ASCII'

    c = None
    offsets = None
    ct = None

    # One of the problem in reading VTK files are POINT_DATA and CELL_DATA
    # fields. They can contain a number of SCALARS+LOOKUP_TABLE tables, without
    # giving and indication of how many there are. Hence, SCALARS must be
    # treated like a first-class section. To associate it with POINT/CELL_DATA,
    # we store the `active` section in this variable.
    active = None

    while True:
        line = f.readline().decode('utf-8')
        if not line:
            # EOF
            break

        line = line.strip()
        # pylint: disable=len-as-condition
        if len(line) == 0:
            continue

        split = line.split()
        section = split[0]

        if section == 'DATASET':
            dataset_type = split[1]
            assert dataset_type == 'UNSTRUCTURED_GRID', \
                'Only VTK UNSTRUCTURED_GRID supported.'

        elif section == 'POINTS':
            active = 'POINTS'
            num_points = int(split[1])
            data_type = split[2]
            dtype = vtk_to_numpy_dtype[data_type]
            if is_ascii:
                points = numpy.fromfile(
                    f, count=num_points*3, sep=' ',
                    dtype=dtype
                    )
            else:
                # binary
                num_bytes = numpy.dtype(dtype).itemsize
                total_num_bytes = num_points * (3 * num_bytes)
                # Binary data is big endian, see
                # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
                dtype = dtype.newbyteorder('>')
                points = \
                    numpy.fromstring(f.read(total_num_bytes), dtype=dtype)
                line = f.readline().decode('utf-8')
                assert line == '\n'

            points = points.reshape((num_points, 3))

        elif section == 'CELLS':
            active = 'CELLS'

            num_items = int(split[2])
            if is_ascii:
                c = numpy.fromfile(f, count=num_items, sep=' ', dtype=int)
            else:
                # binary
                num_bytes = 4
                total_num_bytes = num_items * num_bytes
                c = numpy.fromstring(f.read(total_num_bytes), dtype='>i4')
                line = f.readline().decode('utf-8')
                assert line == '\n'

            offsets = []
            if len(c) > 0:
                offsets.append(0)
                while offsets[-1] + c[offsets[-1]] + 1 < len(c):
                    offsets.append(offsets[-1] + c[offsets[-1]] + 1)
            offsets = numpy.array(offsets)

        elif section == 'CELL_TYPES':
            active = 'CELL_TYPES'

            num_items = int(split[1])
            if is_ascii:
                ct = \
                    numpy.fromfile(f, count=int(num_items), sep=' ', dtype=int)
            else:
                # binary
                num_bytes = 4
                total_num_bytes = num_items * num_bytes
                ct = numpy.fromstring(f.read(total_num_bytes), dtype='>i4')
                line = f.readline().decode('utf-8')
                assert line == '\n'

        elif section == 'POINT_DATA':
            active = 'POINT_DATA'
            num_items = int(split[1])

        elif section == 'CELL_DATA':
            active = 'CELL_DATA'
            num_items = int(split[1])

        elif section == 'SCALARS':
            if active == 'POINT_DATA':
                d = point_data
            else:
                assert active == 'CELL_DATA', \
                    'Illegal SCALARS in section \'{}\'.'.format(active)
                d = cell_data_raw

            d.update(_read_scalar_field(f, num_items, split))

        elif section == 'VECTORS':
            if active == 'POINT_DATA':
                d = point_data
            else:
                assert active == 'CELL_DATA', \
                    'Illegal SCALARS in section \'{}\'.'.format(active)
                d = cell_data_raw

            d.update(_read_vector_field(f, num_items, split))

        elif section == 'TENSORS':
            if active == 'POINT_DATA':
                d = point_data
            else:
                assert active == 'CELL_DATA', \
                    'Illegal SCALARS in section \'{}\'.'.format(active)
                d = cell_data_raw

            d.update(_read_tensor_field(f, num_items, split))

        else:
            assert section == 'FIELD', \
                'Unknown section \'{}\'.'.format(section)

            if active == 'POINT_DATA':
                d = point_data
            else:
                assert active == 'CELL_DATA', \
                    'Illegal FIELD in section \'{}\'.'.format(active)
                d = cell_data_raw

            d.update(_read_fields(f, int(split[2]), is_ascii))

    assert c is not None, \
        'Required section CELLS not found.'
    assert ct is not None, \
        'Required section CELL_TYPES not found.'

    cells, cell_data = translate_cells(c, offsets, ct, cell_data_raw)

    return points, cells, point_data, cell_data, field_data


def _read_scalar_field(f, num_data, split):
    data_name = split[1]
    data_type = split[2]
    try:
        num_comp = int(split[3])
    except IndexError:
        num_comp = 1

    # The standard says:
    # > The parameter numComp must range between (1,4) inclusive; [...]
    assert 0 < num_comp < 5

    dtype = vtk_to_numpy_dtype[data_type]
    lt, _ = f.readline().decode('utf-8').split()
    assert lt == 'LOOKUP_TABLE'
    data = numpy.fromfile(f, count=num_data, sep=' ', dtype=dtype)

    return {data_name: data}


def _read_vector_field(f, num_data, split):
    data_name = split[1]
    data_type = split[2]

    dtype = vtk_to_numpy_dtype[data_type]
    data = numpy.fromfile(
        f, count=3*num_data, sep=' ', dtype=dtype
        ).reshape(-1, 3)

    return {data_name: data}


def _read_tensor_field(f, num_data, split):
    data_name = split[1]
    data_type = split[2]

    dtype = vtk_to_numpy_dtype[data_type]
    data = numpy.fromfile(
        f, count=9*num_data, sep=' ', dtype=dtype
        ).reshape(-1, 3, 3)

    return {data_name: data}


def _read_fields(f, num_fields, is_ascii):
    data = {}
    for _ in range(num_fields):
        name, shape0, shape1, data_type = \
            f.readline().decode('utf-8').split()
        shape0 = int(shape0)
        shape1 = int(shape1)
        dtype = vtk_to_numpy_dtype[data_type]

        if is_ascii:
            dat = numpy.fromfile(
                f, count=shape0 * shape1, sep=' ', dtype=dtype
                )
        else:
            # binary
            num_bytes = numpy.dtype(dtype).itemsize
            total_num_bytes = shape0 * shape1 * num_bytes
            # Binary data is big endian, see
            # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
            dtype = dtype.newbyteorder('>')
            dat = numpy.fromstring(f.read(total_num_bytes), dtype=dtype)
            line = f.readline().decode('utf-8')
            assert line == '\n'

        if shape0 != 1:
            dat = dat.reshape((shape1, shape0))

        data[name] = dat

    return data


def raw_from_cell_data(cell_data):
    # merge cell data
    cell_data_raw = {}
    for d in cell_data.values():
        for name, values in d.items():
            if name in cell_data_raw:
                cell_data_raw[name].append(values)
            else:
                cell_data_raw[name] = [values]
    for name in cell_data_raw:
        cell_data_raw[name] = numpy.concatenate(cell_data_raw[name])

    return cell_data_raw


def translate_cells(data, offsets, types, cell_data_raw):
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

    cells = {}
    cell_data = {}
    for tpe, b in bins.items():
        meshio_type = vtk_to_meshio_type[tpe]
        n = data[offsets[b[0]]]
        assert (data[offsets[b]] == n).all()
        indices = numpy.array([
            numpy.arange(1, n+1) + o for o in offsets[b]
            ])
        cells[meshio_type] = data[indices]
        cell_data[meshio_type] = \
            {key: value[b] for key, value in cell_data_raw.items()}

    return cells, cell_data


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None,
          write_binary=True):
    if not write_binary:
        logging.warning('VTK ASCII files are only meant for debugging.')

    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    with open(filename, 'wb') as f:
        f.write('# vtk DataFile Version 4.2\n'.encode('utf-8'))
        f.write('written by meshio v{}\n'.format(__version__).encode('utf-8'))
        f.write(('BINARY\n' if write_binary else 'ASCII\n').encode('utf-8'))
        f.write('DATASET UNSTRUCTURED_GRID\n'.encode('utf-8'))

        # write points and cells
        _write_points(f, points, write_binary)
        _write_cells(f, cells, write_binary)

        # write point data
        if point_data:
            num_points = len(points)
            f.write('POINT_DATA {}\n'.format(num_points).encode('utf-8'))
            _write_field_data(f, point_data, write_binary)

        # write cell data
        if cell_data:
            total_num_cells = sum([len(c) for c in cells.values()])
            cell_data_raw = raw_from_cell_data(cell_data)
            f.write('CELL_DATA {}\n'.format(total_num_cells).encode('utf-8'))
            _write_field_data(f, cell_data_raw, write_binary)

    return


def _write_points(f, points, write_binary):
    f.write(
        'POINTS {} {}\n'.format(
            len(points), numpy_to_vtk_dtype[points.dtype]
            ).encode('utf-8'))

    if write_binary:
        # Binary data must be big endian, see
        # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        points.astype(points.dtype.newbyteorder('>')).tofile(f, sep='')
    else:
        # ascii
        points.tofile(f, sep=' ')
    f.write('\n'.encode('utf-8'))
    return


def _write_cells(f, cells, write_binary):
    total_num_cells = sum([len(c) for c in cells.values()])
    total_num_idx = sum([numpy.prod(c.shape) for c in cells.values()])
    # For each cell, the number of nodes is stored
    total_num_idx += total_num_cells
    f.write(
        'CELLS {} {}\n'.format(total_num_cells, total_num_idx)
        .encode('utf-8'))
    if write_binary:
        for key in cells:
            n = cells[key].shape[1]
            d = numpy.column_stack([
                numpy.full(len(cells[key]), n), cells[key]
                ]).astype(numpy.dtype('>i4'))
            f.write(d.tostring())
        if write_binary:
            f.write('\n'.encode('utf-8'))
    else:
        # ascii
        for key in cells:
            n = cells[key].shape[1]
            for cell in cells[key]:
                f.write((' '.join([
                    '{}'.format(idx)
                    for idx in numpy.concatenate([[n], cell])
                    ]) + '\n').encode('utf-8'))

    # write cell types
    f.write('CELL_TYPES {}\n'.format(total_num_cells).encode('utf-8'))
    if write_binary:
        for key in cells:
            d = numpy.full(
                len(cells[key]), meshio_to_vtk_type[key]
                ).astype(numpy.dtype('>i4'))
            f.write(d.tostring())
        f.write('\n'.encode('utf-8'))
    else:
        # ascii
        for key in cells:
            for _ in range(len(cells[key])):
                f.write(
                    '{}\n'.format(meshio_to_vtk_type[key]).encode('utf-8')
                    )
    return


def _write_field_data(f, data, write_binary):
    f.write((
        'FIELD FieldData {}\n'.format(len(data))
        ).encode('utf-8'))
    for name, values in data.items():
        if len(values.shape) == 1:
            num_tuples = values.shape[0]
            num_components = 1
        else:
            assert len(values.shape) == 2, \
                'Only one and two-dimensional field data supported.'
            num_tuples = values.shape[0]
            num_components = values.shape[1]
        f.write(('{} {} {} {}\n'.format(
            name, num_components, num_tuples,
            numpy_to_vtk_dtype[values.dtype]
            )).encode('utf-8'))
        if write_binary:
            values.astype(values.dtype.newbyteorder('>')).tofile(f, sep='')
        else:
            # ascii
            values.tofile(f, sep=' ')
            # numpy.savetxt(f, points)
        f.write('\n'.encode('utf-8'))
    return
