# -*- coding: utf-8 -*-
#
'''
I/O for VTK <https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging

import numpy

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


def read(filetype, filename, is_little_endian=True):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        out = read_buffer(f, is_little_endian=is_little_endian)
    return out


def read_buffer(f, is_little_endian=True):
    # pylint: disable=import-error

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

    endian = '>' if is_little_endian else '<'

    vtk_to_numpy_dtype = {
        'bit': (numpy.bool, '?4', 4),
        # 'unsigned_char':
        # 'char':
        # 'unsigned_short':
        # 'short':
        'unsigned_int': (numpy.uint32, endian + 'u4', 4),
        'int': (numpy.int32, endian + 'i4', 4),
        'unsigned_long': (numpy.int64, endian + 'u8', 8),
        'long': (numpy.int64, endian + 'i8', 8),
        'vtktypeint64': (numpy.int64, endian + 'i8', 8),
        'float': (numpy.float32, endian + 'f4', 4),
        'double': (numpy.float64, endian + 'f8', 8),
        }

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
            dtype, bdtype, num_bytes = vtk_to_numpy_dtype[data_type]
            if is_ascii:
                points = numpy.fromfile(
                    f, count=num_points*3, sep=' ',
                    dtype=dtype
                    )
            else:
                # binary
                total_num_bytes = num_points * (3 * num_bytes)
                points = \
                    numpy.fromstring(f.read(total_num_bytes), dtype=bdtype)
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
                # cell data is little endian. okay.
                c = numpy.fromstring(
                    f.read(total_num_bytes), dtype=endian+'i4'
                    )
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
                ct = numpy.fromstring(
                    f.read(total_num_bytes), dtype=endian+'i4'
                    )
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

            d.update(
                _read_scalar_field(f, num_items, split, vtk_to_numpy_dtype)
                )

        elif section == 'VECTORS':
            if active == 'POINT_DATA':
                d = point_data
            else:
                assert active == 'CELL_DATA', \
                    'Illegal SCALARS in section \'{}\'.'.format(active)
                d = cell_data_raw

            d.update(
                _read_vector_field(f, num_items, split, vtk_to_numpy_dtype)
                )

        else:
            assert section == 'FIELD', \
                'Unknown section \'{}\'.'.format(section)

            if active == 'POINT_DATA':
                d = point_data
            else:
                assert active == 'CELL_DATA', \
                    'Illegal FIELD in section \'{}\'.'.format(active)
                d = cell_data_raw

            d.update(
                _read_fields(
                    f, int(split[2]), vtk_to_numpy_dtype, is_ascii
                    ))

    assert c is not None
    assert ct is not None

    cells = translate_cells(c, offsets, ct)

    cell_data = cell_data_from_raw(cells, cell_data_raw)

    return points, cells, point_data, cell_data, field_data


def _read_scalar_field(f, num_data, split, vtk_to_numpy_dtype):
    data_name = split[1]
    data_type = split[2]
    try:
        num_comp = int(split[3])
    except IndexError:
        num_comp = 1

    # The standard says:
    # > The parameter numComp must range between (1,4) inclusive; [...]
    assert 0 < num_comp < 5

    dtype, _, _ = vtk_to_numpy_dtype[data_type]
    lt, name = f.readline().decode('utf-8').split()
    assert lt == 'LOOKUP_TABLE'
    data = numpy.fromfile(f, count=num_data, sep=' ', dtype=dtype)

    return {data_name: data}


def _read_vector_field(f, num_data, split, vtk_to_numpy_dtype):
    data_name = split[1]
    data_type = split[2]

    dtype, _, _ = vtk_to_numpy_dtype[data_type]
    data = numpy.fromfile(
        f, count=3*num_data, sep=' ', dtype=dtype
        ).reshape(-1, 3)

    return {data_name: data}


def _read_fields(f, num_fields, vtk_to_numpy_dtype, is_ascii):
    data = {}
    for _ in range(num_fields):
        name, shape0, shape1, data_type = \
            f.readline().decode('utf-8').split()
        shape0 = int(shape0)
        shape1 = int(shape1)
        dtype, bdtype, num_bytes = vtk_to_numpy_dtype[data_type]

        if is_ascii:
            dat = numpy.fromfile(
                f, count=shape0 * shape1, sep=' ', dtype=dtype
                )
        else:
            # binary
            total_num_bytes = shape0 * shape1 * num_bytes
            dat = numpy.fromstring(f.read(total_num_bytes), dtype=bdtype)
            line = f.readline().decode('utf-8')
            assert line == '\n'

        if shape0 != 1:
            dat = dat.reshape((shape1, shape0))

        data[name] = dat

    return data


def cell_data_from_raw(cells, cell_data_raw):
    cell_data = {k: {} for k in cells}
    num_all_cells = sum([len(c) for c in cells.values()])
    for key in cell_data_raw:
        d = cell_data_raw[key]
        if len(d) != num_all_cells:
            d = d.reshape(num_all_cells, -1)

        r = 0
        for k in cells:
            cell_data[k][key] = d[r:r+len(cells[k])]
            r += len(cells[k])

    return cell_data


def translate_cells(data, offsets, types):
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

    cells = {}
    for tpe, b in bins.items():
        meshio_type = vtk_to_meshio_type[tpe]
        n = data[offsets[b[0]]]
        assert (data[offsets[b]] == n).all()
        indices = numpy.array([
            numpy.arange(1, n+1) + o for o in offsets[b]
            ])
        cells[meshio_type] = data[indices]

    return cells


def write(filetype,
          filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    # pylint: disable=import-error
    import vtk

    def _create_vtkarray(X, name):
        array = vtk.util.numpy_support.numpy_to_vtk(X, deep=1)
        array.SetName(name)
        return array

    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    # assert data integrity
    for key in point_data:
        assert len(point_data[key]) == len(points), \
                'Point data mismatch.'
    for key in cell_data:
        assert key in cells, 'Cell data without cell'
        for key2 in cell_data[key]:
            assert len(cell_data[key][key2]) == len(cells[key]), \
                    'Cell data mismatch.'

    vtk_mesh = _generate_vtk_mesh(points, cells)

    # add point data
    pd = vtk_mesh.GetPointData()
    for name, X in point_data.items():
        # There is a naming inconsistency in VTK when it comes to multivectors
        # in Exodus files:
        # If a vector 'v' has two components, they are called 'v_x', 'v_y'
        # (note the underscore), if it has three, then they are called 'vx',
        # 'vy', 'vz'. See bug <http://www.vtk.org/Bug/view.php?id=15894>.
        # For VT{K,U} files, no underscore is ever added.
        pd.AddArray(_create_vtkarray(X, name))

    # Add cell data.
    # The cell_data is structured like
    #
    #  cell_type ->
    #      key -> array
    #      key -> array
    #      [...]
    #  cell_type ->
    #      key -> array
    #      key -> array
    #      [...]
    #  [...]
    #
    # VTK expects one array for each `key`, so assemble the keys across all
    # mesh_types. This requires each key to be present for each mesh_type, of
    # course.
    all_keys = []
    for cell_type in cell_data:
        all_keys += cell_data[cell_type].keys()
    # create unified cell data
    for key in all_keys:
        for cell_type in cell_data:
            assert key in cell_data[cell_type]
    unified_cell_data = {
        key: numpy.concatenate([
            cell_data[cell_type][key]
            for cell_type in cell_data
            ])
        for key in all_keys
        }
    # add the array data to the mesh
    cd = vtk_mesh.GetCellData()
    for name, array in unified_cell_data.items():
        cd.AddArray(_create_vtkarray(array, name))

    # add field data
    fd = vtk_mesh.GetFieldData()
    for key, value in field_data.items():
        fd.AddArray(_create_vtkarray(value, key))

    if filetype in 'vtk-ascii':
        logging.warning('ASCII files are only meant for debugging.')
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileTypeToASCII()
    elif filetype == 'vtk-binary':
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileTypeToBinary()
    elif filetype == 'vtu-ascii':
        logging.warning('ASCII files are only meant for debugging.')
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetDataModeToAscii()
    elif filetype == 'vtu-binary':
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetDataModeToBinary()
    elif filetype == 'xdmf':
        writer = vtk.vtkXdmfWriter()
    elif filetype == 'xdmf3':
        writer = vtk.vtkXdmf3Writer()
    else:
        assert filetype == 'exodus', \
            'Unknown file type \'{}\'.'.format(filename)
        writer = vtk.vtkExodusIIWriter()
        # if the mesh contains vtkmodeldata information, make use of it
        # and write out all time steps.
        writer.WriteAllTimeStepsOn()

    writer.SetFileName(filename)
    try:
        writer.SetInput(vtk_mesh)
    except AttributeError:
        writer.SetInputData(vtk_mesh)
    writer.Write()

    return


def _generate_vtk_mesh(points, cells):
    # pylint: disable=import-error
    import vtk
    from vtk.util import numpy_support

    mesh = vtk.vtkUnstructuredGrid()

    # set points
    vtk_points = vtk.vtkPoints()
    # Not using a deep copy here results in a segfault.
    vtk_array = numpy_support.numpy_to_vtk(points, deep=True)
    vtk_points.SetData(vtk_array)
    mesh.SetPoints(vtk_points)

    # Set cells.
    meshio_to_vtk_type = {y: x for x, y in vtk_to_meshio_type.items()}

    # create cell_array. It's a one-dimensional vector with
    # (num_points2, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
    cell_types = []
    cell_offsets = []
    cell_connectivity = []
    len_array = 0
    for meshio_type, data in cells.items():
        numcells, num_local_nodes = data.shape
        vtk_type = meshio_to_vtk_type[meshio_type]
        # add cell types
        cell_types.append(numpy.empty(numcells, dtype=numpy.ubyte))
        cell_types[-1].fill(vtk_type)
        # add cell offsets
        cell_offsets.append(numpy.arange(
            len_array,
            len_array + numcells * (num_local_nodes + 1),
            num_local_nodes + 1,
            dtype=numpy.int64
            ))
        cell_connectivity.append(
            numpy.c_[
                num_local_nodes * numpy.ones(numcells, dtype=data.dtype),
                data
            ].flatten()
            )
        len_array += len(cell_connectivity[-1])

    cell_types = numpy.concatenate(cell_types)
    cell_offsets = numpy.concatenate(cell_offsets)
    cell_connectivity = numpy.concatenate(cell_connectivity)

    connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(
        cell_connectivity.astype(numpy.int64),
        deep=1
        )

    # wrap the data into a vtkCellArray
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(cell_types), connectivity)

    # Add cell data to the mesh
    mesh.SetCells(
        numpy_support.numpy_to_vtk(
            cell_types,
            deep=1,
            array_type=vtk.vtkUnsignedCharArray().GetDataType()
            ),
        numpy_support.numpy_to_vtk(
            cell_offsets,
            deep=1,
            array_type=vtk.vtkIdTypeArray().GetDataType()
            ),
        cell_array
        )

    return mesh
