# -*- coding: utf-8 -*-
#
'''
I/O for VTK <https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging
import re

import numpy

vtk_to_meshio_type = {
    1: 'vertex',
    3: 'line',
    5: 'triangle',
    9: 'quad',
    10: 'tetra',
    12: 'hexahedron',
    13: 'wedge',
    14: 'pyramid',
    21: 'line3',
    22: 'triangle6',
    23: 'quad8',
    24: 'tetra10',
    25: 'hexahedron20',
    }


def read(filetype, filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # pylint: disable=import-error

    # initialize output data
    points = []
    cells = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    # skip header and title
    f.readline()
    f.readline()

    data_type = f.readline().decode('utf-8').strip()
    assert data_type in ['ASCII', 'BINARY'], \
        'Unknown VTK data type \'{}\'.'.format(data_type)

    line = f.readline().decode('utf-8').strip()
    match = re.match('DATASET +([^ ]+)', line)
    assert match is not None
    dataset_type = match.group(1)
    assert dataset_type == 'UNSTRUCTURED_GRID', \
        'Only VTK UNSTRUCTURED_GRID supported.'

    vtk_to_numpy_dtype = {
        'double': numpy.float64
        }

    c = None
    ct = None

    while True:
        line = f.readline().decode('utf-8')
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0:
            continue

        section = line.split()[0]

        if section == 'POINTS':
            _, num_points, data_type = line.split()
            num_points = int(num_points)
            points = numpy.fromfile(
                f, count=num_points*3, sep=' ',
                dtype=vtk_to_numpy_dtype[data_type]
                ).reshape((num_points, 3))

        elif section == 'CELLS':
            _, num_cells, num_items = line.split()
            num_cells = int(num_cells)
            num_items = int(num_items)
            c = numpy.fromfile(
                f, count=num_items, sep=' ', dtype=int
                )

        else:
            assert section == 'CELL_TYPES', \
                'Unknown section \'{}\'.'.format(section)

            _, num_ints = line.split()
            num_items = int(num_items)

            ct = numpy.fromfile(
                f, count=num_items, sep=' ', dtype=int
                )

    assert c is not None
    assert ct is not None

    print(c)
    print(ct)

    exit(1)

    return points, cells, point_data, cell_data, field_data


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
    meshio_to_vtk_type = {
        'vertex': vtk.VTK_VERTEX,
        'line': vtk.VTK_LINE,
        'line3': vtk.VTK_QUADRATIC_EDGE,
        'triangle': vtk.VTK_TRIANGLE,
        'triangle6': vtk.VTK_QUADRATIC_TRIANGLE,
        'quad': vtk.VTK_QUAD,
        'quad8': vtk.VTK_QUADRATIC_QUAD,
        'tetra': vtk.VTK_TETRA,
        'tetra10': vtk.VTK_QUADRATIC_TETRA,
        'hexahedron': vtk.VTK_HEXAHEDRON,
        'hexahedron20': vtk.VTK_QUADRATIC_HEXAHEDRON,
        'wedge': vtk.VTK_WEDGE,
        'pyramid': vtk.VTK_PYRAMID
        }

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
