# -*- coding: utf-8 -*-
#
'''
I/O for VTK, VTU, Exodus etc.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import logging
import numpy

# Make explicit copies of the data; some (all?) of it is quite volatile and
# contains garbage once the vtk_mesh goes out of scopy.


def read(filetype, filename):
    # pylint: disable=import-error
    import vtk
    from vtk.util import numpy_support

    def _read_data(data):
        '''Extract numpy arrays from a VTK data set.
        '''
        # Go through all arrays, fetch data.
        out = {}
        for k in range(data.GetNumberOfArrays()):
            array = data.GetArray(k)
            if array:
                array_name = array.GetName()
                out[array_name] = numpy.copy(
                    vtk.util.numpy_support.vtk_to_numpy(array)
                    )
        return out

    def _read_cells(vtk_mesh):
        data = numpy.copy(vtk.util.numpy_support.vtk_to_numpy(
                vtk_mesh.GetCells().GetData()
                ))
        offsets = numpy.copy(vtk.util.numpy_support.vtk_to_numpy(
                vtk_mesh.GetCellLocationsArray()
                ))
        types = numpy.copy(vtk.util.numpy_support.vtk_to_numpy(
                vtk_mesh.GetCellTypesArray()
                ))

        vtk_to_meshio_type = {
            vtk.VTK_VERTEX: 'vertex',
            vtk.VTK_LINE: 'line',
            vtk.VTK_TRIANGLE: 'triangle',
            vtk.VTK_QUAD: 'quad',
            vtk.VTK_TETRA: 'tetra',
            vtk.VTK_HEXAHEDRON: 'hexahedron',
            vtk.VTK_WEDGE: 'wedge',
            vtk.VTK_PYRAMID: 'pyramid'
            }

        # `data` is a one-dimensional vector with
        # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
        # Translate it into the cells dictionary.
        cells = {}
        for vtk_type, meshio_type in vtk_to_meshio_type.items():
            # Get all offsets for vtk_type
            os = offsets[numpy.argwhere(types == vtk_type).transpose()[0]]
            num_cells = len(os)
            if num_cells > 0:
                num_pts = data[os[0]]
                # instantiate the array
                arr = numpy.empty((num_cells, num_pts), dtype=int)
                # sort the num_pts entries after the offsets into the columns
                # of arr
                for k in range(num_pts):
                    arr[:, k] = data[os+k+1]
                cells[meshio_type] = arr

        return cells

    if filetype in ['vtk', 'vtk-ascii', 'vtk-binary']:
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_mesh = reader.GetOutput()
    elif filetype in ['vtu', 'vtu-ascii', 'vtu-binary']:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_mesh = reader.GetOutput()
    elif filetype in ['xdmf', 'xdmf2']:
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_mesh = reader.GetOutputDataObject(0)
    elif filetype == 'xdmf3':
        reader = vtk.vtkXdmf3Reader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_mesh = reader.GetOutputDataObject(0)
    else:
        assert filetype == 'exodus', \
            'Unknown file type \'{}\'.'.format(filename)
        reader = vtk.vtkExodusIIReader()
        reader.SetFileName(filename)
        vtk_mesh = _read_exodusii_mesh(reader)

    # Explicitly extract points, cells, point data, field data
    points = numpy.copy(numpy_support.vtk_to_numpy(
            vtk_mesh.GetPoints().GetData()
            ))
    cells = _read_cells(vtk_mesh)

    point_data = _read_data(vtk_mesh.GetPointData())
    field_data = _read_data(vtk_mesh.GetFieldData())

    cell_data = _read_data(vtk_mesh.GetCellData())
    # split cell_data by the cell type
    cd = {}
    index = 0
    for cell_type in cells:
        num_cells = len(cells[cell_type])
        cd[cell_type] = {}
        for name, array in cell_data.items():
            cd[cell_type][name] = array[index:index+num_cells]
        index += num_cells
    cell_data = cd

    return points, cells, point_data, cell_data, field_data


# def _read_exodus_mesh(reader, file_name):
#     '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
#     '''
#     reader.SetFileName(file_name)
#
#     # Create Exodus metadata that can be used later when writing the file.
#     reader.ExodusModelMetadataOn()
#
#     # Fetch metadata.
#     reader.UpdateInformation()
#
#     # Make sure the point fields are read during Update().
#     for k in range(reader.GetNumberOfPointArrays()):
#         arr_name = reader.GetPointArrayName(k)
#         reader.SetPointArrayStatus(arr_name, 1)
#
#     # Read the file.
#     reader.Update()
#
#     return reader.GetOutput()


def _read_exodusii_mesh(reader, timestep=None):
    '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
    '''
    # Fetch metadata.
    reader.UpdateInformation()

    # Set time step to read.
    if timestep:
        reader.SetTimeStep(timestep)

    # Make sure the point fields are read during Update().
    for k in range(reader.GetNumberOfPointResultArrays()):
        arr_name = reader.GetPointResultArrayName(k)
        reader.SetPointResultArrayStatus(arr_name, 1)

    # Make sure the point fields are read during Update().
    for k in range(reader.GetNumberOfElementResultArrays()):
        arr_name = reader.GetElementResultArrayName(k)
        reader.SetElementResultArrayStatus(arr_name, 1)

    # Make sure all field data is read.
    for k in range(reader.GetNumberOfGlobalResultArrays()):
        arr_name = reader.GetGlobalResultArrayName(k)
        reader.SetGlobalResultArrayStatus(arr_name, 1)

    # Read the file.
    reader.Update()
    out = reader.GetOutput()

    # Loop through the blocks and search for a vtkUnstructuredGrid.
    # In Exodus, different element types are stored different meshes, with
    # point information possibly duplicated.
    vtk_mesh = []
    for i in range(out.GetNumberOfBlocks()):
        blk = out.GetBlock(i)
        for j in range(blk.GetNumberOfBlocks()):
            sub_block = blk.GetBlock(j)
            if sub_block.IsA('vtkUnstructuredGrid'):
                vtk_mesh.append(sub_block)

    assert vtk_mesh, 'No \'vtkUnstructuredGrid\' found!'
    assert len(vtk_mesh) == 1, 'More than one \'vtkUnstructuredGrid\' found!'

    # Cut off trailing '_' from array names.
    for k in range(vtk_mesh[0].GetPointData().GetNumberOfArrays()):
        array = vtk_mesh[0].GetPointData().GetArray(k)
        array_name = array.GetName()
        if array_name[-1] == '_':
            array.SetName(array_name[0:-1])

    # time_values = reader.GetOutputInformation(0).Get(
    #     vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    #     )

    return vtk_mesh[0]  # , time_values


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
        'triangle': vtk.VTK_TRIANGLE,
        'quad': vtk.VTK_QUAD,
        'tetra': vtk.VTK_TETRA,
        'hexahedron': vtk.VTK_HEXAHEDRON,
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
