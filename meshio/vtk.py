def read_vtk_mesh(reader, file_name):
    '''Uses a vtkReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


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
    vtk_mesh = []
    for i in range(out.GetNumberOfBlocks()):
        blk = out.GetBlock(i)
        for j in range(blk.GetNumberOfBlocks()):
            sub_block = blk.GetBlock(j)
            if sub_block.IsA('vtkUnstructuredGrid'):
                vtk_mesh.append(sub_block)

    if len(vtk_mesh) == 0:
        raise IOError('No \'vtkUnstructuredGrid\' found!')
    elif len(vtk_mesh) > 1:
        raise IOError('More than one \'vtkUnstructuredGrid\' found!')

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


def _read_cells_nodes(vtk_mesh):

    num_cells = vtk_mesh.GetNumberOfCells()
    array = vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetCells().GetData())
    # array is a one-dimensional vector with
    # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
    num_nodes_per_cell = array[0]
    assert all(array[::num_nodes_per_cell+1] == num_nodes_per_cell)
    cells = array.reshape(num_cells, num_nodes_per_cell+1)

    # remove first column; it only lists the number of points
    return numpy.delete(cells, 0, 1)


def _read_data(data):
    '''Extract numpy arrays from a VTK data set.
    '''
    # Go through all arrays, fetch data.
    out = {}
    for k in range(data.GetNumberOfArrays()):
        array = data.GetArray(k)
        array_name = array.GetName()
        out[array_name] = vtk.util.numpy_support.vtk_to_numpy(array)

    return out


def _generate_vtk_mesh(points, cellsNodes):
    mesh = vtk.vtkUnstructuredGrid()

    # set points
    vtk_points = vtk.vtkPoints()
    # Not using a deep copy here results in a segfault.
    vtk_array = numpy_support.numpy_to_vtk(points, deep=True)
    vtk_points.SetData(vtk_array)
    mesh.SetPoints(vtk_points)

    # Set cells.
    # create cell_array. It's a one-dimensional vector with
    # (num_points2, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
    numcells, num_local_nodes = cellsNodes.shape
    cc = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(
        numpy.c_[
            num_local_nodes * numpy.ones(numcells, dtype=cellsNodes.dtype),
            cellsNodes
            ].flatten(),
        deep=1
        )
    # wrap the data into a vtkCellArray
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(numcells, cc)

    numnodes_to_type = {
        2: vtk.VTK_LINE,
        3: vtk.VTK_TRIANGLE,
        4: vtk.VTK_TETRA
        }
    mesh.SetCells(
        numnodes_to_type[num_local_nodes],
        cell_array
        )

    return mesh


def _create_vtkarray(X, name):
    array = vtk.util.numpy_support.numpy_to_vtk(X, deep=1)
    array.SetName(name)
    return array
