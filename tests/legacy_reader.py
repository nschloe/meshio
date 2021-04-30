import numpy as np

from meshio import Mesh
from meshio.vtk_io import vtk_to_meshio_type


def read(filetype, filename):
    import vtk
    from vtk.util import numpy as numpy_support

    def _read_data(data):
        """Extract numpy arrays from a VTK data set."""
        # Go through all arrays, fetch data.
        out = {}
        for k in range(data.GetNumberOfArrays()):
            array = data.GetArray(k)
            if array:
                array_name = array.GetName()
                out[array_name] = np.copy(vtk.util.numpy_support.vtk_to_numpy(array))
        return out

    def _read_cells(vtk_mesh):
        data = np.copy(
            vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetCells().GetData())
        )
        offsets = np.copy(
            vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetCellLocationsArray())
        )
        types = np.copy(
            vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetCellTypesArray())
        )

        # `data` is a one-dimensional vector with
        # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
        # Translate it into the cells dictionary.
        cells = {}
        for vtk_type, meshio_type in vtk_to_meshio_type.items():
            # Get all offsets for vtk_type
            os = offsets[np.argwhere(types == vtk_type).transpose()[0]]
            num_cells = len(os)
            if num_cells > 0:
                if meshio_type == "polygon":
                    for idx_cell in range(num_cells):
                        num_pts = data[os[idx_cell]]
                        cell = data[os[idx_cell] + 1 : os[idx_cell] + 1 + num_pts]
                        key = meshio_type + str(num_pts)
                        if key in cells:
                            cells[key] = np.vstack([cells[key], cell])
                        else:
                            cells[key] = cell
                else:
                    num_pts = data[os[0]]
                    # instantiate the array
                    arr = np.empty((num_cells, num_pts), dtype=int)
                    # store the num_pts entries after the offsets into the columns
                    # of arr
                    for k in range(num_pts):
                        arr[:, k] = data[os + k + 1]
                    cells[meshio_type] = arr

        return cells

    if filetype in ["vtk", "vtk-ascii", "vtk-binary"]:
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.SetReadAllNormals(1)
        reader.SetReadAllScalars(1)
        reader.SetReadAllTensors(1)
        reader.SetReadAllVectors(1)
        reader.Update()
        vtk_mesh = reader.GetOutput()
    elif filetype in ["vtu", "vtu-ascii", "vtu-binary"]:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_mesh = reader.GetOutput()
    elif filetype in ["xdmf", "xdmf2"]:
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(filename)
        reader.SetReadAllColorScalars(1)
        reader.SetReadAllFields(1)
        reader.SetReadAllNormals(1)
        reader.SetReadAllScalars(1)
        reader.SetReadAllTCoords(1)
        reader.SetReadAllTensors(1)
        reader.SetReadAllVectors(1)
        reader.Update()
        vtk_mesh = reader.GetOutputDataObject(0)
    elif filetype == "xdmf3":
        reader = vtk.vtkXdmf3Reader()
        reader.SetFileName(filename)
        reader.SetReadAllColorScalars(1)
        reader.SetReadAllFields(1)
        reader.SetReadAllNormals(1)
        reader.SetReadAllScalars(1)
        reader.SetReadAllTCoords(1)
        reader.SetReadAllTensors(1)
        reader.SetReadAllVectors(1)
        reader.Update()
        vtk_mesh = reader.GetOutputDataObject(0)
    else:
        assert filetype == "exodus", f"Unknown file type '{filename}'."
        reader = vtk.vtkExodusIIReader()
        reader.SetFileName(filename)
        vtk_mesh = _read_exodusii_mesh(reader)

    # Explicitly extract points, cells, point data, field data
    points = np.copy(numpy_support.vtk_to_numpy(vtk_mesh.GetPoints().GetData()))
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
            cd[cell_type][name] = array[index : index + num_cells]
        index += num_cells
    cell_data = cd

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def _read_exodusii_mesh(reader, timestep=None):
    """Uses a vtkExodusIIReader to return a vtkUnstructuredGrid."""
    # Fetch metadata.
    reader.UpdateInformation()

    # Set time step to read.
    if timestep:
        reader.SetTimeStep(timestep)

    # Make sure the point data are read during Update().
    for k in range(reader.GetNumberOfPointResultArrays()):
        arr_name = reader.GetPointResultArrayName(k)
        reader.SetPointResultArrayStatus(arr_name, 1)

    # Make sure the cell data are read during Update().
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
            if sub_block is not None and sub_block.IsA("vtkUnstructuredGrid"):
                vtk_mesh.append(sub_block)

    assert vtk_mesh, "No 'vtkUnstructuredGrid' found!"
    assert len(vtk_mesh) == 1, "More than one 'vtkUnstructuredGrid' found!"

    # Cut off trailing '_' from array names.
    for k in range(vtk_mesh[0].GetPointData().GetNumberOfArrays()):
        array = vtk_mesh[0].GetPointData().GetArray(k)
        array_name = array.GetName()
        if array_name[-1] == "_":
            array.SetName(array_name[0:-1])

    # time_values = reader.GetOutputInformation(0).Get(
    #     vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    #     )

    return vtk_mesh[0]  # , time_values
