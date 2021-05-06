import logging

import numpy as np

# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
vtk_to_meshio_type = {
    0: "empty",
    1: "vertex",
    # 2: 'poly_vertex',
    3: "line",
    # 4: 'poly_line',
    5: "triangle",
    # 6: 'triangle_strip',
    7: "polygon",
    # 8: 'pixel',
    9: "quad",
    10: "tetra",
    # 11: 'voxel',
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
    15: "penta_prism",
    16: "hexa_prism",
    21: "line3",
    22: "triangle6",
    23: "quad8",
    24: "tetra10",
    25: "hexahedron20",
    26: "wedge15",
    27: "pyramid13",
    28: "quad9",
    29: "hexahedron27",
    30: "quad6",
    31: "wedge12",
    32: "wedge18",
    33: "hexahedron24",
    34: "triangle7",
    35: "line4",
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


def _get_writer(filetype, filename):
    import vtk

    if filetype in "vtk-ascii":
        logging.warning("VTK ASCII files are only meant for debugging.")
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileTypeToASCII()
    elif filetype == "vtk-binary":
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileTypeToBinary()
    elif filetype == "vtu-ascii":
        logging.warning("VTU ASCII files are only meant for debugging.")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetDataModeToAscii()
    elif filetype == "vtu-binary":
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetDataModeToBinary()
    elif filetype == "xdmf2":
        writer = vtk.vtkXdmfWriter()
    elif filetype == "xdmf3":
        writer = vtk.vtkXdmf3Writer()
    else:
        assert filetype == "exodus", f"Unknown file type '{filename}'."
        writer = vtk.vtkExodusIIWriter()
        # if the mesh contains vtkmodeldata information, make use of it
        # and write out all time steps.
        writer.WriteAllTimeStepsOn()

    return writer


def write(filetype, filename, mesh):
    import vtk

    def _create_vtkarray(X, name):
        array = vtk.util.numpy_support.numpy_to_vtk(X, deep=1)
        array.SetName(name)
        return array

    # assert data integrity
    for val in mesh.point_data.values():
        assert len(val) == len(mesh.points), "Point data mismatch."

    for key, value in mesh.cell_data.items():
        assert key in mesh.cells, "Cell data without cell"
        for key2 in value:
            assert len(value[key2]) == len(mesh.cells[key]), "Cell data mismatch."

    vtk_mesh = _generate_vtk_mesh(mesh.points, mesh.cells)

    # add point data
    pd = vtk_mesh.GetPointData()
    for name, X in mesh.point_data.items():
        # There is a naming inconsistency in VTK when it comes to multivectors
        # in Exodus files:
        # If a vector 'v' has two components, they are called 'v_x', 'v_y'
        # (note the underscore), if it has three, then they are called 'vx',
        # 'vy', 'vz'. See bug <https://vtk.org/Bug/view.php?id=15894>.
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
    for cell_type in mesh.cell_data:
        all_keys += mesh.cell_data[cell_type].keys()
    # create unified cell data
    for key in all_keys:
        for cell_type in mesh.cell_data:
            assert key in mesh.cell_data[cell_type]
    unified_cell_data = {
        key: np.concatenate([value[key] for value in mesh.cell_data.values()])
        for key in all_keys
    }
    # add the array data to the mesh
    cd = vtk_mesh.GetCellData()
    for name, array in unified_cell_data.items():
        cd.AddArray(_create_vtkarray(array, name))

    # add field data
    fd = vtk_mesh.GetFieldData()
    for key, value in mesh.field_data.items():
        fd.AddArray(_create_vtkarray(value, key))

    writer = _get_writer(filetype, filename)

    writer.SetFileName(filename)
    try:
        writer.SetInput(vtk_mesh)
    except AttributeError:
        writer.SetInputData(vtk_mesh)
    writer.Write()

    return


def _generate_vtk_mesh(points, cells):
    import vtk
    from vtk.util import numpy as numpy_support

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
        cell_types.append(np.empty(numcells, dtype=np.ubyte))
        cell_types[-1].fill(vtk_type)
        # add cell offsets
        cell_offsets.append(
            np.arange(
                len_array,
                len_array + numcells * (num_local_nodes + 1),
                num_local_nodes + 1,
                dtype=np.int64,
            )
        )
        cell_connectivity.append(
            np.c_[num_local_nodes * np.ones(numcells, dtype=data.dtype), data].flatten()
        )
        len_array += len(cell_connectivity[-1])

    cell_types = np.concatenate(cell_types)
    cell_offsets = np.concatenate(cell_offsets)
    cell_connectivity = np.concatenate(cell_connectivity)

    connectivity = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(
        cell_connectivity.astype(np.int64), deep=1
    )

    # wrap the data into a vtkCellArray
    cell_array = vtk.vtkCellArray()
    cell_array.SetCells(len(cell_types), connectivity)

    # Add cell data to the mesh
    mesh.SetCells(
        numpy_support.numpy_to_vtk(
            cell_types, deep=1, array_type=vtk.vtkUnsignedCharArray().GetDataType()
        ),
        numpy_support.numpy_to_vtk(
            cell_offsets, deep=1, array_type=vtk.vtkIdTypeArray().GetDataType()
        ),
        cell_array,
    )

    return mesh
