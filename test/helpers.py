import copy
import os
import string
import tempfile

import numpy

import meshio

# In general:
# Use values with an infinite decimal representation to test precision.

tri_mesh_2d = meshio.Mesh(
    numpy.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) / 3,
    {"triangle": numpy.array([[0, 1, 2], [0, 2, 3]])},
)

tri_mesh = meshio.Mesh(
    numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    / 3,
    {"triangle": numpy.array([[0, 1, 2], [0, 2, 3]])},
)

triangle6_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.25, 0.0],
            [1.25, 0.5, 0.0],
            [0.25, 0.75, 0.0],
            [2.0, 1.0, 0.0],
            [1.5, 1.25, 0.0],
            [1.75, 0.25, 0.0],
        ]
    )
    / 3.0,
    {"triangle6": numpy.array([[0, 1, 2, 3, 4, 5], [1, 6, 2, 8, 7, 4]])},
)

quad_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    / 3.0,
    {"quad": numpy.array([[0, 1, 4, 5], [1, 2, 3, 4]])},
)

d = 0.1
quad8_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, d, 0.0],
            [1 - d, 0.5, 0.0],
            [0.5, 1 - d, 0.0],
            [d, 0.5, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.5, -d, 0.0],
            [2 + d, 0.5, 0.0],
            [1.5, 1 + d, 0.0],
        ]
    )
    / 3.0,
    {"quad8": numpy.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 10, 11, 12, 5]])},
)

tri_quad_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    / 3.0,
    {
        "triangle": numpy.array([[0, 1, 4], [0, 4, 5]]),
        "quad": numpy.array([[1, 2, 3, 4]]),
    },
)

tet_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    / 3.0,
    {"tetra": numpy.array([[0, 1, 2, 4], [0, 2, 3, 4]])},
)

tet10_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            #
            [0.5, 0.0, 0.1],
            [1.0, 0.5, 0.1],
            [0.5, 0.5, 0.1],
            [0.25, 0.3, 0.25],
            [0.8, 0.25, 0.25],
            [0.7, 0.7, 0.3],
        ]
    )
    / 3.0,
    {"tetra10": numpy.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])},
)

hex_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    ),
    {"hexahedron": numpy.array([[0, 1, 2, 3, 4, 5, 6, 7]])},
)

hex20_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            #
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            #
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            #
            [0.5, 0.0, 1.0],
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 1.0],
            [0.0, 0.5, 1.0],
        ]
    ),
    {"hexahedron20": numpy.array([range(20)])},
)

polygon_mesh = meshio.Mesh(
    numpy.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.7, 0.5, 0.0],
            [1.5, 1.2, 0.0],
            [-0.1, 1.1, 0.0],
            [-0.5, 1.4, 0.0],
            [-0.7, 0.8, 0.0],
            [-0.3, -0.1, 0.0],
        ]
    ),
    {
        "triangle": numpy.array([[0, 1, 2], [4, 5, 6]]),
        "quad": numpy.array([[0, 1, 2, 3]]),
        "polygon5": numpy.array([[1, 4, 5, 6, 2]]),
        "polygon6": numpy.array([[0, 3, 7, 8, 9, 10], [1, 3, 7, 8, 9, 10]]),
    },
)


def add_point_data(mesh, dim, num_tags=2, seed=0, dtype=numpy.float):
    numpy.random.seed(seed)
    mesh2 = copy.deepcopy(mesh)

    shape = (len(mesh.points),) if dim == 1 else (len(mesh.points), dim)
    data = [(100 * numpy.random.rand(*shape)).astype(dtype) for _ in range(num_tags)]

    mesh2.point_data = {string.ascii_lowercase[k]: d for k, d in enumerate(data)}
    return mesh2


def add_cell_data(mesh, dim, num_tags=2, dtype=numpy.float):
    mesh2 = copy.deepcopy(mesh)
    numpy.random.seed(0)
    cell_data = {}
    for cell_type in mesh.cells:
        num_cells = len(mesh.cells[cell_type])
        if dim == 1:
            cell_data[cell_type] = {
                string.ascii_lowercase[k]: numpy.random.rand(num_cells).astype(dtype)
                for k in range(num_tags)
            }
        else:
            cell_data[cell_type] = {
                string.ascii_lowercase[k]: numpy.random.rand(num_cells, dim).astype(
                    dtype
                )
                for k in range(num_tags)
            }

    mesh2.cell_data = cell_data
    return mesh2


def add_field_data(mesh, value, dtype):
    mesh2 = copy.deepcopy(mesh)
    mesh2.field_data = {"a": numpy.array(value, dtype=dtype)}
    return mesh2


def add_node_sets(mesh):
    mesh2 = copy.deepcopy(mesh)
    mesh2.node_sets = {"fixed": numpy.array([1, 2])}
    return mesh2


def write_read(writer, reader, input_mesh, atol, extension=".dat"):
    """Write and read a file, and make sure the data is the same as before.
    """
    in_mesh = copy.deepcopy(input_mesh)

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test" + extension)
        writer(filepath, input_mesh)
        mesh = reader(filepath)

    # Make sure the output is writeable
    assert mesh.points.flags["WRITEABLE"]
    for cell_type, data in input_mesh.cells.items():
        assert mesh.cells[cell_type].flags["WRITEABLE"]

    # assert that the input mesh hasn't changed at all
    assert numpy.allclose(in_mesh.points, input_mesh.points, atol=atol, rtol=0.0)

    # Numpy's array_equal is too strict here, cf.
    # <https://mail.scipy.org/pipermail/numpy-discussion/2015-December/074410.html>.
    # Use allclose.
    n = in_mesh.points.shape[1]
    assert numpy.allclose(in_mesh.points, mesh.points[:, :n], atol=atol, rtol=0.0)

    for cell_type, data in input_mesh.cells.items():
        assert numpy.allclose(data, mesh.cells[cell_type])

    for key in input_mesh.point_data.keys():
        assert numpy.allclose(
            input_mesh.point_data[key], mesh.point_data[key], atol=atol, rtol=0.0
        )

    for cell_type, cell_type_data in input_mesh.cell_data.items():
        for key, data in cell_type_data.items():
            assert numpy.allclose(
                data, mesh.cell_data[cell_type][key], atol=atol, rtol=0.0
            )

    for name, data in input_mesh.field_data.items():
        assert numpy.allclose(data, mesh.field_data[name], atol=atol, rtol=0.0)

    return


def generic_io(filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, filename)
        meshio.write_points_cells(filepath, tri_mesh.points, tri_mesh.cells)
        out_mesh = meshio.read(filepath)
        assert (abs(out_mesh.points - tri_mesh.points) < 1.0e-15).all()
        assert (tri_mesh.cells["triangle"] == out_mesh.cells["triangle"]).all()
    return
