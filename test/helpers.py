import copy
import os
import string
import tempfile
from pathlib import Path

import numpy

import meshio

TEST_DIR = Path(__file__).resolve().parent
MESHES_DIR = TEST_DIR / "meshes"

# In general:
# Use values with an infinite decimal representation to test precision.

line_mesh = meshio.Mesh(
    numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    / 3,
    [("line", numpy.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]))],
)

tri_mesh_2d = meshio.Mesh(
    numpy.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) / 3,
    [("triangle", numpy.array([[0, 1, 2], [0, 2, 3]]))],
)

tri_mesh = meshio.Mesh(
    numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    / 3,
    [("triangle", numpy.array([[0, 1, 2], [0, 2, 3]]))],
)

line_tri_mesh = meshio.Mesh(line_mesh.points, line_mesh.cells + tri_mesh.cells)

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
    [("triangle6", numpy.array([[0, 1, 2, 3, 4, 5], [1, 6, 2, 8, 7, 4]]))],
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
    [("quad", numpy.array([[0, 1, 4, 5], [1, 2, 3, 4]]))],
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
    [("quad8", numpy.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 10, 11, 12, 5]]))],
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
    [
        ("triangle", numpy.array([[0, 1, 4], [0, 4, 5]])),
        ("quad", numpy.array([[1, 2, 3, 4]])),
    ],
)

# same as tri_quad_mesh with reversed cell type order
quad_tri_mesh = meshio.Mesh(
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
    [
        ("quad", numpy.array([[1, 2, 3, 4]])),
        ("triangle", numpy.array([[0, 1, 4], [0, 4, 5]])),
    ],
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
    [("tetra", numpy.array([[0, 1, 2, 4], [0, 2, 3, 4]]))],
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
    [("tetra10", numpy.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))],
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
    [("hexahedron", numpy.array([[0, 1, 2, 3, 4, 5, 6, 7]]))],
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
    [("hexahedron20", numpy.array([numpy.arange(20)]))],
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
    [
        ("triangle", numpy.array([[0, 1, 2], [4, 5, 6]])),
        ("quad", numpy.array([[0, 1, 2, 3]])),
        ("polygon5", numpy.array([[1, 4, 5, 6, 2]])),
        ("polygon6", numpy.array([[0, 3, 7, 8, 9, 10], [1, 3, 7, 8, 9, 10]])),
    ],
)

polyhedron_mesh = meshio.Mesh(
    numpy.array(
        [  # Two layers of a unit square
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
    [  # Split the cube into tets and pyramids.
        (
            "polyhedron4",
            [
                [
                    numpy.array([1, 2, 5]),
                    numpy.array([1, 2, 7]),
                    numpy.array([1, 5, 7]),
                    numpy.array([2, 5, 7]),
                ],
                [
                    numpy.array([2, 5, 6]),
                    numpy.array([2, 6, 7]),
                    numpy.array([2, 5, 7]),
                    numpy.array([5, 6, 7]),
                ],
            ],
        ),
        (
            "polyhedron5",
            [
                [
                    numpy.array([0, 1, 2, 3]),  # pyramid base is a rectangle
                    numpy.array([0, 1, 7]),
                    numpy.array([1, 2, 7]),
                    numpy.array([2, 3, 7]),
                    numpy.array([3, 0, 7]),
                ],
                [
                    numpy.array([0, 1, 5]),  # pyramid base split in two triangles
                    numpy.array([0, 4, 5]),
                    numpy.array([0, 1, 7]),
                    numpy.array([1, 5, 7]),
                    numpy.array([5, 4, 7]),
                    numpy.array([0, 4, 7]),
                ],
            ],
        ),
    ],
)


def add_point_data(mesh, dim, num_tags=2, seed=0, dtype=numpy.float):
    numpy.random.seed(seed)
    mesh2 = copy.deepcopy(mesh)

    shape = (len(mesh.points),) if dim == 1 else (len(mesh.points), dim)
    data = [(100 * numpy.random.rand(*shape)).astype(dtype) for _ in range(num_tags)]

    mesh2.point_data = {string.ascii_lowercase[k]: d for k, d in enumerate(data)}
    return mesh2


def add_cell_data(mesh, specs):
    mesh2 = copy.deepcopy(mesh)
    numpy.random.seed(0)
    mesh2.cell_data = {
        name: [
            (100 * numpy.random.rand(*((len(cells),) + shape))).astype(dtype)
            for _, cells in mesh.cells
        ]
        for name, shape, dtype in specs
    }
    # Keep cell-data from the original mesh. This is needed to preserve
    # face-cell relations for polyhedral meshes.
    for key, val in mesh.cell_data.items():
        mesh2.cell_data[key] = val
    return mesh2


def add_field_data(mesh, value, dtype):
    mesh2 = copy.deepcopy(mesh)
    mesh2.field_data = {"a": numpy.array(value, dtype=dtype)}
    return mesh2


def add_point_sets(mesh):
    mesh2 = copy.deepcopy(mesh)
    mesh2.point_sets = {"fixed": numpy.array([1, 2])}
    return mesh2


def add_cell_sets(mesh):
    mesh2 = copy.deepcopy(mesh)
    assert len(mesh.cells) == 1
    n = len(mesh.cells[0])
    mesh2.cell_sets = {
        "grain0": [numpy.array([0])],
        "grain1": [numpy.arange(1, n)],
    }
    return mesh2


def write_read(writer, reader, input_mesh, atol, extension=".dat"):
    """Write and read a file, and make sure the data is the same as before."""
    in_mesh = copy.deepcopy(input_mesh)

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test" + extension)
        p = Path(filepath)
        writer(p, input_mesh)
        mesh = reader(p)

    # Make sure the output is writeable
    assert mesh.points.flags["WRITEABLE"]
    for cells in input_mesh.cells:
        if isinstance(cells.data, numpy.ndarray):
            assert cells.data.flags["WRITEABLE"]
        else:
            # This is assumed to be a polyhedron
            for cell in cells.data:
                for face in cell:
                    assert face.flags["WRITEABLE"]

    # assert that the input mesh hasn't changed at all
    assert numpy.allclose(in_mesh.points, input_mesh.points, atol=atol, rtol=0.0)

    # Numpy's array_equal is too strict here, cf.
    # <https://mail.scipy.org/pipermail/numpy-discussion/2015-December/074410.html>.
    # Use allclose.
    n = in_mesh.points.shape[1]
    assert numpy.allclose(in_mesh.points, mesh.points[:, :n], atol=atol, rtol=0.0)

    # To avoid errors from sorted (below), specify the key as first cell type
    # then index of the first point of the first cell. This may still lead to
    # comparison of what should be different blocks, but chances seem low.
    def cell_sorter(cell):
        if cell.type[:10] == "polyhedron":
            # Polyhedra blocks should be well enough distinguished by their type
            return cell.type
        else:
            return (cell.type, cell.data[0, 0])

    # to make sure we are testing same type of cells we sort the list
    for cells0, cells1 in zip(
        sorted(input_mesh.cells, key=cell_sorter), sorted(mesh.cells, key=cell_sorter)
    ):
        assert cells0.type == cells1.type, f"{cells0.type} != {cells1.type}"

        if cells0.type[:10] == "polyhedron":
            # Special treatment of polyhedron cells
            # Data is a list (one item per cell) of numpy arrays
            for c_in, c_out in zip(cells0.data, cells1.data):
                for face_in, face_out in zip(c_in, c_out):
                    assert numpy.allclose(face_in, face_out, atol=atol, rtol=0.0)
        else:
            assert numpy.array_equal(cells0.data, cells1.data)

    for key in input_mesh.point_data.keys():
        assert numpy.allclose(
            input_mesh.point_data[key], mesh.point_data[key], atol=atol, rtol=0.0
        )

    for name, cell_type_data in input_mesh.cell_data.items():
        for d0, d1 in zip(cell_type_data, mesh.cell_data[name]):
            # assert d0.dtype == d1.dtype, (d0.dtype, d1.dtype)
            assert numpy.allclose(d0, d1, atol=atol, rtol=0.0)

    for name, data in input_mesh.field_data.items():
        assert numpy.allclose(data, mesh.field_data[name], atol=atol, rtol=0.0)

    # Test of cell sets (assumed to be a list of numpy arrays),
    for name, data in input_mesh.cell_sets.items():
        # Skip the test if the key is not in the read cell set
        if name not in mesh.cell_sets.keys():
            continue
        data2 = mesh.cell_sets[name]
        for var1, var2 in zip(data, data2):
            assert numpy.allclose(var1, var2, atol=atol, rtol=0.0)


def generic_io(filename):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, filename)
        meshio.write_points_cells(filepath, tri_mesh.points, tri_mesh.cells)
        out_mesh = meshio.read(filepath)
        assert (abs(out_mesh.points - tri_mesh.points) < 1.0e-15).all()
        for c0, c1 in zip(tri_mesh.cells, out_mesh.cells):
            assert c0.type == c1.type
            assert (c0.data == c1.data).all()
