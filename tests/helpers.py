from __future__ import annotations

import copy
import string

import numpy as np

import meshio

# In general:
# Use values with an infinite decimal representation to test precision.

empty_mesh = meshio.Mesh(np.empty((0, 3)), [])

line_mesh = meshio.Mesh(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    [("line", [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]])],
)

tri_mesh_one_cell = meshio.Mesh(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    [("triangle", [[0, 1, 2]])],
)

tri_mesh_2d = meshio.Mesh(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    [("triangle", [[0, 1, 2], [0, 2, 3]])],
)

tri_mesh_5 = meshio.Mesh(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ],
    [("triangle", [[0, 1, 5], [0, 5, 6], [1, 2, 5], [2, 4, 5], [2, 3, 4]])],
)

tri_mesh = meshio.Mesh(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    [("triangle", [[0, 1, 2], [0, 2, 3]])],
)

line_tri_mesh = meshio.Mesh(line_mesh.points, line_mesh.cells + tri_mesh.cells)

triangle6_mesh = meshio.Mesh(
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
    ],
    [("triangle6", [[0, 1, 2, 3, 4, 5], [1, 6, 2, 8, 7, 4]])],
)

quad_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    [("quad", [[0, 1, 4, 5], [1, 2, 3, 4]])],
)

d = 0.1
quad8_mesh = meshio.Mesh(
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
    ],
    [("quad8", [[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 10, 11, 12, 5]])],
)

tri_quad_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    [
        ("triangle", [[0, 1, 5], [0, 5, 6]]),
        ("quad", [[1, 2, 4, 5]]),
        ("triangle", [[2, 3, 4]]),
    ],
)

# same as tri_quad_mesh with reversed cell type order
quad_tri_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    [
        ("quad", [[1, 2, 3, 4]]),
        ("triangle", [[0, 1, 4], [0, 4, 5]]),
    ],
)

tet_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5],
    ],
    [("tetra", [[0, 1, 2, 4], [0, 2, 3, 4]])],
)

tet10_mesh = meshio.Mesh(
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
    ],
    [("tetra10", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])],
)

hex_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    [("hexahedron", [[0, 1, 2, 3, 4, 5, 6, 7]])],
)

wedge_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    [("wedge", [[0, 1, 2, 3, 4, 5]])],
)

pyramid_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ],
    [("pyramid", [[0, 1, 2, 3, 4]])],
)

hex20_mesh = meshio.Mesh(
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
    ],
    [("hexahedron20", [np.arange(20)])],
)

polygon_mesh = meshio.Mesh(
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
    ],
    [
        ("triangle", [[0, 1, 2], [4, 5, 6]]),
        ("quad", [[0, 1, 2, 3]]),
        ("polygon", [[1, 4, 5, 6, 2]]),
        ("polygon", [[0, 3, 7, 8, 9, 10], [1, 3, 7, 8, 9, 10]]),
    ],
)

polygon_mesh_one_cell = meshio.Mesh(
    [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.7, 0.5, 0.0],
        [1.5, 1.2, 0.0],
    ],
    [
        ("polygon", [[0, 2, 3, 4, 1]]),
    ],
)

# Make sure that the polygon cell blocking works.
# This mesh is identical with tri_quad_mesh.
polygon2_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    [
        ("polygon", [[0, 1, 5], [0, 5, 6]]),
        ("polygon", [[1, 2, 4, 5]]),
        ("polygon", [[2, 3, 4]]),
    ],
)

polyhedron_mesh = meshio.Mesh(
    [  # Two layers of a unit square
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    # Split the cube into tets and pyramids.
    [
        (
            "polyhedron4",
            [
                [
                    [1, 2, 5],
                    [1, 2, 7],
                    [1, 5, 7],
                    [2, 5, 7],
                ],
                [
                    [2, 5, 6],
                    [2, 6, 7],
                    [2, 5, 7],
                    [5, 6, 7],
                ],
            ],
        ),
        (
            "polyhedron5",
            [
                [
                    # np.asarray on this causes a numpy warning
                    # ```
                    # VisibleDeprecationWarning: Creating an ndarray from ragged nested
                    # sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays
                    # with different lengths or shapes) is deprecated. If you meant to
                    # do this, you must specify 'dtype=object' when creating the
                    # ndarray.
                    # ```
                    # TODO come up with a better data structure for polyhedra
                    [0, 1, 2, 3],  # pyramid base is a rectangle
                    [0, 1, 7],
                    [1, 2, 7],
                    [2, 3, 7],
                    [3, 0, 7],
                ],
                [
                    [0, 1, 5],  # pyramid base split in two triangles
                    [0, 4, 5],
                    [0, 1, 7],
                    [1, 5, 7],
                    [5, 4, 7],
                    [0, 4, 7],
                ],
            ],
        ),
    ],
)

# From <https://github.com/nschloe/meshio/issues/1065>:
lagrange_high_order_mesh = meshio.Mesh(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.14285714924, 0.0, 0.0],
        [0.28571429849, 0.0, 0.0],
        [0.42857143283, 0.0, 0.0],
        [0.57142859697, 0.0, 0.0],
        [0.71428573132, 0.0, 0.0],
        [0.85714286566, 0.0, 0.0],
        [0.0, 0.14285714924, 0.0],
        [0.14285714924, 0.14285714924, 0.0],
        [0.28571428359, 0.14285714924, 0.0],
        [0.42857144773, 0.14285714924, 0.0],
        [0.57142858207, 0.14285714924, 0.0],
        [0.71428571641, 0.14285714924, 0.0],
        [0.85714285076, 0.14285714924, 0.0],
        [0.0, 0.28571429849, 0.0],
        [0.14285713434, 0.28571429849, 0.0],
        [0.28571429849, 0.28571429849, 0.0],
        [0.42857143283, 0.28571429849, 0.0],
        [0.57142856717, 0.28571429849, 0.0],
        [0.71428570151, 0.28571429849, 0.0],
        [0.0, 0.42857143283, 0.0],
        [0.14285716414, 0.42857143283, 0.0],
        [0.28571429849, 0.42857143283, 0.0],
        [0.42857143283, 0.42857143283, 0.0],
        [0.57142856717, 0.42857143283, 0.0],
        [0.0, 0.57142859697, 0.0],
        [0.14285713434, 0.57142859697, 0.0],
        [0.28571426868, 0.57142859697, 0.0],
        [0.42857140303, 0.57142859697, 0.0],
        [0.0, 0.71428573132, 0.0],
        [0.14285713434, 0.71428573132, 0.0],
        [0.28571426868, 0.71428573132, 0.0],
        [0.0, 0.85714286566, 0.0],
        [0.14285713434, 0.85714286566, 0.0],
        [0.0, 0.0, 0.14285714924],
        [0.14285714179, 0.0, 0.14285714924],
        [0.28571429104, 0.0, 0.14285714924],
        [0.42857142538, 0.0, 0.14285714924],
        [0.57142855972, 0.0, 0.14285714924],
        [0.71428569406, 0.0, 0.14285714924],
        [0.8571428284, 0.0, 0.14285714924],
        [0.0, 0.14285714179, 0.14285714924],
        [0.14285714924, 0.14285714179, 0.14285714924],
        [0.28571428359, 0.14285714179, 0.14285714924],
        [0.42857141793, 0.14285714179, 0.14285714924],
        [0.57142855227, 0.14285714179, 0.14285714924],
        [0.71428568661, 0.14285714179, 0.14285714924],
        [0.0, 0.28571429104, 0.14285714924],
        [0.14285713434, 0.28571429104, 0.14285714924],
        [0.28571426868, 0.28571429104, 0.14285714924],
        [0.42857140303, 0.28571429104, 0.14285714924],
        [0.57142853737, 0.28571429104, 0.14285714924],
        [0.0, 0.42857142538, 0.14285714924],
        [0.14285713434, 0.42857142538, 0.14285714924],
        [0.28571426868, 0.42857142538, 0.14285714924],
        [0.42857140303, 0.42857142538, 0.14285714924],
        [0.0, 0.57142855972, 0.14285714924],
        [0.14285713434, 0.57142855972, 0.14285714924],
        [0.28571426868, 0.57142855972, 0.14285714924],
        [0.0, 0.71428569406, 0.14285714924],
        [0.14285713434, 0.71428569406, 0.14285714924],
        [0.0, 0.8571428284, 0.14285714924],
        [0.0, 0.0, 0.28571429849],
        [0.14285714924, 0.0, 0.28571429849],
        [0.28571428359, 0.0, 0.28571429849],
        [0.42857144773, 0.0, 0.28571429849],
        [0.57142858207, 0.0, 0.28571429849],
        [0.71428571641, 0.0, 0.28571429849],
        [0.0, 0.14285714924, 0.28571429849],
        [0.14285713434, 0.14285714924, 0.28571429849],
        [0.28571429849, 0.14285714924, 0.28571429849],
        [0.42857143283, 0.14285714924, 0.28571429849],
        [0.57142856717, 0.14285714924, 0.28571429849],
        [0.0, 0.28571428359, 0.28571429849],
        [0.14285716414, 0.28571428359, 0.28571429849],
        [0.28571429849, 0.28571428359, 0.28571429849],
        [0.42857143283, 0.28571428359, 0.28571429849],
        [0.0, 0.42857144773, 0.28571429849],
        [0.14285713434, 0.42857144773, 0.28571429849],
        [0.28571426868, 0.42857144773, 0.28571429849],
        [0.0, 0.57142858207, 0.28571429849],
        [0.14285713434, 0.57142858207, 0.28571429849],
        [0.0, 0.71428571641, 0.28571429849],
        [0.0, 0.0, 0.42857143283],
        [0.14285714924, 0.0, 0.42857143283],
        [0.28571428359, 0.0, 0.42857143283],
        [0.42857141793, 0.0, 0.42857143283],
        [0.57142855227, 0.0, 0.42857143283],
        [0.0, 0.14285714924, 0.42857143283],
        [0.14285713434, 0.14285714924, 0.42857143283],
        [0.28571426868, 0.14285714924, 0.42857143283],
        [0.42857140303, 0.14285714924, 0.42857143283],
        [0.0, 0.28571428359, 0.42857143283],
        [0.14285713434, 0.28571428359, 0.42857143283],
        [0.28571426868, 0.28571428359, 0.42857143283],
        [0.0, 0.42857141793, 0.42857143283],
        [0.14285713434, 0.42857141793, 0.42857143283],
        [0.0, 0.57142855227, 0.42857143283],
        [0.0, 0.0, 0.57142859697],
        [0.14285713434, 0.0, 0.57142859697],
        [0.28571429849, 0.0, 0.57142859697],
        [0.42857143283, 0.0, 0.57142859697],
        [0.0, 0.14285713434, 0.57142859697],
        [0.14285716414, 0.14285713434, 0.57142859697],
        [0.28571429849, 0.14285713434, 0.57142859697],
        [0.0, 0.28571429849, 0.57142859697],
        [0.14285713434, 0.28571429849, 0.57142859697],
        [0.0, 0.42857143283, 0.57142859697],
        [0.0, 0.0, 0.71428573132],
        [0.14285713434, 0.0, 0.71428573132],
        [0.28571426868, 0.0, 0.71428573132],
        [0.0, 0.14285713434, 0.71428573132],
        [0.14285713434, 0.14285713434, 0.71428573132],
        [0.0, 0.28571426868, 0.71428573132],
        [0.0, 0.0, 0.85714286566],
        [0.14285716414, 0.0, 0.85714286566],
        [0.0, 0.14285716414, 0.85714286566],
    ],
    [
        (
            "VTK_LAGRANGE_TETRAHEDRON",
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    16,
                    22,
                    27,
                    31,
                    34,
                    36,
                    35,
                    32,
                    28,
                    23,
                    17,
                    10,
                    37,
                    65,
                    86,
                    101,
                    111,
                    117,
                    43,
                    70,
                    90,
                    104,
                    113,
                    118,
                    64,
                    85,
                    100,
                    110,
                    116,
                    119,
                    38,
                    42,
                    112,
                    39,
                    40,
                    41,
                    69,
                    89,
                    103,
                    102,
                    87,
                    66,
                    67,
                    68,
                    88,
                    63,
                    115,
                    49,
                    84,
                    99,
                    109,
                    107,
                    94,
                    75,
                    54,
                    58,
                    61,
                    82,
                    97,
                    79,
                    44,
                    114,
                    62,
                    71,
                    91,
                    105,
                    108,
                    98,
                    83,
                    59,
                    55,
                    50,
                    76,
                    95,
                    80,
                    11,
                    33,
                    15,
                    18,
                    24,
                    29,
                    30,
                    26,
                    21,
                    14,
                    13,
                    12,
                    19,
                    25,
                    20,
                    45,
                    48,
                    60,
                    106,
                    46,
                    47,
                    53,
                    57,
                    56,
                    51,
                    72,
                    92,
                    74,
                    93,
                    81,
                    96,
                    73,
                    78,
                    77,
                    52,
                ]
            ],
        )
    ],
)


def add_point_data(mesh, dim, num_tags=2, seed=0, dtype=float):
    rng = np.random.default_rng(seed)

    mesh2 = copy.deepcopy(mesh)

    shape = (len(mesh.points),) if dim == 1 else (len(mesh.points), dim)
    data = [(100 * rng.random(shape)).astype(dtype) for _ in range(num_tags)]

    mesh2.point_data = {string.ascii_lowercase[k]: d for k, d in enumerate(data)}
    return mesh2


def add_cell_data(mesh, specs: list[tuple[str, tuple[int, ...], type]]):
    mesh2 = copy.deepcopy(mesh)

    rng = np.random.default_rng(0)

    mesh2.cell_data = {
        name: [
            (100 * rng.random((len(cellblock),) + shape)).astype(dtype)
            for cellblock in mesh.cells
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
    mesh2.field_data = {"a": np.array(value, dtype=dtype)}
    return mesh2


def add_point_sets(mesh):
    mesh2 = copy.deepcopy(mesh)
    n = len(mesh.points)
    mesh2.point_sets = {
        "fixed": np.arange(0, n // 2),
        "loose": np.arange(n // 2, n),
    }
    return mesh2


def add_cell_sets(mesh):
    mesh2 = copy.deepcopy(mesh)
    assert len(mesh.cells) == 1
    n = len(mesh.cells[0])
    mesh2.cell_sets = {
        "grain0": [np.arange(0, n // 2)],
        "grain1": [np.arange(n // 2, n)],
    }
    return mesh2


def write_read(tmp_path, writer, reader, input_mesh, atol, extension=".dat"):
    """Write and read a file, and make sure the data is the same as before."""
    in_mesh = copy.deepcopy(input_mesh)

    p = tmp_path / ("test" + extension)
    writer(p, input_mesh)
    mesh = reader(p)

    # Make sure the output is writeable
    assert mesh.points.flags["WRITEABLE"]
    for cells in input_mesh.cells:
        if isinstance(cells.data, np.ndarray):
            assert cells.data.flags["WRITEABLE"]
        else:
            # This is assumed to be a polyhedron
            for cell in cells.data:
                for face in cell:
                    assert face.flags["WRITEABLE"]

    # assert that the input mesh hasn't changed at all
    assert in_mesh.points.dtype == input_mesh.points.dtype
    assert np.allclose(in_mesh.points, input_mesh.points, atol=atol, rtol=0.0)
    for c0, c1 in zip(in_mesh.cells, input_mesh.cells):
        if c0.type.startswith("polyhedron"):
            continue
        assert c0.type == c1.type
        assert c0.data.shape == c1.data.shape, f"{c0.data.shape} != {c1.data.shape}"
        assert c0.data.dtype == c1.data.dtype, f"{c0.data.dtype} != {c1.data.dtype}"
        assert np.all(c0.data == c1.data)

    # Numpy's array_equal is too strict here, cf.
    # <https://mail.python.org/archives/list/numpy-discussion@python.org/message/3LUSBW5BGD6NY6I76W5WSBOEBHNDKA4Y/>.
    # Use allclose.
    if in_mesh.points.shape[0] == 0:
        assert mesh.points.shape[0] == 0
    else:
        n = in_mesh.points.shape[1]
        assert np.allclose(in_mesh.points, mesh.points[:, :n], atol=atol, rtol=0.0)

    # To avoid errors from sorted (below), specify the key as first cell type then index
    # of the first point of the first cell. This may still lead to comparison of what
    # should be different blocks, but chances seem low.
    def cell_sorter(cell):
        if cell.type.startswith("polyhedron"):
            # Polyhedra blocks should be well enough distinguished by their type
            return cell.type
        else:
            return (cell.type, cell.data[0, 0])

    # to make sure we are testing the same type of cells we sort the list
    for cells0, cells1 in zip(
        sorted(input_mesh.cells, key=cell_sorter), sorted(mesh.cells, key=cell_sorter)
    ):
        assert cells0.type == cells1.type, f"{cells0.type} != {cells1.type}"

        if cells0.type.startswith("polyhedron"):
            # Special treatment of polyhedron cells
            # Data is a list (one item per cell) of numpy arrays
            for c_in, c_out in zip(cells0.data, cells1.data):
                for face_in, face_out in zip(c_in, c_out):
                    assert np.allclose(face_in, face_out, atol=atol, rtol=0.0)
        else:
            print("a", cells0.data)
            print("b", cells1.data)
            assert np.array_equal(cells0.data, cells1.data)

    for key in input_mesh.point_data.keys():
        assert np.allclose(
            input_mesh.point_data[key], mesh.point_data[key], atol=atol, rtol=0.0
        )

    for name, cell_type_data in input_mesh.cell_data.items():
        for d0, d1 in zip(cell_type_data, mesh.cell_data[name]):
            # assert d0.dtype == d1.dtype, (d0.dtype, d1.dtype)
            assert np.allclose(d0, d1, atol=atol, rtol=0.0)

    print()
    print("helpers:")
    print(input_mesh.field_data)
    print()
    print(mesh.field_data)
    for name, data in input_mesh.field_data.items():
        if isinstance(data, list):
            assert data == mesh.field_data[name]
        else:
            assert np.allclose(data, mesh.field_data[name], atol=atol, rtol=0.0)

    # Test of cell sets (assumed to be a list of numpy arrays),
    for name, data in input_mesh.cell_sets.items():
        # Skip the test if the key is not in the read cell set
        if name not in mesh.cell_sets.keys():
            continue
        data2 = mesh.cell_sets[name]
        for var1, var2 in zip(data, data2):
            assert np.allclose(var1, var2, atol=atol, rtol=0.0)


def generic_io(filepath):
    meshio.write_points_cells(filepath, tri_mesh.points, tri_mesh.cells)
    out_mesh = meshio.read(filepath)
    assert (abs(out_mesh.points - tri_mesh.points) < 1.0e-15).all()
    for c0, c1 in zip(tri_mesh.cells, out_mesh.cells):
        assert c0.type == c1.type
        assert (c0.data == c1.data).all()
