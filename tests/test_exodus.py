import numpy as np
import pytest

import meshio

from . import helpers

netCDF4 = pytest.importorskip("netCDF4")

test_set = [
    helpers.empty_mesh,
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_point_data(helpers.tri_mesh, 2),
    helpers.add_point_data(helpers.tri_mesh, 3),
    helpers.add_point_sets(helpers.tri_mesh),
    helpers.add_point_sets(helpers.tet_mesh),
]


@pytest.mark.parametrize("mesh", test_set)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.exodus.write, meshio.exodus.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.e")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.e")


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("elem_type", ["HEX20", "hex20"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_read_hex20_connectivity(tmp_path, mixed, elem_type, dtype):
    # Exodus numbers the vertical midside nodes before the top-face midside
    # nodes (Exodus manual, Figure 4.13); meshio uses VTK's opposite order.
    connectivity = np.arange(1, 41, dtype=dtype).reshape(2, 20)
    expected = np.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15]]
    ) + [[0], [20]]
    blocks = [(elem_type, "hexahedron20", connectivity, expected)]
    if mixed:
        blocks = [
            ("HEX8", "hexahedron", connectivity[:, :8], connectivity[:, :8] - 1),
            *blocks,
            ("QUAD8", "quad8", connectivity[:, :8], connectivity[:, :8] - 1),
            *blocks,
        ]
    points = np.arange(120, dtype=float).reshape(40, 3)
    filename = tmp_path / "hex20.e"
    # Construct the input directly, independently of meshio's writer.
    with netCDF4.Dataset(filename, "w") as nc:
        nc.createDimension("num_nodes", len(points))
        nc.createDimension("num_dim", 3)
        nc.createVariable("coord", "f8", ("num_dim", "num_nodes"))[:] = points.T
        for k, (exodus_type, _, data, _) in enumerate(blocks, start=1):
            dims = (f"num_el_in_blk{k}", f"num_nod_per_el{k}")
            nc.createDimension(dims[0], data.shape[0])
            nc.createDimension(dims[1], data.shape[1])
            variable = nc.createVariable(f"connect{k}", data.dtype, dims)
            variable.elem_type = exodus_type
            variable[:] = data

    mesh = meshio.exodus.read(filename)
    np.testing.assert_array_equal(mesh.points, points)
    assert len(mesh.cells) == len(blocks)
    for cell, (_, meshio_type, data, expected) in zip(mesh.cells, blocks):
        assert cell.type == meshio_type
        # Preserve the existing masked-array subtraction's dtype, which may
        # differ from the on-disk dtype between NumPy versions.
        assert cell.data.dtype == (np.ma.array(data) - 1).dtype
        np.testing.assert_array_equal(cell.data, expected)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_write_hex20_connectivity(tmp_path, mixed, dtype):
    connectivity = np.arange(40, dtype=dtype).reshape(2, 20)
    expected = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 19, 20, 13, 14, 15, 16]]
    ) + [[0], [20]]
    blocks = [("hexahedron20", "HEX20", connectivity, expected)]
    if mixed:
        blocks = [
            ("hexahedron", "HEX8", connectivity[:, :8], connectivity[:, :8] + 1),
            *blocks,
            ("quad8", "QUAD8", connectivity[:, :8], connectivity[:, :8] + 1),
            *blocks,
        ]
    mesh = meshio.Mesh(
        np.arange(120, dtype=float).reshape(40, 3),
        [(cell_type, data) for cell_type, _, data, _ in blocks],
    )
    original = [cell.data.copy() for cell in mesh.cells]
    for cell in mesh.cells:
        cell.data.setflags(write=False)
    filename = tmp_path / "hex20.e"
    meshio.exodus.write(filename, mesh)

    # Inspect raw Exodus connectivity, independently of meshio's reader.
    with netCDF4.Dataset(filename) as nc:
        assert len(nc.dimensions["num_el_blk"]) == len(blocks)
        for k, (_, exodus_type, _, expected) in enumerate(blocks, start=1):
            variable = nc.variables[f"connect{k}"]
            assert variable.elem_type == exodus_type
            assert variable.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(variable[:], expected)
    for cell, data in zip(mesh.cells, original):
        np.testing.assert_array_equal(cell.data, data)
