"""Unsigned mesh connectivity must serialize to signed ANSYS cell sections."""

import struct

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import meshio


@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "uint64", ">u4", ">u8", "int32", "int64"]
)
@pytest.mark.parametrize("binary", [False, True])
def test_connectivity_roundtrip(dtype, binary, tmp_path):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    triangle = np.array([[0, 1, 2]], dtype=dtype)
    tetra = np.array([[0, 1, 2, 3]], dtype=dtype)
    mesh = meshio.Mesh(points, [("triangle", triangle), ("tetra", tetra)])
    before = [c.data.copy() for c in mesh.cells]
    for c in mesh.cells:
        c.data.flags.writeable = False
    path = tmp_path / "mesh.msh"
    meshio.write(path, mesh, file_format="ansys", binary=binary)
    result = meshio.read(path, file_format="ansys")
    assert_allclose(result.points, points, atol=0, rtol=0)
    assert [c.type for c in result.cells] == ["triangle", "tetra"]
    for c, expected, original in zip(result.cells, before, mesh.cells):
        assert_array_equal(c.data, expected)
        assert_array_equal(original.data, expected)
        assert original.data.dtype == np.dtype(dtype)


def test_gmsh_to_ansys(tmp_path):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    gmsh = tmp_path / "source.msh"
    meshio.write(gmsh, meshio.Mesh(points, [("tetra", cells)]), file_format="gmsh")
    mesh = meshio.read(gmsh, file_format="gmsh")
    original = mesh.cells[0].data.copy()
    ansys = tmp_path / "converted.msh"
    meshio.write(ansys, mesh, file_format="ansys")
    result = meshio.read(ansys, file_format="ansys")
    assert_allclose(result.points, points, atol=0, rtol=0)
    assert_array_equal(result.cells[0].data, cells)
    assert_array_equal(mesh.cells[0].data, original)


@pytest.mark.parametrize("value", [2**63 - 1, 2**63, 2**64 - 1])
def test_unsigned_ids_outside_ansys_range(value, tmp_path):
    mesh = meshio.Mesh(
        np.zeros((3, 3)), [("triangle", np.array([[0, 1, value]], dtype=np.uint64))]
    )
    with pytest.raises(meshio.WriteError, match="signed 64-bit"):
        meshio.ansys.write(tmp_path / "overflow.msh", mesh)


@pytest.mark.parametrize("value", [2**31, 2**32 - 1, 2**63 - 2])
def test_unsigned_binary_wire_width(value, tmp_path):
    # Inspect serialization limits without allocating billions of mesh points.
    mesh = meshio.Mesh(
        np.zeros((3, 3)),
        [("triangle", np.array([[0, 1, value]], dtype=np.uint64))],
    )
    path = tmp_path / "wide.msh"
    meshio.ansys.write(path, mesh)
    data = path.read_bytes().split(b"(3012 (1 0 0 1 1)(\n", 1)[1]
    assert data[:24] == struct.pack("=qqq", 1, 2, value + 1)
