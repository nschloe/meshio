import pathlib

import numpy as np
import pytest

import meshio

from . import helpers

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tri_mesh,
        helpers.quad_mesh,
        # helpers.tri_quad_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
@pytest.mark.parametrize(
    "accuracy,ext",
    [
        (1.0e-7, ".ugrid"),
        (1.0e-15, ".b8.ugrid"),
        (1.0e-7, ".b4.ugrid"),
        (1.0e-15, ".lb8.ugrid"),
        (1.0e-7, ".lb4.ugrid"),
        (1.0e-15, ".r8.ugrid"),
        (1.0e-7, ".r4.ugrid"),
        (1.0e-15, ".lr8.ugrid"),
        (1.0e-7, ".lr4.ugrid"),
    ],
)
def test_io(mesh, accuracy, ext, tmp_path):
    helpers.write_read(
        tmp_path, meshio.ugrid.write, meshio.ugrid.read, mesh, accuracy, ext
    )


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.lb8.ugrid")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.lb8.ugrid")


# sphere_mixed.1.lb8.ugrid and hch_strct.4.lb8.ugrid created
# using the codes from http://cfdbooks.com
@pytest.mark.parametrize(
    "filename, ref_num_points, ref_num_triangle, ref_num_quad, ref_num_wedge, ref_num_tet, ref_num_hex, ref_tag_counts",
    [
        (
            "sphere_mixed.1.lb8.ugrid",
            3270,
            864,
            0,
            3024,
            9072,
            0,
            {1: 432, 2: 216, 3: 216},
        ),
        ("hch_strct.4.lb8.ugrid", 306, 12, 178, 96, 0, 144, {1: 15, 2: 15, 3: 160}),
    ],
)
def test_reference_file(
    filename,
    ref_num_points,
    ref_num_triangle,
    ref_num_quad,
    ref_num_wedge,
    ref_num_tet,
    ref_num_hex,
    ref_tag_counts,
):
    filename = this_dir / "meshes" / "ugrid" / filename

    mesh = meshio.read(filename)
    assert mesh.points.shape[0] == ref_num_points
    assert mesh.points.shape[1] == 3

    ugrid_meshio_id = {
        "triangle": None,
        "quad": None,
        "tetra": None,
        "pyramid": None,
        "wedge": None,
        "hexahedron": None,
    }

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in ugrid_meshio_id:
            ugrid_meshio_id[cell_block.type] = i

    # validate element counts
    if ref_num_triangle > 0:
        c = mesh.cells[ugrid_meshio_id["triangle"]]
        assert c.data.shape == (ref_num_triangle, 3)
    else:
        assert ugrid_meshio_id["triangle"] is None

    if ref_num_quad > 0:
        c = mesh.cells[ugrid_meshio_id["quad"]]
        assert c.data.shape == (ref_num_quad, 4)
    else:
        assert ugrid_meshio_id["quad"] is None

    if ref_num_tet > 0:
        c = mesh.cells[ugrid_meshio_id["tetra"]]
        assert mesh.cells[1].data.shape == (ref_num_tet, 4)
    else:
        assert ugrid_meshio_id["tetra"] is None

    if ref_num_wedge > 0:
        c = mesh.cells[ugrid_meshio_id["wedge"]]
        assert c.data.shape == (ref_num_wedge, 6)
    else:
        assert ugrid_meshio_id["wedge"] is None

    if ref_num_hex > 0:
        c = mesh.cells[ugrid_meshio_id["hexahedron"]]
        assert c.data.shape == (ref_num_hex, 8)
    else:
        assert ugrid_meshio_id["hexahedron"] is None

    # validate boundary tags

    # gather tags
    all_tags = []
    for k, c in enumerate(mesh.cells):
        if c.type not in ["triangle", "quad"]:
            continue
        all_tags.append(mesh.cell_data["ugrid:ref"][k])

    all_tags = np.concatenate(all_tags)

    # validate against known values
    unique, counts = np.unique(all_tags, return_counts=True)
    tags = dict(zip(unique, counts))
    assert tags.keys() == ref_tag_counts.keys()
    for key in tags.keys():
        assert tags[key] == ref_tag_counts[key]


def _tet_volume(cell):
    """
    Evaluate the volume of a tetrahedron using the value
    of the deteminant
    | a_x a_y a_z 1 |
    | b_x b_y b_z 1 |
    | c_x c_y c_z 1 |
    | d_x d_y d_z 1 |
    """

    t = np.ones((4, 1))
    cell = np.append(cell, t, axis=1)
    vol = -np.linalg.det(cell) / 6.0
    return vol


def _pyramid_volume(cell):
    """
    Evaluate pyramid volume by splitting it into
    two tetrahedra
    """
    tet0 = cell[[0, 1, 3, 4]]
    tet1 = cell[[1, 2, 3, 4]]

    vol = 0.0
    vol += _tet_volume(tet0)
    vol += _tet_volume(tet1)
    return vol


# ugrid node ordering is the same for all elements except the pyramids. In order to make
# sure we got it right read a cube split into pyramids and evaluate its volume
@pytest.mark.parametrize("filename, volume,accuracy", [("pyra_cube.ugrid", 1.0, 1e-15)])
def test_volume(filename, volume, accuracy):
    filename = this_dir / "meshes" / "ugrid" / filename

    mesh = meshio.read(filename)

    assert mesh.cells[0].type == "pyramid"
    assert mesh.cells[0].data.shape == (6, 5)
    vol = 0.0
    for _cell in mesh.cells[0].data:
        cell = np.array([mesh.points[i] for i in _cell])
        v = _pyramid_volume(cell)
        vol += v
    assert np.isclose(vol, 1.0, accuracy)


def _triangle_area(cell):
    """
    Evaluate triangle area using cross product of two edge vectors
    """
    u = cell[0] - cell[1]
    v = cell[0] - cell[2]
    return np.linalg.norm(np.cross(u, v)) / 2.0


def _quad_area(cell):
    """
    Evaluate triangle area using cross product of two edge vectors
    """
    tri1 = cell[[0, 1, 2]]
    tri2 = cell[[2, 3, 0]]
    return _triangle_area(tri1) + _triangle_area(tri2)


# Test whether any of the surface elements is  connected in the wrong way
@pytest.mark.parametrize(
    "filename, area_tria_ref,area_quad_ref,accuracy",
    [("hch_strct.4.lb8.ugrid", 9587.10463, 74294.529256, 1e-5)],
)
def test_area(filename, area_tria_ref, area_quad_ref, accuracy):
    filename = this_dir / "meshes" / "ugrid" / filename

    mesh = meshio.read(filename)
    ugrid_meshio_id = {
        "triangle": None,
        "quad": None,
        "tetra": None,
        "pyramid": None,
        "wedge": None,
        "hexahedron": None,
    }

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in ugrid_meshio_id:
            ugrid_meshio_id[cell_block.type] = i

    tria = mesh.cells[ugrid_meshio_id["triangle"]]
    total_tri_area = 0
    for _cell in tria.data:
        cell = np.array([mesh.points[i] for i in _cell])
        a = _triangle_area(cell)
        total_tri_area += a
    assert np.isclose(total_tri_area, area_tria_ref, accuracy)

    quad = mesh.cells[ugrid_meshio_id["quad"]]
    total_quad_area = 0
    for _cell in quad.data:
        cell = np.array([mesh.points[i] for i in _cell])
        a = _quad_area(cell)
        total_quad_area += a
    assert np.isclose(total_quad_area, area_quad_ref, accuracy)
