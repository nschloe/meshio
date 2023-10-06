import pathlib

import numpy as np
import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.add_cell_data(helpers.tri_mesh, [("medit:ref", (), int)]),
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.medit.write, meshio.medit.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.mesh")
    # same for binary files
    helpers.generic_io(tmp_path / "test.meshb")
    helpers.generic_io(tmp_path / "test.0.meshb")


# same tests with ugrid format files converted with UGC from
# https://www.simcenter.msstate.edu/


@pytest.mark.parametrize(
    "filename, ref_num_points, ref_num_triangle, ref_num_quad, ref_num_wedge, ref_num_tet, ref_num_hex, ref_tag_counts",
    [
        (
            "sphere_mixed.1.meshb",
            3270,
            864,
            0,
            3024,
            9072,
            0,
            {1: 432, 2: 216, 3: 216},
        ),
        ("hch_strct.4.meshb", 306, 12, 178, 96, 0, 144, {1: 15, 2: 15, 3: 160}),
        ("hch_strct.4.be.meshb", 306, 12, 178, 96, 0, 144, {1: 15, 2: 15, 3: 160}),
        ("cube86.mesh", 39, 72, 0, 0, 86, 0, {1: 14, 2: 14, 3: 14, 4: 8, 5: 14, 6: 8}),
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
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "medit" / filename

    mesh = meshio.read(filename)
    assert mesh.points.shape[0] == ref_num_points
    assert mesh.points.shape[1] == 3

    medit_meshio_id = {
        "triangle": None,
        "quad": None,
        "tetra": None,
        "pyramid": None,
        "wedge": None,
        "hexahedron": None,
    }

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in medit_meshio_id:
            medit_meshio_id[cell_block.type] = i

    # validate element counts
    if ref_num_triangle > 0:
        c = mesh.cells[medit_meshio_id["triangle"]]
        assert c.data.shape == (ref_num_triangle, 3)
    else:
        assert medit_meshio_id["triangle"] is None

    if ref_num_quad > 0:
        c = mesh.cells[medit_meshio_id["quad"]]
        assert c.data.shape == (ref_num_quad, 4)
    else:
        assert medit_meshio_id["quad"] is None

    if ref_num_tet > 0:
        c = mesh.cells[medit_meshio_id["tetra"]]
        assert c.data.shape == (ref_num_tet, 4)
    else:
        assert medit_meshio_id["tetra"] is None

    if ref_num_wedge > 0:
        c = mesh.cells[medit_meshio_id["wedge"]]
        assert c.data.shape == (ref_num_wedge, 6)
    else:
        assert medit_meshio_id["wedge"] is None

    if ref_num_hex > 0:
        c = mesh.cells[medit_meshio_id["hexahedron"]]
        assert c.data.shape == (ref_num_hex, 8)
    else:
        assert medit_meshio_id["hexahedron"] is None

    # validate boundary tags

    # gather tags
    all_tags = []
    for k, c in enumerate(mesh.cells):
        if c.type not in ["triangle", "quad"]:
            continue
        all_tags.append(mesh.cell_data["medit:ref"][k])

    all_tags = np.concatenate(all_tags)

    # validate against known values
    unique, counts = np.unique(all_tags, return_counts=True)
    tags = dict(zip(unique, counts))
    assert tags.keys() == ref_tag_counts.keys()
    for key in tags.keys():
        assert tags[key] == ref_tag_counts[key]
