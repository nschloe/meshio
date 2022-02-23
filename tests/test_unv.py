import pathlib

import numpy as np
import pytest

import meshio
from . import helpers

test_set = [
    helpers.tri_mesh_2d,
    helpers.tet_mesh,
    helpers.hex_mesh,
]

this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize("mesh", test_set)
def test(mesh, tmp_path):
    # helpers.write_read(tmp_path, meshio.su2.write, meshio.su2.read, mesh, 1.0e-15)
    helpers.write_read(tmp_path, meshio.unv.write, meshio.unv.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_num_cells, ref_num_points,ref_num_unique_tags,sum_tags",
    [("simple_cube.unv", 42, 12, 1, 2)],
)
def test_structured(
    filename, ref_num_cells, ref_num_points, ref_num_unique_tags, sum_tags
):
    filename = this_dir / "meshes" / "unv" / filename

    mesh = meshio.unv.read(filename)

    assert sum(len(block.data) for block in mesh.cells) == ref_num_cells
    assert len(mesh.points) == ref_num_points

    all_tags = np.concatenate([tags for tags in mesh.cell_data["cell_tags"]])

    assert sum(all_tags) == sum_tags

    all_unique_tags = np.unique(all_tags)

    assert len(all_unique_tags) == ref_num_unique_tags + 1


@pytest.mark.parametrize(
    "filename",
    [
        "empty.unv",
        "no_points.unv",
        "no_cells.unv",
        "cells_no_points.unv",
        "unknown_group.unv",
        "non_matching_num_nodes.unv",
        "non_matching_num_nodes2.unv",
        "unsupported_element_type.unv",
    ],
)
def test_corrupted(filename):
    with pytest.raises(meshio.ReadError):
        _ = meshio.unv.read(this_dir / "meshes" / "unv" / filename)


def test_dofs():
    filename = this_dir / "meshes" / "unv" / "block_with_inlet_outlet.unv"
    mesh = meshio.unv.read(filename)
    patch_names = set([x[0] for x in mesh.point_tags.values()])

    assert len(patch_names) == 2
    assert patch_names == set(["inlet", "Outlet"])

    tags = list(mesh.point_tags.keys())
    inlet_tag = tags[0] if mesh.point_tags[tags[0]] == "inlet" else tags[1]

    assert (
        len(mesh.point_data["point_tags"][mesh.point_data["point_tags"] == inlet_tag])
        == 18
    )


def test_groups():
    filename = this_dir / "meshes" / "unv" / "threeZonesAll.unv"
    mesh = meshio.unv.read(filename)

    tag_name_tag_id = {v[0]: k for k, v in mesh.cell_tags.items()}
    i = tag_name_tag_id["Face_16"]
    counter = 0

    for cell_data in mesh.cell_data["cell_tags"]:
        counter += np.count_nonzero(cell_data[cell_data == i])

    assert counter == 16


@pytest.mark.parametrize("mesh_fname", ["bad_units.unv", "unknown_units.unv"])
def test_units(mesh_fname):
    filename = this_dir / "meshes" / "unv" / "simple_cube.unv"
    mesh = meshio.unv.read(filename)
    assert mesh.field_data["unit_system"] == "SI: Meter (newton)"

    with pytest.raises(KeyError):
        filename = this_dir / "meshes" / "unv" / mesh_fname
        mesh = meshio.unv.read(filename)
        _ = mesh.field_data["unit_system"]
