import pathlib

import helpers
import numpy
import pytest

import meshio

test_set = [
    helpers.tri_mesh_2d,
    helpers.tet_mesh,
    helpers.hex_mesh,
]
this_dir = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize("mesh", test_set)
def test(mesh):
    def writer(*args, **kwargs):
        return meshio.su2.write(*args, **kwargs)

    helpers.write_read(writer, meshio.su2.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_num_cells, ref_num_points,ref_num_unique_tags,sum_tags",
    [("square.su2", 16, 9, 4, 20), ("mixgrid.su2", 30, 16, 6, 62)],
)
def test_structured(
    filename, ref_num_cells, ref_num_points, ref_num_unique_tags, sum_tags
):
    filename = this_dir / "meshes" / "su2" / filename

    mesh = meshio.read(filename)

    assert sum(len(block.data) for block in mesh.cells) == ref_num_cells
    assert len(mesh.points) == ref_num_points

    all_tags = numpy.concatenate([tags for tags in mesh.cell_data["su2:tag"]])

    assert sum(all_tags) == sum_tags

    all_unique_tags = numpy.unique(all_tags)

    assert len(all_unique_tags) == ref_num_unique_tags + 1
