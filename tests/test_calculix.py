import pathlib

import numpy as np
import pytest

import meshio


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells, ref_num_cell_sets",
    [
        ("static_cantilever_calculix.frd", 1705.0, 480, 0),
    ],
)
def test_reference_file(filename, ref_sum, ref_num_cells, ref_num_cell_sets):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "calculix" / filename

    mesh = meshio.read(filename)

    assert np.isclose(np.sum(mesh.points), ref_sum)
    assert sum(len(cells.data) for cells in mesh.cells) == ref_num_cells
    assert len(mesh.cell_sets) == ref_num_cell_sets
