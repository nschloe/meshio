import os
import pathlib
import tempfile

import numpy
import pytest

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
    ],
)
def test(mesh):
    def writer(*args, **kwargs):
        return meshio.abaqus.write(*args, **kwargs)

    helpers.write_read(writer, meshio.abaqus.read, mesh, 1.0e-15)


@pytest.mark.parametrize(
    "filename, ref_sum, ref_num_cells, ref_num_cell_sets",
    [
        ("UUea.inp", 4950.0, 50, 10),
        ("nle1xf3c.inp", 32.215275528, 12, 3),
        ("element_elset.inp", 6.0, 2, 3),
    ],
)
def test_reference_file(filename, ref_sum, ref_num_cells, ref_num_cell_sets):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "abaqus" / filename

    mesh = meshio.read(filename)

    assert numpy.isclose(numpy.sum(mesh.points), ref_sum)
    assert sum([len(cells.data) for cells in mesh.cells]) == ref_num_cells
    assert len(mesh.cell_sets) == ref_num_cell_sets


def test_elset():
    points = numpy.array(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.5, 0.0], [0.0, 0.5, 0.0]]
    )
    cells = [
        ("triangle", numpy.array([[0, 1, 2]])),
        ("triangle", numpy.array([[0, 1, 3]])),
    ]
    cell_sets = {
        "right": [numpy.array([0]), numpy.array([])],
        "left": [numpy.array([]), numpy.array([1])],
    }
    mesh_ref = meshio.Mesh(points, cells, cell_sets=cell_sets)

    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test.inp")
        meshio.abaqus.write(filepath, mesh_ref)
        mesh = meshio.abaqus.read(filepath)

    assert numpy.allclose(mesh_ref.points, mesh.points)

    assert len(mesh_ref.cells) == len(mesh.cells)
    for ic, cell in enumerate(mesh_ref.cells):
        assert cell.type == mesh.cells[ic].type
        assert numpy.allclose(cell.data, mesh.cells[ic].data)

    assert sorted(mesh_ref.cell_sets.keys()) == sorted(mesh.cell_sets.keys())
    for k, v in mesh_ref.cell_sets.items():
        for ic in range(len(mesh_ref.cells)):
            assert numpy.allclose(v[ic], mesh.cell_sets[k][ic])
