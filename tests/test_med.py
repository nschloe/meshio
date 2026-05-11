import pathlib

import numpy as np
import pytest

import meshio
from meshio.med._med import numpy_to_med_type
from . import helpers

h5py = pytest.importorskip("h5py")


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh_2d,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.quad_tri_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.add_point_data(helpers.tri_mesh, 1),
        helpers.add_point_data(helpers.tri_mesh, 2),
        helpers.add_point_data(helpers.tri_mesh, 3),
        helpers.add_point_data(helpers.hex_mesh, 3),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (2,), np.float64)]),
        helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.med")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.med")


def test_reference_file_with_mixed_cells(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "cylinder.med"
    mesh = meshio.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 16.53169892762988)

    # CellBlock
    ref_num_cells = {"pyramid": 18, "quad": 18, "line": 17, "tetra": 63, "triangle": 4}
    assert {
        cell_block.type: len(cell_block) for cell_block in mesh.cells
    } == ref_num_cells

    # Point tags
    assert mesh.point_data["point_tags"].sum() == 52
    ref_point_tags_info = {2: ["Side"], 3: ["Side", "Top"], 4: ["Top"]}
    assert mesh.point_tags == ref_point_tags_info

    # Cell tags
    ref_sum_cell_tags = {
        "pyramid": -116,
        "quad": -75,
        "line": -48,
        "tetra": -24,
        "triangle": -30,
    }
    assert {
        c.type: sum(d) for c, d in zip(mesh.cells, mesh.cell_data["cell_tags"])
    } == ref_sum_cell_tags
    ref_cell_tags_info = {
        -6: ["Top circle"],
        -7: ["Top", "Top and down"],
        -8: ["Top and down"],
        -9: ["A", "B"],
        -10: ["B"],
        -11: ["B", "C"],
        -12: ["C"],
    }
    assert mesh.cell_tags == ref_cell_tags_info

    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)


def test_reference_file_with_point_cell_data(tmp_path):
    this_dir = pathlib.Path(__file__).resolve().parent
    filename = this_dir / "meshes" / "med" / "box.med"

    mesh = meshio.read(filename)

    # Points
    assert np.isclose(mesh.points.sum(), 12)

    # CellBlock
    assert {cell_block.type: len(cell_block) for cell_block in mesh.cells} == {
        "hexahedron": 1
    }

    # Point data
    data_u = mesh.point_data["resu____DEPL"]
    assert data_u.shape == (8, 3)
    assert np.isclose(data_u.sum(), 12)

    # Cell data
    # ELNO (1 data point for every node of each element)
    data_eps = mesh.cell_data["resu____EPSI_ELNO"][0]
    assert data_eps.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_eps_mean = np.mean(data_eps, axis=1)[0]
    eps_ref = np.array([1, 0, 0, 0.5, 0.5, 0])
    assert np.allclose(data_eps_mean, eps_ref)

    data_sig = mesh.cell_data["resu____SIEF_ELNO"][0]
    assert data_sig.shape == (1, 8, 6)  # (n_cells, n_nodes_per_element, n_components)
    data_sig_mean = np.mean(data_sig, axis=1)[0]
    sig_ref = np.array(
        [7328.44611253, 2645.87030114, 2034.06063679, 1202.6, 569.752, 0]
    )
    assert np.allclose(data_sig_mean, sig_ref)

    data_psi = mesh.cell_data["resu____ENEL_ELNO"][0]
    assert data_psi.shape == (1, 8, 1)  # (n_cells, n_nodes_per_element, n_components)

    # ELEM (1 data point for each element)
    data_psi_elem = mesh.cell_data["resu____ENEL_ELEM"][0]
    assert np.isclose(np.mean(data_psi, axis=1)[0, 0], data_psi_elem[0])

    helpers.write_read(tmp_path, meshio.med.write, meshio.med.read, mesh, 1.0e-15)

@pytest.mark.parametrize("dtype, expected_med_type", [
    (np.float32, 4),   # MED_FLOAT32
    (np.float64, 6),   # MED_FLOAT64
    (np.int32,   24),  # MED_INT32
    (np.int64,   26),  # MED_INT64
])
def test_med_type_mapping(dtype, expected_med_type):
    """Vérifie que le mapping numpy dtype -> constante MED est correct."""
    data = np.array([1, 2, 3], dtype=dtype)
    result = numpy_to_med_type[data.dtype]
    assert result == expected_med_type, (
        f"dtype={dtype.__name__}: "
        f"attendu {expected_med_type}, obtenu {result}"
    )


def test_med_type_mapping_unknown_dtype():
    """Un dtype non supporté doit lever une KeyError."""
    data = np.array([1, 2, 3], dtype=np.complex128)
    with pytest.raises(KeyError):
        _ = numpy_to_med_type[data.dtype]


@pytest.mark.parametrize("dtype, expected_med_type", [
    (np.float32, 4),   # MED_FLOAT32
    (np.float64, 6),   # MED_FLOAT64
    (np.int32,   24),  # MED_INT32
    (np.int64,   26),  # MED_INT64
])
def test_med_type_preserved_after_write_read(tmp_path, dtype, expected_med_type):
    """
    Vérifie que le TYP écrit dans le HDF5 correspond
    bien au type MED attendu après un write meshio.
    """
    filename = tmp_path / f"test_roundtrip_{dtype.__name__}.med"

    mesh = helpers.add_point_data(helpers.tri_mesh, 1)
    for key in mesh.point_data:
        mesh.point_data[key] = mesh.point_data[key].astype(dtype)

    meshio.med.write(filename, mesh)

    with h5py.File(filename, "r") as f:
        if "CHA" in f:
            for field_name in f["CHA"]:
                written_type = f["CHA"][field_name].attrs.get("TYP")
                if written_type is not None:
                    assert written_type == expected_med_type, (
                        f"Champ '{field_name}', dtype={dtype.__name__}: "
                        f"TYP={written_type}, attendu={expected_med_type}"
                    )
