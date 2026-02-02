import pytest
import numpy as np

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


# --- tests for i/o updates allowing for field data, time inference, and timeseries --- #
def test_single_step_roundtrip(tmp_path):
    # basic mesh
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    cells = [("triangle", np.array([[0, 1, 2]]))]
    point_data = {"temp": np.array([100.0, 150.0, 200.0])}
    point_sets = {"boundary": np.array([0, 1])}
    
    mesh_out = meshio.Mesh(points, cells, point_data=point_data, point_sets=point_sets)
    
    # test writing
    mesh_out.write(tmp_path / "test.exo", time_whole=5.0)
    
    # test reading
    mesh_in = meshio.read(tmp_path / "test.exo", timestep=0)
    
    # verification that nothing important changed
    assert np.allclose(mesh_in.points, points)
    assert mesh_in.field_data["time"] == 5.0
    assert "temp" in mesh_in.point_data
    assert "boundary" in mesh_in.point_sets

def test_transient_multiblock(tmp_path):
    # TODO: what if they aren't np.array
    # static mesh with 5 points, 1 quad, 1 triangle
    points = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0.5,2,0]])
    cells = [
        ("quad", np.array([[0, 1, 2, 3]])),
        ("triangle", np.array([[3, 2, 4]]))
    ]
    
    # timestep 0 data (point & cell)
    pd0 = {"vel": np.zeros((5, 3))}
    cd0 = {"press": [np.array([10.0]), np.array([20.0])]} # One array per block
    
    # timestep 1 data (point & cell)
    pd1 = {"vel": np.ones((5, 3))}
    cd1 = {"press": [np.array([15.0]), np.array([25.0])]}
    
    # write
    mesh_out = meshio.Mesh(points, cells)
    mesh_out.write(
        tmp_path / "test.exo",
        point_data=[pd0, pd1],
        cell_data=[cd0, cd1],
        time_step_size=0.5
    )
    
    # read all timesteps
    meshes = meshio.read(tmp_path / "test.exo", timestep="all")
    
    # verify that the result isn't changed (aside from being split into 2 meshes)
    assert len(meshes) == 2
    assert meshes[1].field_data["time"] == 0.5
    assert meshes[1].cell_data["press"][1] == 25.0

def test_time_inference(tmp_path):
    points = np.array([[0,0,0], [1,0,0], [0,1,0]])
    cells = [("triangle", np.array([[0, 1, 2]]))]
    
    # define global "time" in field data
    fd = [{"time": 0.0, "energy": 1.0}, {"time": 0.25, "energy": 0.8}]
    
    mesh_out = meshio.Mesh(points, cells)
    mesh_out.write(tmp_path / "test.exo", field_data=fd)
    
    # read back all timesteps and check if field_data became the time axis
    mesh_in = meshio.read(tmp_path / "test.exo", timestep="all")
    
    assert mesh_in[1].field_data["time"] == 0.25
    assert mesh_in[1].field_data["energy"] == 0.8

def test_comprehensive(tmp_path):
    # static geometry: 4 points, 2 triangles
    points = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]], dtype=float)
    cells = [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]

    # transient data (point & field): 2 timesteps
    pd = [
        {"temp": np.array([10., 10., 10., 10.])},
        {"temp": np.array([20., 25., 20., 25.])}
    ]
    fd = [
        {"energy": 0.5, "time": 0.0, "status": 1.0},
        {"energy": 0.7, "time": 1.0, "status": 1.0}
    ]

    # static data (cell)
    cd = {"material_id": [np.array([1, 1])]}

    # manually specified side sets (left and right boundaries)
    manual_ss = {
        "left_wall": ([0], [2]),
        "right_wall": ([1], [1])
    }

    # write out
    mesh_out = meshio.Mesh(points, cells)
    mesh_out.write(
        tmp_path / "test.exo",
        point_data=pd,
        cell_data=cd,
        field_data=fd,
        side_sets=manual_ss
    )

    # read back
    meshes = meshio.read(tmp_path / "test.exo", timestep="all")

    # verify key characteristics stayed the same
    assert len(meshes) == 2
    assert meshes[1].field_data["energy"] == 0.7
    assert meshes[1].field_data["time"] == 1.0
    assert "material_id" in meshes[1].cell_data
    assert meshes[1].cell_data["material_id"][0][0] == 1
