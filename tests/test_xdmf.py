import numpy as np
import pytest

import meshio

from . import helpers

test_set_full = [
    helpers.empty_mesh,
    helpers.line_mesh,
    helpers.tri_mesh,
    helpers.line_tri_mesh,
    helpers.tri_mesh_2d,
    helpers.triangle6_mesh,
    helpers.quad_mesh,
    helpers.quad8_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.tet10_mesh,
    helpers.hex_mesh,
    helpers.hex20_mesh,
    helpers.pyramid_mesh,
    helpers.wedge_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
]

test_set_reduced = [
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.quad_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.hex_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
]


@pytest.mark.parametrize("mesh", test_set_full)
@pytest.mark.parametrize(
    "kwargs0",
    [
        {"data_format": "XML"},
        {"data_format": "Binary"},
        {"data_format": "HDF", "compression": None},
        {"data_format": "HDF", "compression": "gzip"},
    ],
)
def test_xdmf3(mesh, kwargs0, tmp_path):
    def write(*args, **kwargs):
        return meshio.xdmf.write(*args, **{**kwargs0, **kwargs})

    helpers.write_read(tmp_path, write, meshio.xdmf.read, mesh, 1.0e-14)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.xdmf")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.xdmf")


def test_time_series():
    # write the data
    filename = "out.xdmf"

    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(helpers.tri_mesh_2d.points, helpers.tri_mesh_2d.cells)
        n = helpers.tri_mesh_2d.points.shape[0]

        times = np.linspace(0.0, 1.0, 5)
        point_data = [
            {
                "phi": np.full(n, t),
                "u": np.full(helpers.tri_mesh_2d.points.shape, t),
            }
            for t in times
        ]
        for t, pd in zip(times, point_data):
            writer.write_data(
                t, point_data=pd, cell_data={"a": {"triangle": [3.0, 4.2]}}
            )

    # read it back in
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, pd, cd = reader.read_data(k)
            assert np.abs(times[k] - t) < 1.0e-12
            for key, value in pd.items():
                assert np.all(np.abs(value - point_data[k][key]) < 1.0e-12)


# def test_information_xdmf():
#     mesh_out = meshio.Mesh(
#         np.array(
#             [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
#         )
#         / 3,
#         [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
#         field_data={
#             "bottom": np.array([1, 1]),
#             "right": np.array([2, 1]),
#             "top": np.array([3, 1]),
#             "left": np.array([4, 1]),
#         },
#     )
#     # write the data
#     points, cells, field_data = mesh_out.points, mesh_out.cells, mesh_out.field_data
#
#     assert cells[0].type == "triangle"
#     meshio.write(
#         "mesh.xdmf",
#         meshio.Mesh(points=points, cells=[cells[0]], field_data=field_data),
#     )
#
#     # read it back in
#     mesh_in = meshio.read("mesh.xdmf")
#     assert len(mesh_in.field_data) == len(mesh_out.field_data)


if __name__ == "__main__":
    test_time_series()
