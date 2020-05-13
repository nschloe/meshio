import numpy
import pytest

import helpers
import meshio

test_set_full = [
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
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), numpy.float64)]),
]

test_set_reduced = [
    helpers.tri_mesh,
    helpers.tri_mesh_2d,
    helpers.quad_mesh,
    helpers.tri_quad_mesh,
    helpers.tet_mesh,
    helpers.hex_mesh,
    helpers.add_point_data(helpers.tri_mesh, 1),
    helpers.add_cell_data(helpers.tri_mesh, [("a", (), numpy.float64)]),
]


@pytest.mark.parametrize("mesh", test_set_full)
@pytest.mark.parametrize("data_format", ["XML", "Binary", "HDF"])
def test_xdmf3(mesh, data_format):
    def write(*args, **kwargs):
        return meshio.xdmf.write(*args, data_format=data_format, **kwargs)

    helpers.write_read(write, meshio.xdmf.read, mesh, 1.0e-14)


# HDF5 compressed I/O
@pytest.mark.parametrize("mesh", test_set_full)
def test_compression(mesh):
    def write(*args, **kwargs):
        return meshio.xdmf.write(*args, data_format="HDF", compression="gzip", **kwargs)

    helpers.write_read(write, meshio.xdmf.read, mesh, 1.0e-14)


def test_generic_io():
    helpers.generic_io("test.xdmf")
    # With additional, insignificant suffix:
    helpers.generic_io("test.0.xdmf")


def test_time_series():
    # write the data
    filename = "out.xdmf"

    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(helpers.tri_mesh_2d.points, helpers.tri_mesh_2d.cells)
        n = helpers.tri_mesh_2d.points.shape[0]

        times = numpy.linspace(0.0, 1.0, 5)
        point_data = [
            {
                "phi": numpy.full(n, t),
                "u": numpy.full(helpers.tri_mesh_2d.points.shape, t),
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
            assert numpy.abs(times[k] - t) < 1.0e-12
            for key, value in pd.items():
                assert numpy.all(numpy.abs(value - point_data[k][key]) < 1.0e-12)


# def test_information_xdmf():
#     mesh_out = meshio.Mesh(
#         numpy.array(
#             [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
#         )
#         / 3,
#         [("triangle", numpy.array([[0, 1, 2], [0, 2, 3]]))],
#         field_data={
#             "bottom": numpy.array([1, 1]),
#             "right": numpy.array([2, 1]),
#             "top": numpy.array([3, 1]),
#             "left": numpy.array([4, 1]),
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
