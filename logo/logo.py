import numpy as np
import optimesh
import pygmsh

import meshio

# def _old_logo()
#     with pygmsh.occ.Geometry() as geom:
#         characteristic_length_min = 0.5
#         characteristic_length_max = 0.5
#
#         container = geom.add_rectangle([0.0, 0.0, 0.0], 10.0, 10.0)
#
#         letter_i = geom.add_rectangle([2.0, 2.0, 0.0], 1.0, 4.5)
#         i_dot = geom.add_disk([2.5, 7.5, 0.0], 0.6)
#
#         disk1 = geom.add_disk([6.25, 4.5, 0.0], 2.5)
#         disk2 = geom.add_disk([6.25, 4.5, 0.0], 1.5)
#         letter_o = geom.boolean_difference([disk1], [disk2])
#
#         geom.boolean_difference([container], [letter_i, i_dot, letter_o])
#
#         mesh = pygmsh.generate_mesh(geom)
#
#     X, cells = mesh.points, mesh.cells
#     X, cells = optimesh.cvt.lloyd.quasi_newton_uniform_lloyd(
#         X, cells["triangle"], 1.0e-3, 1000
#     )
#     return X, cells


def create_logo2(y=0.0):
    with pygmsh.geo.Geometry() as geom:
        mesh_size = 0.15

        arrow1 = geom.add_polygon(
            [
                [0.10, 0.70 - y, 0.0],
                [0.35, 0.60 - y, 0.0],
                [0.35, 0.65 - y, 0.0],
                [0.80, 0.65 - y, 0.0],
                [0.80, 0.75 - y, 0.0],
                [0.35, 0.75 - y, 0.0],
                [0.35, 0.80 - y, 0.0],
            ],
            mesh_size=mesh_size,
            make_surface=False,
        )

        arrow2 = geom.add_polygon(
            [
                [0.90, 0.30 + y, 0.0],
                [0.65, 0.40 + y, 0.0],
                [0.65, 0.35 + y, 0.0],
                [0.20, 0.35 + y, 0.0],
                [0.20, 0.25 + y, 0.0],
                [0.65, 0.25 + y, 0.0],
                [0.65, 0.20 + y, 0.0],
            ],
            mesh_size=mesh_size,
            make_surface=False,
        )

        geom.add_polygon(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            mesh_size=mesh_size,
            holes=[arrow1, arrow2],
        )

        mesh = geom.generate_mesh()
    # return mesh.points, mesh.cells["triangle"]
    X = mesh.points
    cells = mesh.get_cells_type("triangle")

    # np.bincount doesn't work with uint
    # <https://github.com/numpy/numpy/issues/17760>
    cells = cells.astype(int)

    X, cells = optimesh.cvt.quasi_newton_uniform_full(
        X, cells, 1.0e-10, 100, verbose=True
    )
    return X, cells


if __name__ == "__main__":
    X, cells = create_logo2(y=0.08)

    mesh = meshio.Mesh(X, {"triangle": cells})
    meshio.svg.write("logo.svg", mesh, image_width=300)

    X = np.column_stack([X[:, 0], X[:, 1], np.zeros(X.shape[0])])
    meshio.Mesh(X, {"triangle": cells}).write("logo.vtk")
