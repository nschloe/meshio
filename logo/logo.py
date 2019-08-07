import numpy

import meshio
import optimesh
import pygmsh


def create_logo():

    geom = pygmsh.opencascade.Geometry(
        characteristic_length_min=0.5, characteristic_length_max=0.5
    )

    container = geom.add_rectangle([0.0, 0.0, 0.0], 10.0, 10.0)

    letter_i = geom.add_rectangle([2.0, 2.0, 0.0], 1.0, 4.5)
    i_dot = geom.add_disk([2.5, 7.5, 0.0], 0.6)

    disk1 = geom.add_disk([6.25, 4.5, 0.0], 2.5)
    disk2 = geom.add_disk([6.25, 4.5, 0.0], 1.5)
    letter_o = geom.boolean_difference([disk1], [disk2])

    geom.boolean_difference([container], [letter_i, i_dot, letter_o])

    mesh = pygmsh.generate_mesh(geom)
    X, cells = mesh.points, mesh.cells
    X, cells = optimesh.cvt.lloyd.quasi_newton_uniform_lloyd(
        X, cells["triangle"], 1.0e-3, 1000
    )
    return X, cells


def create_logo2():
    geom = pygmsh.built_in.Geometry()

    lcar = 0.1

    arrow1 = geom.add_polygon(
        [
            [0.10, 0.7, 0.0],
            [0.35, 0.5, 0.0],
            [0.35, 0.6, 0.0],
            [0.8, 0.6, 0.0],
            [0.8, 0.8, 0.0],
            [0.35, 0.8, 0.0],
            [0.35, 0.9, 0.0],
        ],
        lcar=lcar,
        make_surface=False,
    )

    arrow2 = geom.add_polygon(
        [
            [0.89, 0.3, 0.0],
            [0.65, 0.5, 0.0],
            [0.65, 0.4, 0.0],
            [0.20, 0.4, 0.0],
            [0.20, 0.2, 0.0],
            [0.65, 0.2, 0.0],
            [0.65, 0.1, 0.0],
        ],
        lcar=lcar,
        make_surface=False,
    )

    geom.add_polygon(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        lcar=lcar,
        holes=[arrow1, arrow2],
    )

    mesh = pygmsh.generate_mesh(geom)
    # return mesh.points, mesh.cells["triangle"]

    X, cells = optimesh.cvt.quasi_newton_uniform_full(
        mesh.points, mesh.cells["triangle"], 1.0e-10, 100
    )
    return X, cells


if __name__ == "__main__":
    X, cells = create_logo2()

    meshio.write_points_cells("logo.svg", X, {"triangle": cells})

    X = numpy.column_stack([X[:, 0], X[:, 1], numpy.zeros(X.shape[0])])
    meshio.write_points_cells("logo.vtk", X, {"triangle": cells})
