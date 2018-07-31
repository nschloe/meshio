# -*- coding: utf-8 -*-
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

    X, cells, _, _, _ = pygmsh.generate_mesh(geom)
    X, cells = optimesh.lloyd(X, cells["triangle"], 1.0e-3, 1000)
    return X, cells


if __name__ == "__main__":
    import meshio
    X, cells = create_logo()
    meshio.write_points_cells('logo.svg', X, {"triangle": cells})
    # import numpy
    # X = numpy.column_stack([X[:, 0], X[:, 1], numpy.zeros(X.shape[0])])
    # meshio.write_points_cells('logo.vtk', X, {"triangle": cells})
