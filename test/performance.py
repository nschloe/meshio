# -*- coding: utf-8 -*-
#
import time

import meshio


def generate_mesh():
    '''Generates a fairly large mesh.
    '''
    # pylint: disable=import-error
    import pygmsh
    geom = pygmsh.built_in.Geometry()

    geom.add_circle(
        [0.0, 0.0, 0.0],
        1.0,
        # 5.0e-3,
        1.0e-2,
        num_sections=4,
        # If compound==False, the section borders have to be points of the
        # discretization. If using a compound circle, they don't; gmsh can
        # choose by itself where to point the circle points.
        compound=True
        )
    X, cells, _, _, _ = pygmsh.generate_mesh(geom)
    return X, cells


def read_write():
    X, cells = generate_mesh()

    formats = [
        'ansys-ascii',
        'ansys-binary',
        'exodus',
        'dolfin-xml',
        'gmsh-ascii',
        'gmsh-binary',
        'med',
        'medit',
        'permas',
        'moab',
        'off',
        'stl-ascii',
        'stl-binary',
        'vtk-ascii',
        'vtk-binary',
        'vtu-ascii',
        'vtu-binary',
        'xdmf',
        ]

    filename = 'foo'
    print()
    print('format        write (s)    read(s)')
    print()
    for fmt in formats:
        t = time.time()
        meshio.write(filename, X, cells, file_format=fmt)
        elapsed_write = time.time() - t

        t = time.time()
        meshio.read(filename, file_format=fmt)
        elapsed_read = time.time() - t
        print('{0: <12}  {1:e} {2:e}'.format(fmt, elapsed_write, elapsed_read))

    return


if __name__ == '__main__':
    read_write()
