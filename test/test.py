# -*- coding: utf-8 -*-
#
import meshio

import numpy


def test_io():
    # create a dummy mesh
    points = numpy.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
        ])
    cells = numpy.array([
        [0, 1, 2],
        [0, 2, 3]
        ])

    for extension in ['.e', '.vtk', '.vtu', '.h5m']:
        filename = 'test' + extension
        yield _write_read, filename, points, cells

    return


def _write_read(filename, points, cells):
    '''Write and read a file, and make sure the data is the same as before.
    '''
    meshio.write(filename, points, cells)
    p, c, _, _, _ = meshio.read(filename)

    # We cannot compare the exact rows here since the order of the points might
    # have changes. Just compare the sums
    assert numpy.allclose(points, p)
    assert numpy.array_equal(cells, c)
    return

if __name__ == '__main__':
    test_io()
