# -*- coding: utf-8 -*-
#
import meshio

import numpy


def test_io():
    # create a dummy mesh
    points = numpy.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
        ])
    cells = {
        'triangle': numpy.array([
            [0, 1, 4],
            [0, 4, 5]
            ]),
        'quad': numpy.array([
            [1, 2, 3, 4]
            ])
            }

    point_data = {
        'a': numpy.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            ])
        }

    cell_data = {'b': numpy.array([3.14, 2.718, 1.414])}

    #for extension in ['.e', '.vtk', '.vtu', '.h5m']:
    for extension in ['.e', '.vtk', '.vtu']:
        filename = 'test' + extension
        yield _write_read, filename, points, cells, point_data, cell_data

    #for extension in ['.dato', '.msh']:
    #    filename = 'test' + extension
    #    yield _write_read, filename, points, cells

    return


def _write_read(filename, points, cells, point_data={}, cell_data={}):
    '''Write and read a file, and make sure the data is the same as before.
    '''
    meshio.write(
        filename,
        points, cells,
        point_data=point_data,
        cell_data=cell_data
        )
    p, c, pd, cd, _ = meshio.read(filename)

    # We cannot compare the exact rows here since the order of the points might
    # have changes. Just compare the sums
    assert numpy.allclose(points, p)

    for key, data in cells.items():
        assert numpy.array_equal(data, c[key])
    for key, data in point_data.items():
        assert numpy.array_equal(data, pd[key])
    for key, data in cell_data.items():
        assert numpy.array_equal(data, cd[key])
    return

if __name__ == '__main__':
    test_io()
