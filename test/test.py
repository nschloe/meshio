# -*- coding: utf-8 -*-
#
import meshio

import copy
import numpy

simple = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]),
        'cells': {
            'triangle': numpy.array([
                [0, 1, 2],
                [0, 2, 3]
                ])
            },
        'point_data': {},
        'cell_data': {}
        }

simple_data = copy.deepcopy(simple)
simple_data['point_data'] = {
        'a': numpy.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
            ])}
simple_data['cell_data'] = {
            'b': numpy.array([3.14, 2.718])
            }

tri_quad = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]),
        'cells':  {
            'triangle': numpy.array([
                [0, 1, 4],
                [0, 4, 5]
                ]),
            'quad': numpy.array([
                [1, 2, 3, 4]
                ])
            },
        'point_data': {},
        'cell_data': {}
        }

tri_quad_data = copy.deepcopy(tri_quad)
tri_quad_data['point_data'] = {
        'a': numpy.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            ])
        }
tri_quad_data['cell_data'] = {
        'b': numpy.array([3.14, 2.718, 1.414])
        }


def test_io():
    for extension in ['.vtk', '.vtu']:
        filename = 'test' + extension
        for mesh in [simple_data, tri_quad_data]:
            yield _write_read, filename, mesh

    for extension in ['.e']:
        filename = 'test' + extension
        for mesh in [simple_data]:
            yield _write_read, filename, mesh

    for extension in ['.msh']:
        filename = 'test' + extension
        for mesh in [simple, tri_quad]:
            yield _write_read, filename, mesh

    for extension in ['.dato', '.h5m']:
        filename = 'test' + extension
        for mesh in [simple]:
            yield _write_read, filename, mesh

    return


def _write_read(filename, mesh):
    '''Write and read a file, and make sure the data is the same as before.
    '''
    meshio.write(
        filename,
        mesh['points'], mesh['cells'],
        point_data=mesh['point_data'],
        cell_data=mesh['cell_data']
        )
    points, cells, point_data, cell_data, _ = meshio.read(filename)

    # We cannot compare the exact rows here since the order of the points might
    # have changes. Just compare the sums
    assert numpy.allclose(mesh['points'], points)

    for key, data in mesh['cells'].items():
        assert numpy.array_equal(data, cells[key])
    for key, data in mesh['point_data'].items():
        assert numpy.array_equal(data, point_data[key])
    for key, data in mesh['cell_data'].items():
        assert numpy.array_equal(data, cell_data[key])
    return

if __name__ == '__main__':
    test_io()
