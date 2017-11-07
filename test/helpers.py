# -*- coding: utf-8 -*-
#
import os

import meshio
import numpy

# In general:
# Use values with an infinite decimal representation to test precision.

tri_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]) / 3.0,
        'cells': {
            'triangle': numpy.array([
                [0, 1, 2],
                [0, 2, 3]
                ])
            },
        }

quad_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]) / 3.0,
        'cells': {
            'quad': numpy.array([
                [0, 1, 4, 5],
                [1, 2, 3, 4]
                ])
            },
        }

tri_quad_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]) / 3.0,
        'cells':  {
            'triangle': numpy.array([
                [0, 1, 4],
                [0, 4, 5]
                ]),
            'quad': numpy.array([
                [1, 2, 3, 4]
                ])
            }
        }

tet_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            ]) / 3.0,
        'cells': {
            'tetra': numpy.array([
                [0, 1, 2, 4],
                [0, 2, 3, 4]
                ])
            },
        }


hex_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            ]),
        'cells': {
            'hexahedron': numpy.array([
                [0, 1, 2, 4, 5, 6, 7, 8],
                ])
            },
        }


def _clone(mesh):
    mesh2 = {
        'points': numpy.copy(mesh['points'])
        }
    mesh2['cells'] = {}
    for key, data in mesh['cells'].items():
        mesh2['cells'][key] = numpy.copy(data)
    return mesh2


def add_point_data(mesh, dim):
    numpy.random.seed(0)
    mesh2 = _clone(mesh)

    if dim == 1:
        data = numpy.random.rand(len(mesh['points']))
    else:
        data = numpy.random.rand(len(mesh['points']), dim)

    mesh2['point_data'] = {'a': data}
    return mesh2


def add_cell_data(mesh, dim):
    mesh2 = _clone(mesh)
    numpy.random.seed(0)
    cell_data = {}
    for cell_type in mesh['cells']:
        num_cells = len(mesh['cells'][cell_type])
        if dim == 1:
            cell_data[cell_type] = {
                'b': numpy.random.rand(num_cells)
                }
        else:
            cell_data[cell_type] = {
                'b': numpy.random.rand(num_cells, dim)
                }

    mesh2['cell_data'] = cell_data
    return mesh2


def write_read(filename, file_format, mesh, atol):
    '''Write and read a file, and make sure the data is the same as before.
    '''
    try:
        input_point_data = mesh['point_data']
    except KeyError:
        input_point_data = {}

    try:
        input_cell_data = mesh['cell_data']
    except KeyError:
        input_cell_data = {}

    meshio.write(
        filename,
        mesh['points'], mesh['cells'],
        file_format=file_format,
        point_data=input_point_data,
        cell_data=input_cell_data
        )
    points, cells, point_data, cell_data, _ = \
        meshio.read(filename, file_format)

    # Numpy's array_equal is too strict here, cf.
    # <https://mail.scipy.org/pipermail/numpy-discussion/2015-December/074410.html>.
    # Use allclose.

    # We cannot compare the exact rows here since the order of the points might
    # have changes. Just compare the sums
    assert numpy.allclose(mesh['points'], points, atol=atol, rtol=0.0)

    for cell_type, data in mesh['cells'].items():
        assert numpy.allclose(data, cells[cell_type])
    for key in input_point_data.keys():
        assert numpy.allclose(
            input_point_data[key], point_data[key],
            atol=atol, rtol=0.0
            )
    for cell_type, cell_type_data in input_cell_data.items():
        for key, data in cell_type_data.items():
            assert numpy.allclose(
                    data, cell_data[cell_type][key],
                    atol=atol, rtol=0.0
                    )

    os.remove(filename)
    return
