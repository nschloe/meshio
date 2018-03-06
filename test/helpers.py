# -*- coding: utf-8 -*-
#
import os
import string
import tempfile

import numpy

# In general:
# Use values with an infinite decimal representation to test precision.

tri_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]) / 3,
        'cells': {
            'triangle': numpy.array([
                [0, 1, 2],
                [0, 2, 3]
                ])
            },
        }

triangle6_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.25, 0.0],
            [1.25, 0.5, 0.0],
            [0.25, 0.75, 0.0],
            [2.0, 1.0, 0.0],
            [1.5, 1.25, 0.0],
            [1.75, 0.25, 0.0],
            ]) / 3.0,
        'cells': {
            'triangle6': numpy.array([
                [0, 1, 2, 3, 4, 5],
                [1, 6, 2, 8, 7, 4]
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
            [0.0, 1.0, 0.0],
            ]) / 3.0,
        'cells': {
            'quad': numpy.array([
                [0, 1, 4, 5],
                [1, 2, 3, 4],
                ])
            },
        }

d = 0.1
quad8_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, d, 0.0],
            [1-d, 0.5, 0.0],
            [0.5, 1-d, 0.0],
            [d, 0.5, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.5, -d, 0.0],
            [2+d, 0.5, 0.0],
            [1.5, 1+d, 0.0],
            ]) / 3.0,
        'cells': {
            'quad8': numpy.array([
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 8, 9, 2, 10, 11, 12, 5],
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

tet10_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
            #
            [0.5, 0.0, 0.1],
            [1.0, 0.5, 0.1],
            [0.5, 0.5, 0.1],
            [0.25, 0.3, 0.25],
            [0.8, 0.25, 0.25],
            [0.7, 0.7, 0.3],
            ]) / 3.0,
        'cells': {
            'tetra10': numpy.array([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                ])
            },
        }

hex_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            ]),
        'cells': {
            'hexahedron': numpy.array([
                [0, 1, 2, 3, 4, 5, 6, 7],
                ])
            },
        }

hex20_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            #
            [0.5, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            #
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
            #
            [0.5, 0.0, 1.0],
            [1.0, 0.5, 1.0],
            [0.5, 1.0, 1.0],
            [0.0, 0.5, 1.0],
            ]),
        'cells': {
            'hexahedron20': numpy.array([range(20)])
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


def add_point_data(mesh, dim, num_tags=2):
    numpy.random.seed(0)
    mesh2 = _clone(mesh)

    if dim == 1:
        data = [
            numpy.random.rand(len(mesh['points']))
            for _ in range(num_tags)
            ]
    else:
        data = [
            numpy.random.rand(len(mesh['points']), dim)
            for _ in range(num_tags)
            ]

    mesh2['point_data'] = {
            string.ascii_lowercase[k]: d
            for k, d in enumerate(data)
            }
    return mesh2


def add_cell_data(mesh, dim, num_tags=2):
    mesh2 = _clone(mesh)
    numpy.random.seed(0)
    cell_data = {}
    for cell_type in mesh['cells']:
        num_cells = len(mesh['cells'][cell_type])
        if dim == 1:
            cell_data[cell_type] = {
                string.ascii_lowercase[k]: numpy.random.rand(num_cells)
                for k in range(num_tags)
                }
        else:
            cell_data[cell_type] = {
                string.ascii_lowercase[k]: numpy.random.rand(num_cells, dim)
                for k in range(num_tags)
                }

    mesh2['cell_data'] = cell_data
    return mesh2


def add_field_data(mesh, value, dtype):
    mesh2 = _clone(mesh)
    field_data = {
        'a': numpy.array(value, dtype=dtype),
    }
    mesh2['field_data'] = field_data
    return mesh2


def write_read(writer, reader, mesh, atol):
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

    try:
        input_field_data = mesh['field_data']
    except KeyError:
        input_field_data = {}

    handle, filename = tempfile.mkstemp()
    os.close(handle)

    writer(
        filename,
        mesh['points'], mesh['cells'],
        point_data=input_point_data,
        cell_data=input_cell_data,
        field_data=input_field_data,
        )

    points, cells, point_data, cell_data, field_data = reader(filename)

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

    for name, data in input_field_data.items():
        assert numpy.allclose(
            data, field_data[name],
            atol=atol, rtol=0.0
            )

    os.remove(filename)
    return
