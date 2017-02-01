# -*- coding: utf-8 -*-
#
import meshio

import numpy
import os
import pytest

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


def _clone(mesh):
    mesh2 = {
        'points': numpy.copy(mesh['points'])
        }
    mesh2['cells'] = {}
    for key, data in mesh['cells'].items():
        mesh2['cells'][key] = numpy.copy(data)
    return mesh2


def _add_point_data(mesh, dim):
    numpy.random.seed(0)
    mesh2 = _clone(mesh)

    # Don't use default float64 here because of a bug in VTK that prevents
    # actual double data from being written out, cf.
    # <http://www.vtk.org/Bug/view.php?id=15889>.
    if dim == 1:
        data = numpy.random.rand(len(mesh['points'])).astype('float32')
    else:
        data = numpy.random.rand(len(mesh['points']), dim).astype('float32')

    mesh2['point_data'] = {'a': data}
    return mesh2


def _add_cell_data(mesh, dim):
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


@pytest.mark.parametrize('extension, meshes', [
    ('dato', [tri_mesh, quad_mesh, tri_quad_mesh, tet_mesh]),
    # ('e', [
    #     tri_mesh,
    #     _add_point_data(tri_mesh, 2),
    #     _add_point_data(tri_mesh, 3)
    #     ]),
    ('h5m', [tri_mesh, tet_mesh]),
    ('msh', [tri_mesh, quad_mesh, tri_quad_mesh, tet_mesh]),
    ('mesh', [tri_mesh, quad_mesh, tri_quad_mesh, tet_mesh]),
    ('off', [tri_mesh]),
    # These tests needed to be disabled since travis-ci only offers trusty with
    # the buggy VTK 6.0.
    # TODO enable once we can use a more recent version of VTK
    # ('vtk', [
    #      tri_mesh,
    #      quad_mesh,
    #      tri_quad_mesh,
    #      tet_mesh,
    #      _add_point_data(tri_mesh, 1),
    #      _add_point_data(tri_mesh, 2),
    #      _add_point_data(tri_mesh, 3),
    #      _add_cell_data(tri_mesh, 1),
    #      _add_cell_data(tri_mesh, 2),
    #      _add_cell_data(tri_mesh, 3),
    #      ]),
    # ('vtu', [
    #      tri_mesh,
    #      quad_mesh,
    #      tri_quad_mesh,
    #      tet_mesh,
    #      _add_point_data(tri_mesh, 1),
    #      _add_point_data(tri_mesh, 2),
    #      _add_point_data(tri_mesh, 3),
    #      _add_cell_data(tri_mesh, 1),
    #      _add_cell_data(tri_mesh, 2),
    #      _add_cell_data(tri_mesh, 3),
    #      ]),
    #
    # 2016-04-27: Temporarily disabled due to vtkXdmfWriter not being available
    #             through VTK
    # ('xdmf', [
    #     tri_mesh,
    #     quad_mesh,
    #     tet_mesh
    #     # The two following tests pass, but errors are emitted on the
    #     # console.
    #     # _add_point_data(tri_mesh, 1),
    #     # _add_cell_data(tri_mesh, 1)
    #     ]),
    ('xml', [tri_mesh, tet_mesh]),
    ])
def test_io(extension, meshes):
    filename = 'test.' + extension
    for mesh in meshes:
        _write_read(filename, mesh)


def _write_read(filename, mesh):
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
        point_data=input_point_data,
        cell_data=input_cell_data
        )
    points, cells, point_data, cell_data, _ = meshio.read(filename)

    # Numpy's array_equal is too strict here, cf.
    # <https://mail.scipy.org/pipermail/numpy-discussion/2015-December/074410.html>.
    # Use allclose.

    # We cannot compare the exact rows here since the order of the points might
    # have changes. Just compare the sums
    assert numpy.allclose(mesh['points'], points, atol=1.0e-15, rtol=0.0)

    for cell_type, data in mesh['cells'].items():
        assert numpy.allclose(data, cells[cell_type])
    for key, data in input_point_data.items():
        assert numpy.allclose(data, point_data[key], atol=1.0e-15, rtol=0.0)
    for cell_type, cell_type_data in input_cell_data.items():
        for key, data in cell_type_data.items():
            assert numpy.allclose(
                    data, cell_data[cell_type][key],
                    atol=1.0e-15, rtol=0.0
                    )

    os.remove(filename)

    return
