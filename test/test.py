# -*- coding: utf-8 -*-
#
import meshio

import numpy
import os

tri_mesh = {
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
        }

quad_mesh = {
        'points': numpy.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
            ]),
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
            ]),
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
            ]),
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
    for key, data in mesh['cells'].iteritems():
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
    num_cells = 0
    for _, cells in mesh2['cells'].iteritems():
        num_cells += len(cells)

    if dim == 1:
        data = numpy.random.rand(num_cells)
    else:
        data = numpy.random.rand(num_cells, dim)

    mesh2['cell_data'] = {'b': data}
    return mesh2


def test_io():

    for extension in ['.dato']:
        filename = 'test' + extension
        test_meshes = [
            tri_mesh,
            quad_mesh,
            tri_quad_mesh,
            tet_mesh,
            ]
        for mesh in test_meshes:
            yield _write_read, filename, mesh

    for extension in ['.e']:
        filename = 'test' + extension
        test_meshes = [
            tri_mesh,
            _add_point_data(tri_mesh, 2),
            _add_point_data(tri_mesh, 3),
            ]
        for mesh in test_meshes:
            yield _write_read, filename, mesh

    for extension in ['.h5m']:
        filename = 'test' + extension
        test_meshes = [
            tri_mesh,
            tet_mesh,
            ]
        for mesh in test_meshes:
            yield _write_read, filename, mesh

    for extension in ['.msh']:
        filename = 'test' + extension
        test_meshes = [
            tri_mesh,
            quad_mesh,
            tri_quad_mesh,
            tet_mesh,
            ]
        for mesh in test_meshes:
            yield _write_read, filename, mesh

    for extension in ['.vtk', '.vtu']:
        filename = 'test' + extension
        test_meshes = [
            tri_mesh,
            quad_mesh,
            tri_quad_mesh,
            tet_mesh,
            _add_point_data(tri_mesh, 1),
            _add_point_data(tri_mesh, 2),
            _add_point_data(tri_mesh, 3),
            _add_cell_data(tri_mesh, 1),
            _add_cell_data(tri_mesh, 2),
            _add_cell_data(tri_mesh, 3),
            ]
        for mesh in test_meshes:
            yield _write_read, filename, mesh

    # 2016-04-27: Temporarily disabled due to vtkXdmfWriter not being available
    #             through VTK
    if False:
        for extension in ['.xmf']:
            filename = 'test' + extension
            test_meshes = [
                tri_mesh,
                quad_mesh,
                tet_mesh,
                # The two following tests pass, but errors are emitted on the
                # console.
                # _add_point_data(tri_mesh, 1),
                # _add_cell_data(tri_mesh, 1)
                ]
            for mesh in test_meshes:
                yield _write_read, filename, mesh

    for extension in ['.xml']:
        filename = 'test' + extension
        test_meshes = [
            tri_mesh,
            tet_mesh,
            ]
        for mesh in test_meshes:
            yield _write_read, filename, mesh

    return


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
    assert numpy.allclose(mesh['points'], points)

    for key, data in mesh['cells'].items():
        assert numpy.allclose(data, cells[key])
    for key, data in input_point_data.items():
        assert numpy.allclose(data, point_data[key])
    for key, data in input_cell_data.items():
        assert numpy.allclose(data, cell_data[key])

    os.remove(filename)

    return

if __name__ == '__main__':
    test_io()
