# -*- coding: utf-8 -*-
#
'''
I/O for Exodus etc.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy


def merge_variables(variable_names):
    # Exodus stores variables as "<name>_Z", "<name>_R" to form a
    # two-dimensional value and "<name>X", "<name>Y", "<name>Z" to form a
    # three-dimensional value. Separate those out first
    n = len(variable_names)
    is_accounted_for = n * [False]

    variable_indices = {}

    for k, var in enumerate(variable_names):
        if is_accounted_for[k]:
            continue

        if var[-2:] == '_R':
            k2 = [
                i for i in range(k, n)
                if variable_names[i] == var[:-2] + '_Z'
                ]
            assert len(k2) < 2
            if k2:
                variable_indices[var[:-2]] = [k, k2[0]]
                is_accounted_for[k2[0]] = True
            else:
                variable_indices[var] = [k]
            is_accounted_for[k] = True
        elif var[-1] == 'X':
            ky = [
                i for i in range(k, n)
                if variable_names[i] == var[:-1] + 'Y'
                ]
            assert len(ky) < 2
            kz = [
                i for i in range(k, n)
                if variable_names[i] == var[:-1] + 'Z'
                ]
            assert len(kz) < 2
            if ky and kz:
                variable_indices[var[:-1]] = [k, ky[0], kz[0]]
                is_accounted_for[ky[0]] = True
                is_accounted_for[kz[0]] = True
            else:
                variable_indices[var] = [k]
            is_accounted_for[k] = True
        else:
            variable_indices[var] = [k]
            is_accounted_for[k] = True

    assert all(is_accounted_for)

    return variable_indices


def read(filename):
    import netCDF4

    nc = netCDF4.Dataset(filename)

    if 'coordx' in nc.variables:
        points = [
            nc.variables['coordx'][:],
            nc.variables['coordy'][:],
            ]
        if 'coordz' in nc.variables:
            points.append(nc.variables['coordz'][:])
        else:
            points.append(numpy.zeros(len(points[0])))
        points = numpy.array(points).T
    else:
        assert 'coord' in nc.variables
        points = nc.variables['coord'][:].T
        if points.shape[1] == 2:
            points = numpy.column_stack([
                points,
                numpy.zeros(len(points))
                ])

    # handle point data
    point_data = {}
    if 'name_nod_var' in nc.variables:
        variable_names = [
            b''.join(c).decode('UTF-8')
            for c in nc.variables['name_nod_var'][:]
            ]
        variable_indices = merge_variables(variable_names)
        for name, indices in variable_indices.items():
            if len(indices) == 1:
                point_data[name] = \
                    nc.variables['vals_nod_var{}'.format(indices[0]+1)][:]
            else:
                point_data[name] = numpy.concatenate([
                    nc.variables['vals_nod_var{}'.format(index+1)][:]
                    for index in indices
                    ]).T

    elem_types = {
        # curves
        'BEAM': 'line',
        'BEAM2': 'line',
        'BEAM3': 'line',
        # surfaces
        'QUAD': 'quad',
        'QUAD4': 'quad',
        'QUAD5': 'quad',
        'QUAD8': 'quad',
        'QUAD9': 'quad',
        'SHELL': 'quad',
        'SHELL4': 'quad',
        'SHELL8': 'quad',
        'SHELL9': 'quad',
        #
        'TRIANGLE': 'triangle',
        'TRI': 'triangle',
        'TRI3': 'triangle',
        'TRI6': 'triangle',
        'TRI7': 'triangle',
        'TRISHELL': 'triangle',
        'TRISHELL3': 'triangle',
        'TRISHELL6': 'triangle',
        'TRISHELL7': 'triangle',
        # volumes
        'HEX': 'hexahedron',
        'HEX8': 'hexahedron',
        'HEX9': 'hexahedron',
        'HEX20': 'hexahedron',
        'HEX27': 'hexahedron',
        #
        'TETRA': 'tetra',
        'TETRA4': 'tetra',
        'TETRA8': 'tetra',
        'TETRA10': 'tetra',
        'TETRA14': 'tetra',
        #
        'PYRAMID': 'pyramid',
        }

    cells = {}
    for key in nc.variables:
        var = nc.variables[key]
        try:
            if var.elem_type.upper() in elem_types:
                cells[elem_types[var.elem_type.upper()]] = var[:] - 1
        except AttributeError:
            pass

    nc.close()

    return points, cells, point_data, {}, {}


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    from . import vtk_io
    vtk_io.write(
            'exodus', filename, points, cells,
            point_data, cell_data, field_data
            )
    return
