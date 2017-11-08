# -*- coding: utf-8 -*-
#
'''
I/O for Exodus etc.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy


def read(filename):
    import netCDF4

    nc = netCDF4.Dataset(filename)

    points = numpy.array([
        nc.variables['coordx'][:],
        nc.variables['coordy'][:],
        nc.variables['coordz'][:],
        ]).T

    elem_types = {
        'quad': 'quad',
        'HEX': 'hexahedron',
        'TETRA': 'tetra',
        'TRIANGLE': 'triangle',
        }

    point_data = {}
    if 'name_nod_var' in nc.variables:
        variable_names = [
            b''.join(c).decode('UTF-8')
            for c in nc.variables['name_nod_var'][:]
            ]

        n = len(variable_names)
        is_accounted_for = n * [False]

        # Exodus stores variables as "<name>_Z", "<name>_R" to form a
        # two-dimensional value and "<name>X", "<name>Y", "<name>Z" to form a
        # three-dimensional value. Separate those out first
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

        for name, indices in variable_indices.items():
            if len(indices) == 1:
                point_data[name] = \
                    nc.variables['vals_nod_var{}'.format(indices[0]+1)][:]
            else:
                point_data[name] = numpy.concatenate([
                    nc.variables['vals_nod_var{}'.format(index+1)][:]
                    for index in indices
                    ]).T

    cells = {}
    for key in nc.variables:
        var = nc.variables[key]
        try:
            if var.elem_type in elem_types:
                cells[elem_types[var.elem_type]] = var[:] - 1
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
