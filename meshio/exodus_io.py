# -*- coding: utf-8 -*-
#
'''
I/O for Exodus II.

See
<http://prod.sandia.gov/techlib/access-control.cgi/1992/922137.pdf>,
in particular Appendix A (Implementation of EXODUS II with netCDF).

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import datetime

import numpy

from .__about__ import __version__


exodus_to_meshio_type = {
    # curves
    'BEAM': 'line',
    'BEAM2': 'line2',
    'BEAM3': 'line3',
    # surfaces
    'QUAD': 'quad',
    'QUAD4': 'quad4',
    'QUAD5': 'quad5',
    'QUAD8': 'quad8',
    'QUAD9': 'quad9',
    # 'SHELL': 'quad',
    # 'SHELL4': 'quad',
    # 'SHELL8': 'quad',
    # 'SHELL9': 'quad',
    #
    'TRIANGLE': 'triangle',
    # 'TRI': 'triangle',
    'TRI3': 'triangle3',
    'TRI7': 'triangle7',
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL7': 'triangle',
    #
    'TRI6': 'triangle6',
    # 'TRISHELL6': 'triangle6',
    # volumes
    'HEX': 'hexahedron',
    'HEX8': 'hexahedron8',
    'HEX9': 'hexahedron9',
    'HEX20': 'hexahedron20',
    'HEX27': 'hexahedron27',
    #
    'TETRA': 'tetra',
    'TETRA4': 'tetra4',
    'TETRA8': 'tetra8',
    'TETRA10': 'tetra10',
    'TETRA14': 'tetra14',
    #
    'PYRAMID': 'pyramid',
    }
meshio_to_exodus_type = {v: k for k, v in exodus_to_meshio_type.items()}


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

    assert nc.version == numpy.float32(5.1)
    assert nc.api_version == numpy.float32(5.1)
    assert nc.floating_point_word_size == 8

    assert b''.join(nc.variables['coor_names'][0]) == b'X'
    assert b''.join(nc.variables['coor_names'][1]) == b'Y'
    assert b''.join(nc.variables['coor_names'][2]) == b'Z'

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

    # cells
    cells = {}
    for key in nc.variables:
        var = nc.variables[key]
        try:
            if var.elem_type.upper() in exodus_to_meshio_type:
                cells[exodus_to_meshio_type[var.elem_type.upper()]] \
                    = var[:] - 1
        except AttributeError:
            pass

    # point data
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

    nc.close()
    return points, cells, point_data, {}, {}


numpy_to_exodus_dtype = {
    numpy.dtype(numpy.float32): 'f4',
    numpy.dtype(numpy.float64): 'f8',
    numpy.dtype(numpy.int8): 'i1',
    numpy.dtype(numpy.int16): 'i2',
    numpy.dtype(numpy.int32): 'i4',
    numpy.dtype(numpy.int64): 'i8',
    numpy.dtype(numpy.uint8): 'u1',
    numpy.dtype(numpy.uint16): 'u2',
    numpy.dtype(numpy.uint32): 'u4',
    numpy.dtype(numpy.uint64): 'u8',
    }


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    # from .legacy_writer import write as w
    # w('exodus', filename, points, cells, point_data, cell_data, field_data)
    # return

    import netCDF4

    rootgrp = netCDF4.Dataset(filename, 'w')

    # set global data
    rootgrp.title = \
        'Created by meshio v{}, {}'.format(
            __version__,
            datetime.datetime.now().isoformat()
            )
    rootgrp.version = numpy.float32(5.1)
    rootgrp.api_version = numpy.float32(5.1)
    rootgrp.floating_point_word_size = 8

    # set dimensions
    total_num_elems = sum([v.shape[0] for v in cells.values()])
    rootgrp.createDimension('num_nodes', len(points))
    rootgrp.createDimension('num_dim', 3)
    rootgrp.createDimension('num_elem', total_num_elems)
    rootgrp.createDimension('num_el_blk', len(cells))
    rootgrp.createDimension('len_string', 33)
    rootgrp.createDimension('len_line', 81)
    rootgrp.createDimension('four', 4)
    rootgrp.createDimension('time_step', None)

    # dummy time step
    data = rootgrp.createVariable('time_whole', 'f4', 'time_step')
    data[:] = 0.0

    # points
    coor_names = rootgrp.createVariable(
            'coor_names', 'S1', ('num_dim', 'len_string'),
            fill_value=b'x'
            )
    # Set the value multiple times; see bug
    # <https://github.com/Unidata/netcdf4-python/issues/746>.
    coor_names[:] = b''
    coor_names[0, 0] = 'X'
    coor_names[1, 0] = 'Y'
    coor_names[2, 0] = 'Z'
    data = rootgrp.createVariable(
            'coord',
            numpy_to_exodus_dtype[points.dtype],
            ('num_dim', 'num_nodes')
            )
    data[:] = points.T

    # cells
    # ParaView needs eb_prop1 -- some ID. The values don't seem to matter as
    # long as they are different for the for different blocks.
    data = rootgrp.createVariable('eb_prop1', 'i4', 'num_el_blk')
    for k in range(len(cells)):
        data[k] = k
    for k, (key, values) in enumerate(cells.items()):
        dim1 = 'num_el_in_blk{}'.format(k+1)
        dim2 = 'num_nod_per_el{}'.format(k+1)
        rootgrp.createDimension(dim1, values.shape[0])
        rootgrp.createDimension(dim2, values.shape[1])
        dtype = numpy_to_exodus_dtype[values.dtype]
        data = rootgrp.createVariable(
                'connect{}'.format(k+1), dtype, (dim1, dim2)
                )
        data.elem_type = meshio_to_exodus_type[key]
        # Exodus is 1-based
        data[:] = values + 1

    # point data
    # The variable `name_nod_var` holds the names and indices of the node
    # variables, the variables `vals_nod_var*` hold the actual data.
    # rootgrp.createDimension('num_' + exodus_type, values.shape[0])
    num_nod_var = len(point_data)
    if num_nod_var > 0:
        rootgrp.createDimension('num_nod_var', num_nod_var)
        # set names
        point_data_names = rootgrp.createVariable(
                'name_nod_var', 'S1', ('num_nod_var', 'len_string'),
                fill_value=b'x'
                )
        point_data_names[:] = b''
        for k, name in enumerate(point_data.keys()):
            for i, letter in enumerate(name):
                point_data_names[k, i] = letter.encode('utf-8')

        # set data
        node_data = rootgrp.createVariable(
                'vals_nod_var',
                numpy_to_exodus_dtype[data.dtype],
                ('time_step', 'num_nod_var', 'num_nodes')
                )
        for k, (name, data) in enumerate(point_data.items()):
            node_data[0, k] = data

    rootgrp.close()
    return
