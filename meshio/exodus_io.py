# -*- coding: utf-8 -*-
#
'''
I/O for Exodus II.

See <http://prod.sandia.gov/techlib/access-control.cgi/1992/922137.pdf>, in
particular Appendix A (page 171, Implementation of EXODUS II with netCDF).

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import datetime

import numpy

from .__about__ import __version__


exodus_to_meshio_type = {
    # curves
    'BEAM': 'line',
    'BEAM2': 'line',
    'BEAM3': 'line3',
    'BAR2': 'line',
    # surfaces
    'SHELL': 'quad',
    'SHELL4': 'quad',
    'SHELL8': 'quad8',
    'SHELL9': 'quad9',
    'QUAD': 'quad',
    'QUAD4': 'quad',
    'QUAD5': 'quad5',
    'QUAD8': 'quad8',
    'QUAD9': 'quad9',
    #
    'TRIANGLE': 'triangle',
    # 'TRI': 'triangle',
    'TRI3': 'triangle',
    'TRI7': 'triangle7',
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL7': 'triangle',
    #
    'TRI6': 'triangle6',
    # 'TRISHELL6': 'triangle6',
    # volumes
    'HEX': 'hexahedron',
    'HEXAHEDRON': 'hexahedron',
    'HEX8': 'hexahedron',
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
    'WEDGE': 'wedge'
    }
meshio_to_exodus_type = {v: k for k, v in exodus_to_meshio_type.items()}


def read(filename):
    import netCDF4

    nc = netCDF4.Dataset(filename)

    # assert nc.version == numpy.float32(5.1)
    # assert nc.api_version == numpy.float32(5.1)
    # assert nc.floating_point_word_size == 8

    # assert b''.join(nc.variables['coor_names'][0]) == b'X'
    # assert b''.join(nc.variables['coor_names'][1]) == b'Y'
    # assert b''.join(nc.variables['coor_names'][2]) == b'Z'

    points = numpy.zeros((len(nc.dimensions['num_nodes']), 3))
    point_data_names = []
    pd = []
    cells = {}
    for key, value in nc.variables.items():
        if key[:7] == 'connect':
            meshio_type = exodus_to_meshio_type[value.elem_type.upper()]
            if meshio_type in cells:
                cells[meshio_type] = \
                    numpy.vstack([cells[meshio_type], value[:] - 1])
            else:
                cells[meshio_type] = value[:] - 1
        elif key == 'coord':
            points = nc.variables['coord'][:].T
        elif key == 'coordx':
            points[:, 0] = value[:]
        elif key == 'coordy':
            points[:, 1] = value[:]
        elif key == 'coordz':
            points[:, 2] = value[:]
        elif key == 'name_nod_var':
            value.set_auto_mask(False)
            point_data_names = [b''.join(c).decode('UTF-8') for c in value[:]]
        elif key == 'vals_nod_var':
            pd = value[0, :]

    point_data = {name: dat for name, dat in zip(point_data_names, pd)}

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
          field_data=None):
    import netCDF4

    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

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
        )
    coor_names.set_auto_mask(False)
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
    # variables, the variable `vals_nod_var` hold the actual data.
    num_nod_var = len(point_data)
    if num_nod_var > 0:
        rootgrp.createDimension('num_nod_var', num_nod_var)
        # set names
        point_data_names = rootgrp.createVariable(
            'name_nod_var', 'S1', ('num_nod_var', 'len_string')
            )
        point_data_names.set_auto_mask(False)
        for k, name in enumerate(point_data.keys()):
            for i, letter in enumerate(name):
                point_data_names[k, i] = letter.encode('utf-8')

        # Set data.
        # Deliberately take the dtype from the first data block.
        first_key = list(point_data.keys())[0]
        dtype = numpy_to_exodus_dtype[point_data[first_key].dtype]
        node_data = rootgrp.createVariable(
            'vals_nod_var', dtype,
            ('time_step', 'num_nod_var', 'num_nodes')
            )
        for k, (name, data) in enumerate(point_data.items()):
            node_data[0, k] = data

    rootgrp.close()
    return
