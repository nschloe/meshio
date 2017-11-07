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

    for key in nc.variables:
        var = nc.variables[key]
        print(var)

    cells = {}
    for key in nc.variables:
        var = nc.variables[key]
        try:
            if var.elem_type in elem_types:
                cells[elem_types[var.elem_type]] = var[:] - 1
        except AttributeError:
            pass

    return points, cells, {}, {}, {}


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
