# -*- coding: utf-8 -*-
#
"""
I/O for Exodus II.

See <http://prod.sandia.gov/techlib/access-control.cgi/1992/922137.pdf>, in
particular Appendix A (page 171, Implementation of EXODUS II with netCDF).
"""
import datetime

import numpy

from .__about__ import __version__
from .mesh import Mesh


exodus_to_meshio_type = {
    # curves
    "BEAM": "line",
    "BEAM2": "line",
    "BEAM3": "line3",
    "BAR2": "line",
    # surfaces
    "SHELL": "quad",
    "SHELL4": "quad",
    "SHELL8": "quad8",
    "SHELL9": "quad9",
    "QUAD": "quad",
    "QUAD4": "quad",
    "QUAD5": "quad5",
    "QUAD8": "quad8",
    "QUAD9": "quad9",
    #
    "TRIANGLE": "triangle",
    # 'TRI': 'triangle',
    "TRI3": "triangle",
    "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL7': 'triangle',
    #
    "TRI6": "triangle6",
    # 'TRISHELL6': 'triangle6',
    # volumes
    "HEX": "hexahedron",
    "HEXAHEDRON": "hexahedron",
    "HEX8": "hexahedron",
    "HEX9": "hexahedron9",
    "HEX20": "hexahedron20",
    "HEX27": "hexahedron27",
    #
    "TETRA": "tetra",
    "TETRA4": "tetra4",
    "TETRA8": "tetra8",
    "TETRA10": "tetra10",
    "TETRA14": "tetra14",
    #
    "PYRAMID": "pyramid",
    "WEDGE": "wedge",
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

    points = numpy.zeros((len(nc.dimensions["num_nodes"]), 3))
    point_data_names = []
    pd = {}
    cells = {}
    ns_names = []
    ns = []
    node_sets = {}
    for key, value in nc.variables.items():
        if key[:7] == "connect":
            meshio_type = exodus_to_meshio_type[value.elem_type.upper()]
            if meshio_type in cells:
                cells[meshio_type] = numpy.vstack([cells[meshio_type], value[:] - 1])
            else:
                cells[meshio_type] = value[:] - 1
        elif key == "coord":
            points = nc.variables["coord"][:].T
        elif key == "coordx":
            points[:, 0] = value[:]
        elif key == "coordy":
            points[:, 1] = value[:]
        elif key == "coordz":
            points[:, 2] = value[:]
        elif key == "name_nod_var":
            value.set_auto_mask(False)
            point_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
        elif key[:12] == "vals_nod_var":
            if len(key) == 12:
                idx = 0
            else:
                idx = int(key[12:]) - 1
            value.set_auto_mask(False)
            pd[idx] = value[0]
        elif key == "ns_names":
            value.set_auto_mask(False)
            ns_names = [b"".join(c).decode("UTF-8") for c in value[:]]
        elif key == "node_ns":
            ns = value

    # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
    # triplets in the point data. If yes, they belong together.
    single, double, triple = categorize(point_data_names)

    point_data = {}
    for name, idx in single:
        point_data[name] = pd[idx]
    for name, idx0, idx1 in double:
        point_data[name] = numpy.column_stack([pd[idx0], pd[idx1]])
    for name, idx0, idx1, idx2 in triple:
        point_data[name] = numpy.column_stack([pd[idx0], pd[idx1], pd[idx2]])

    node_sets = {name: dat for name, dat in zip(ns_names, ns)}

    nc.close()
    return Mesh(points, cells, point_data=point_data, node_sets=node_sets)


def categorize(names):
    # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
    # triplets in the point data. If yes, they belong together.
    single = []
    double = []
    triple = []
    is_accounted_for = [False] * len(names)
    k = 0
    while True:
        if k == len(names):
            break
        if is_accounted_for[k]:
            k += 1
            continue
        name = names[k]
        if name[-1] == "X":
            ix = k
            found_y = False
            try:
                iy = names.index(name[:-1] + "Y")
            except ValueError:
                pass
            else:
                found_y = True
            try:
                iz = names.index(name[:-1] + "Z")
            except ValueError:
                pass
            else:
                found_z = True
            if found_y and found_z:
                triple.append((name[:-1], ix, iy, iz))
                is_accounted_for[ix] = True
                is_accounted_for[iy] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ix))
                is_accounted_for[ix] = True
        elif name[-2:] == "_R":
            ir = k
            found_z = False
            try:
                iz = names.index(name[:-2] + "_Z")
            except ValueError:
                pass
            else:
                found_z = True
            if found_z:
                double.append((name[:-2], ir, iz))
                is_accounted_for[ir] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ir))
                is_accounted_for[ir] = True
        else:
            single.append((name, k))
            is_accounted_for[k] = True

        k += 1

    assert all(is_accounted_for)
    return single, double, triple


numpy_to_exodus_dtype = {
    "float32": "f4",
    "float64": "f8",
    "int8": "i1",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
    "uint8": "u1",
    "uint16": "u2",
    "uint32": "u4",
    "uint64": "u8",
}


def write(filename, mesh):
    import netCDF4

    rootgrp = netCDF4.Dataset(filename, "w")

    # set global data
    rootgrp.title = "Created by meshio v{}, {}".format(
        __version__, datetime.datetime.now().isoformat()
    )
    rootgrp.version = numpy.float32(5.1)
    rootgrp.api_version = numpy.float32(5.1)
    rootgrp.floating_point_word_size = 8

    # set dimensions
    total_num_elems = sum([v.shape[0] for v in mesh.cells.values()])
    rootgrp.createDimension("num_nodes", len(mesh.points))
    rootgrp.createDimension("num_dim", mesh.points.shape[1])
    rootgrp.createDimension("num_elem", total_num_elems)
    rootgrp.createDimension("num_el_blk", len(mesh.cells))
    rootgrp.createDimension("num_node_sets", len(mesh.node_sets))
    rootgrp.createDimension("len_string", 33)
    rootgrp.createDimension("len_line", 81)
    rootgrp.createDimension("four", 4)
    rootgrp.createDimension("time_step", None)

    # dummy time step
    data = rootgrp.createVariable("time_whole", "f4", "time_step")
    data[:] = 0.0

    # points
    coor_names = rootgrp.createVariable("coor_names", "S1", ("num_dim", "len_string"))
    coor_names.set_auto_mask(False)
    coor_names[0, 0] = "X"
    coor_names[1, 0] = "Y"
    if mesh.points.shape[1] == 3:
        coor_names[2, 0] = "Z"
    data = rootgrp.createVariable(
        "coord", numpy_to_exodus_dtype[mesh.points.dtype.name], ("num_dim", "num_nodes")
    )
    data[:] = mesh.points.T

    # cells
    # ParaView needs eb_prop1 -- some ID. The values don't seem to matter as
    # long as they are different for the for different blocks.
    data = rootgrp.createVariable("eb_prop1", "i4", "num_el_blk")
    for k in range(len(mesh.cells)):
        data[k] = k
    for k, (key, values) in enumerate(mesh.cells.items()):
        dim1 = "num_el_in_blk{}".format(k + 1)
        dim2 = "num_nod_per_el{}".format(k + 1)
        rootgrp.createDimension(dim1, values.shape[0])
        rootgrp.createDimension(dim2, values.shape[1])
        dtype = numpy_to_exodus_dtype[values.dtype.name]
        data = rootgrp.createVariable("connect{}".format(k + 1), dtype, (dim1, dim2))
        data.elem_type = meshio_to_exodus_type[key]
        # Exodus is 1-based
        data[:] = values + 1

    # point data
    # The variable `name_nod_var` holds the names and indices of the node variables, the
    # variables `vals_nod_var{1,2,...}` hold the actual data.
    num_nod_var = len(mesh.point_data)
    if num_nod_var > 0:
        rootgrp.createDimension("num_nod_var", num_nod_var)
        # set names
        point_data_names = rootgrp.createVariable(
            "name_nod_var", "S1", ("num_nod_var", "len_string")
        )
        point_data_names.set_auto_mask(False)
        for k, name in enumerate(mesh.point_data.keys()):
            for i, letter in enumerate(name):
                point_data_names[k, i] = letter.encode("utf-8")

        # Set data. ParaView might have some problems here, see
        # <https://gitlab.kitware.com/paraview/paraview/issues/18403>.
        for k, (name, data) in enumerate(mesh.point_data.items()):
            print(data.shape)
            for i, s in enumerate(data.shape):
                rootgrp.createDimension("dim_nod_var{}{}".format(k, i), s)
            dims = ["time_step"] + [
                "dim_nod_var{}{}".format(k, i) for i in range(len(data.shape))
            ]
            node_data = rootgrp.createVariable(
                "vals_nod_var{}".format(k + 1),
                numpy_to_exodus_dtype[data.dtype.name],
                tuple(dims),
                fill_value=False,
            )
            node_data[0] = data

    # node sets
    num_node_sets = len(mesh.node_sets)
    if num_node_sets > 0:
        data = rootgrp.createVariable("ns_prop1", "i4", "num_node_sets")
        data_names = rootgrp.createVariable(
            "ns_names", "S1", ("num_node_sets", "len_string")
        )
        for k, name in enumerate(mesh.node_sets.keys()):
            data[k] = k
            for i, letter in enumerate(name):
                data_names[k, i] = letter.encode("utf-8")
        for k, (key, values) in enumerate(mesh.node_sets.items()):
            dim1 = "num_nod_ns{}".format(k + 1)
            rootgrp.createDimension(dim1, values.shape[0])
            dtype = numpy_to_exodus_dtype[values.dtype.name]
            data = rootgrp.createVariable("node_ns{}".format(k + 1), dtype, (dim1,))
            # Exodus is 1-based
            data[:] = values + 1

    rootgrp.close()
    return
