"""
I/O for Exodus II.

See
<https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf>,
in particular Appendix A (page 171, Implementation of EXODUS II with netCDF).
"""

import datetime
import re

import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError
from .._helpers import register_format
from .._mesh import Mesh

exodus_to_meshio_type = {
    "SPHERE": "vertex",
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
    "TRI": "triangle",
    "TRIANGLE": "triangle",
    "TRI3": "triangle",
    "TRI6": "triangle6",
    "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL6': 'triangle6',
    # 'TRISHELL7': 'triangle',
    #
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
    "TET4": "tetra4",
    "TETRA8": "tetra8",
    "TETRA10": "tetra10",
    "TETRA14": "tetra14",
    #
    "PYRAMID": "pyramid",
    "WEDGE": "wedge",
}
meshio_to_exodus_type = {v: k for k, v in exodus_to_meshio_type.items()}


def _read_points(nc):
    points = np.zeros((len(nc.dimensions["num_nodes"]), 3))
    if "coord" in nc.variables:
        points = nc.variables["coord"][:].T
    else:
        if "coordx" in nc.variables:
            points[:, 0] = nc.variables["coordx"][:]
        if "coordy" in nc.variables:
            points[:, 1] = nc.variables["coordy"][:]
        if "coordz" in nc.variables:
            points[:, 2] = nc.variables["coordz"][:]
    return points

def _read_info(nc):
    info = []
    if "info_records" in nc.variables:
        value = nc.variables["info_records"]
        value.set_auto_mask(False)
        for c in value[:]:
            try:
                info += [b"".join(c).decode("UTF-8")]
            except UnicodeDecodeError:
                pass

    if "qa_records" in nc.variables:
        value = nc.variables["qa_records"]
        value.set_auto_mask(False)
        for val in value:
            info += [b"".join(c).decode("UTF-8") for c in val[:]]
    return info

def _read_cells(nc):
    cells = []
    for key, value in nc.variables.items():
        if key[:7] == "connect":
            meshio_type = exodus_to_meshio_type[value.elem_type.upper()]
            cells.append((meshio_type, value[:] - 1))
    return cells

def _read_node_data(nc):
    point_data_names = []
    pd = {}
    if "name_nod_var" in nc.variables:
        value = nc.variables["name_nod_var"]
        value.set_auto_mask(False)
        point_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
    
    for key, value in nc.variables.items():
        if key[:12] == "vals_nod_var":
            idx = 0 if len(key) == 12 else int(key[12:]) - 1
            value.set_auto_mask(False)
            pd[idx] = value[0]
            if len(value) > 1:
                warn("Skipping some time data")
    return point_data_names, pd

def _read_cell_data(nc):
    cell_data_names = []
    cd = {}
    if "name_elem_var" in nc.variables:
        value = nc.variables["name_elem_var"]
        value.set_auto_mask(False)
        cell_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]

    for key, value in nc.variables.items():
        if key[:13] == "vals_elem_var":
            m = re.match("vals_elem_var(\\d+)?(?:eb(\\d+))?", key)
            idx = 0 if m.group(1) is None else int(m.group(1)) - 1
            block = 0 if m.group(2) is None else int(m.group(2)) - 1

            value.set_auto_mask(False)
            if idx not in cd:
                cd[idx] = {}
            cd[idx][block] = value[0]

    return cell_data_names, cd

def _read_point_sets(nc):
    ns_names = []
    ns = []
    if "ns_names" in nc.variables:
        value = nc.variables["ns_names"]
        value.set_auto_mask(False)
        ns_names = [b"".join(c).decode("UTF-8") for c in value[:]]

    for key, value in nc.variables.items():
        if key.startswith("node_ns"):
            ns.append(value[:] - 1)

    return {name: dat for name, dat in zip(ns_names, ns)}

def read(filename):
    import netCDF4

    with netCDF4.Dataset(filename) as nc:
        points = _read_points(nc)
        cells = _read_cells(nc)
        info = _read_info(nc)
        
        point_data_names, pd = _read_node_data(nc)
        cell_data_names, cd = _read_cell_data(nc)
        point_sets = _read_point_sets(nc)

        # merge element block data
        for k, value in cd.items():
            cd[k] = np.concatenate(list(value.values()))

        # Process point data tuples
        single, double, triple = categorize(point_data_names)

        point_data = {}
        for name, idx in single:
            point_data[name] = pd[idx]
        for name, idx0, idx1 in double:
            point_data[name] = np.column_stack([pd[idx0], pd[idx1]])
        for name, idx0, idx1, idx2 in triple:
            point_data[name] = np.column_stack([pd[idx0], pd[idx1], pd[idx2]])

        # Process cell data
        cell_data = {}
        k = 0
        for _, cell in cells:
            n = len(cell)
            for name, data in zip(cell_data_names, cd.values()):
                if name not in cell_data:
                    cell_data[name] = []
                cell_data[name].append(data[k : k + n])
            k += n

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        point_sets=point_sets,
        info=info,
    )


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
            iy = _lookup(names, name, "Y")
            iz = _lookup(names, name, "Z")
            if iy and iz:
                triple.append((name[:-1], ix, iy, iz))
                is_accounted_for[ix] = True
                is_accounted_for[iy] = True
                is_accounted_for[iz] = True
            else:
                single.append((name, ix))
                is_accounted_for[ix] = True
        elif name[-2:] == "_R":
            ir = k
            iz = _lookup(names, name, "_Z", -2)
            if iz:
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

    if not all(is_accounted_for):
        raise ReadError()
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

def _lookup(names, name, suffix, location=-1):

    try:
        response = names.index(name[:location] + suffix)
    except ValueError:
        response = None
    return response

def write(filename, mesh):
    import netCDF4

    with netCDF4.Dataset(filename, "w") as rootgrp:
        # set global data
        now = datetime.datetime.now().isoformat()
        rootgrp.title = f"Created by meshio v{__version__}, {now}"
        rootgrp.version = np.float32(5.1)
        rootgrp.api_version = np.float32(5.1)
        rootgrp.floating_point_word_size = 8

        # set dimensions
        total_num_elems = sum(c.data.shape[0] for c in mesh.cells)
        rootgrp.createDimension("num_nodes", len(mesh.points))
        rootgrp.createDimension("num_dim", mesh.points.shape[1])
        rootgrp.createDimension("num_elem", total_num_elems)
        rootgrp.createDimension("num_el_blk", len(mesh.cells))
        rootgrp.createDimension("num_node_sets", len(mesh.point_sets))
        rootgrp.createDimension("len_string", 33)
        rootgrp.createDimension("len_line", 81)
        rootgrp.createDimension("four", 4)
        rootgrp.createDimension("time_step", None)

        # dummy time step
        data = rootgrp.createVariable("time_whole", "f4", ("time_step",))
        data[:] = 0.0

        # points
        coor_names = rootgrp.createVariable(
            "coor_names", "S1", ("num_dim", "len_string")
        )
        coor_names.set_auto_mask(False)
        coor_names[0, 0] = b"X"
        coor_names[1, 0] = b"Y"
        if mesh.points.shape[1] == 3:
            coor_names[2, 0] = b"Z"
        data = rootgrp.createVariable(
            "coord",
            numpy_to_exodus_dtype[mesh.points.dtype.name],
            ("num_dim", "num_nodes"),
        )
        data[:] = mesh.points.T

        # cells
        # ParaView needs eb_prop1 -- some ID. The values don't seem to matter as
        # long as they are different for the for different blocks.
        data = rootgrp.createVariable("eb_prop1", "i4", "num_el_blk")
        for k in range(len(mesh.cells)):
            data[k] = k
        for k, cell_block in enumerate(mesh.cells):
            dim1 = f"num_el_in_blk{k + 1}"
            dim2 = f"num_nod_per_el{k + 1}"
            rootgrp.createDimension(dim1, cell_block.data.shape[0])
            rootgrp.createDimension(dim2, cell_block.data.shape[1])
            dtype = numpy_to_exodus_dtype[cell_block.data.dtype.name]
            data = rootgrp.createVariable(f"connect{k + 1}", dtype, (dim1, dim2))
            data.elem_type = meshio_to_exodus_type[cell_block.type]
            # Exodus is 1-based
            data[:] = cell_block.data + 1

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
                    point_data_names[k, i] = letter.encode()

            # Set data. ParaView might have some problems here, see
            # <https://gitlab.kitware.com/paraview/paraview/-/issues/18403>.
            for k, (name, data) in enumerate(mesh.point_data.items()):
                for i, s in enumerate(data.shape):
                    rootgrp.createDimension(f"dim_nod_var{k}{i}", s)
                dims = ["time_step"] + [
                    f"dim_nod_var{k}{i}" for i in range(len(data.shape))
                ]
                node_data = rootgrp.createVariable(
                    f"vals_nod_var{k + 1}",
                    numpy_to_exodus_dtype[data.dtype.name],
                    tuple(dims),
                    fill_value=False,
                )
                node_data[0] = data

        # node sets
        num_point_sets = len(mesh.point_sets)
        if num_point_sets > 0:
            data = rootgrp.createVariable("ns_prop1", "i4", "num_node_sets")
            data_names = rootgrp.createVariable(
                "ns_names", "S1", ("num_node_sets", "len_string")
            )
            for k, name in enumerate(mesh.point_sets.keys()):
                data[k] = k
                for i, letter in enumerate(name):
                    data_names[k, i] = letter.encode()
            for k, (key, values) in enumerate(mesh.point_sets.items()):
                dim1 = f"num_nod_ns{k + 1}"
                rootgrp.createDimension(dim1, values.shape[0])
                dtype = numpy_to_exodus_dtype[values.dtype.name]
                data = rootgrp.createVariable(f"node_ns{k + 1}", dtype, (dim1,))
                # Exodus is 1-based
                data[:] = values + 1


register_format("exodus", [".e", ".exo", ".ex2"], read, {"exodus": write})
