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


def read(filename):  # noqa: C901
    import netCDF4

    with netCDF4.Dataset(filename) as nc:
        # assert nc.version == np.float32(5.1)
        # assert nc.api_version == np.float32(5.1)
        # assert nc.floating_point_word_size == 8

        # assert b''.join(nc.variables['coor_names'][0]) == b'X'
        # assert b''.join(nc.variables['coor_names'][1]) == b'Y'
        # assert b''.join(nc.variables['coor_names'][2]) == b'Z'

        points = np.zeros((len(nc.dimensions["num_nodes"]), 3))
        point_data_names = []
        cell_data_names = []
        pd = {}
        cd = {}
        cells = []
        ns_names = []
        # eb_names = []
        ns = []
        point_sets = {}
        info = []

        for key, value in nc.variables.items():
            if key == "info_records":
                value.set_auto_mask(False)
                for c in value[:]:
                    try:
                        info += [b"".join(c).decode("UTF-8")]
                    except UnicodeDecodeError:
                        # https://github.com/nschloe/meshio/issues/983
                        pass
            elif key == "qa_records":
                value.set_auto_mask(False)
                for val in value:
                    info += [b"".join(c).decode("UTF-8") for c in val[:]]
            elif key[:7] == "connect":
                meshio_type = exodus_to_meshio_type[value.elem_type.upper()]
                cells.append((meshio_type, value[:] - 1))
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
                idx = 0 if len(key) == 12 else int(key[12:]) - 1
                value.set_auto_mask(False)
                # For now only take the first value
                pd[idx] = value[0]
                if len(value) > 1:
                    warn("Skipping some time data")
            elif key == "name_elem_var":
                value.set_auto_mask(False)
                cell_data_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            elif key[:13] == "vals_elem_var":
                # eb: element block
                m = re.match("vals_elem_var(\\d+)?(?:eb(\\d+))?", key)
                idx = 0 if m.group(1) is None else int(m.group(1)) - 1
                block = 0 if m.group(2) is None else int(m.group(2)) - 1

                value.set_auto_mask(False)
                # For now only take the first value
                if idx not in cd:
                    cd[idx] = {}
                cd[idx][block] = value[0]

                if len(value) > 1:
                    warn("Skipping some time data")
            elif key == "ns_names":
                value.set_auto_mask(False)
                ns_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            # elif key == "eb_names":
            #     value.set_auto_mask(False)
            #     eb_names = [b"".join(c).decode("UTF-8") for c in value[:]]
            elif key.startswith("node_ns"):  # Expected keys: node_ns1, node_ns2
                ns.append(value[:] - 1)  # Exodus is 1-based

        # merge element block data; can't handle blocks yet
        for k, value in cd.items():
            cd[k] = np.concatenate(list(value.values()))

        # Check if there are any <name>R, <name>Z tuples or <name>X, <name>Y, <name>Z
        # triplets in the point data. If yes, they belong together.
        single, double, triple = categorize(point_data_names)

        point_data = {}
        for name, idx in single:
            point_data[name] = pd[idx]
        for name, idx0, idx1 in double:
            point_data[name] = np.column_stack([pd[idx0], pd[idx1]])
        for name, idx0, idx1, idx2 in triple:
            point_data[name] = np.column_stack([pd[idx0], pd[idx1], pd[idx2]])

        cell_data = {}
        k = 0
        for _, cell in cells:
            n = len(cell)
            for name, data in zip(cell_data_names, cd.values()):
                if name not in cell_data:
                    cell_data[name] = []
                cell_data[name].append(data[k : k + n])
            k += n

        point_sets = {name: dat for name, dat in zip(ns_names, ns)}

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
            try:
                iy = names.index(name[:-1] + "Y")
            except ValueError:
                iy = None
            try:
                iz = names.index(name[:-1] + "Z")
            except ValueError:
                iz = None
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
            try:
                iz = names.index(name[:-2] + "_Z")
            except ValueError:
                iz = None
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


def write(filename, mesh, **kwargs):
    """
    Valid kwargs:
        point_data: list | dict
        cell_data: list | dict
        time_step_size: float
        time_final: float
        timestep_arr: list[float]
    If point/cell data are not specified, those contained in the mesh will be used.
    If data are specified with lists, each entry will be associated with a timestep.
        Otherwise, the data is treated as a single timestep.
    Timestep definition works as follows:
        If timestep_arr (list of all time values) is given, it is used.
        Otherwise, if time_step_size (and optionally time_final) is (are) given,
            a linspace array of times is created based on this fixed timestep.
    """

    import netCDF4

    # get data, compatible with multiple timesteps
    point_data = kwargs.get("point_data", mesh.point_data)
    if isinstance(point_data, dict):
        point_data = [point_data]
    cell_data = kwargs.get("cell_data", mesh.cell_data)
    if isinstance(cell_data, dict):
        cell_data = [cell_data]

    # type checking for data
    if not isinstance(point_data, list):
        raise TypeError(
            f"Incompatible point data (type {type(point_data)}, expected list or dict)"
        )
    for i, d in enumerate(point_data):
        if not isinstance(d, dict):
            raise TypeError(
                f"Incompatible point data for timestep {i} (type {type(d)}, expected dict) "
            )
    if not isinstance(cell_data, list):
        raise TypeError(
            f"Incompatible cell data (type {type(cell_data)}, expected list or dict)"
        )
    for i, d in enumerate(cell_data):
        if not isinstance(d, dict):
            raise TypeError(
                f"Incompatible cell data for timestep {i} (type {type(d)}, expected dict)"
            )

    # determine time steps
    time_step_size = kwargs.get("time_step_size", 1.0)
    time_initial = kwargs.get("time_initial", 0.0)
    time_final = kwargs.get("time_final", None)
    timestep_arr = kwargs.get("timestep_arr", None)
    if timestep_arr is None:
        num_timesteps = max(len(point_data), len(cell_data))
        if time_final is None:
            time_final = time_step_size * (num_timesteps - 1)
        timestep_arr = np.linspace(time_initial, time_final)

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

        # time steps
        data = rootgrp.createVariable("time_whole", "f4", ("time_step",))
        data.set_auto_mask(False)
        data[:] = timestep_arr

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
        data = rootgrp.createVariable("eb_prop1", "i4", "num_el_blk")
        for block_idx in range(len(mesh.cells)):
            eb_id = block_idx + 1
            data[block_idx] = (
                eb_id
            )  # eb_prop1 should be 1-indexed for full Exodus compliance (e.g. for SEACAS tools)
        for block_idx, cell_block in enumerate(mesh.cells):
            eb_id = block_idx + 1
            dim1 = f"num_el_in_blk{eb_id}"
            dim2 = f"num_nod_per_el{eb_id}"
            rootgrp.createDimension(dim1, cell_block.data.shape[0])
            rootgrp.createDimension(dim2, cell_block.data.shape[1])
            dtype = numpy_to_exodus_dtype[cell_block.data.dtype.name]
            data = rootgrp.createVariable(f"connect{eb_id}", dtype, (dim1, dim2))
            data.elem_type = meshio_to_exodus_type[cell_block.type]
            # Exodus is 1-based
            data[:] = cell_block.data + 1

        # point data
        # The variable `name_nod_var` holds the names and indices of the node variables, the
        # variables `vals_nod_var{1,2,...}` hold the actual data.
        # assuming all timesteps have the same variables
        num_nod_var = len(point_data[0])
        if num_nod_var > 0:
            rootgrp.createDimension("num_nod_var", num_nod_var)

            # set names
            point_data_names = rootgrp.createVariable(
                "name_nod_var", "S1", ("num_nod_var", "len_string")
            )
            point_data_names.set_auto_mask(False)

            # fix order of variables once to avoid undefined dict order behavior
            name_arr = list(point_data[0].keys())
            
            # set names and create variables
            for var_idx, name in enumerate(name_arr):
                var_id = var_idx + 1
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    point_data_names[var_idx, i] = letter.encode("ascii")
                dtype = point_data[0][name].dtype.name
                rootgrp.createVariable(
                    f"vals_nod_var{var_id}",
                    numpy_to_exodus_dtype[dtype],
                    ("time_step", "num_nodes",),
                    fill_value=False,
                )
            
            # set data values
            for ts, point_data_ts in enumerate(point_data):
                for var_idx, var_name in enumerate(name_arr):
                    var_id = var_idx + 1
                    var = rootgrp.variables[f"vals_nod_var{var_id}"]
                    values = point_data_ts[name]
                    var[ts] = values

        # cell data
        num_cell_var = len(cell_data[0])
        if num_cell_var > 0:
            rootgrp.createDimension("num_elem_var", num_cell_var)
            
            # set names
            cell_data_names = rootgrp.createVariable(
                "name_elem_var", "S1", ("num_elem_var", "len_string")
            )
            cell_data_names.set_auto_mask(False)

            # fix order of variables once to avoid undefined dict order behavior
            name_arr = list(cell_data[0].keys())
            
            # set names and create variables
            for var_idx, name in enumerate(name_arr):
                var_id = var_idx + 1
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    cell_data_names[var_idx, i] = letter.encode("ascii")
                dtype = cell_data[0][name].dtype.name
                for block_idx in range(len(mesh.cells)):
                    eb_id = block_idx + 1
                    rootgrp.createVariable(
                        f"vals_elem_var{var_id}eb{eb_id}",
                        numpy_to_exodus_dtype[dtype],
                        ("time_step", f"num_elem_in_blk{eb_id}",),
                        fill_value=False,
                    )
            
            # fill data values
            for ts, cell_data_ts in enumerate(point_data):
                for var_idx, var_name in enumerate(name_arr):
                    for block_idx in range(len(mesh.cells)):
                        var_id = var_idx + 1
                        eb_id = block_idx + 1
                        var = rootgrp.variables[f"vals_elem_var{var_id}eb{eb_id}"]
                        values = cell_data_ts[name][block_idx]
                        var[ts] = values
            
        # node sets
        num_point_sets = len(mesh.point_sets)
        if num_point_sets > 0:
            data = rootgrp.createVariable("ns_prop1", "i4", "num_node_sets")
            data_names = rootgrp.createVariable(
                "ns_names", "S1", ("num_node_sets", "len_string")
            )
            for k, name in enumerate(mesh.point_sets.keys()):
                data[k] = k + 1
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    data_names[k, i] = letter.encode("ascii")
            for k, (key, values) in enumerate(mesh.point_sets.items()):
                dim1 = f"num_nod_ns{k + 1}"
                rootgrp.createDimension(dim1, values.shape[0])
                dtype = numpy_to_exodus_dtype[values.dtype.name]
                data = rootgrp.createVariable(f"node_ns{k + 1}", dtype, (dim1,))
                data[:] = values + 1


register_format("exodus", [".e", ".exo", ".ex2"], read, {"exodus": write})
