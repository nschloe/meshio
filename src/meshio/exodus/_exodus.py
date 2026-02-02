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


def read(filename, **kwargs):  # noqa: C901
    """
    Valid kwargs:
        timestep: int | str
    By default, timestep=0 is used. Providing an integer timestep will return a single Mesh object
        with the data at that index timestep. If instead timestep="all" is passed, a list of Mesh
        objects (one per timestep) will be returned.
    """
    import netCDF4

    with netCDF4.Dataset(filename) as nc:
        # read static geometry
        num_nodes = len(nc.dimensions["num_nodes"])
        if "num_dim" in nc.dimensions:
            num_dim = len(nc.dimensions["num_dim"])
        else:
            num_dim = 3
        points = np.zeros((num_nodes, num_dim))
        point_data_names = []
        cell_data_names = []
        field_data_names = []
        cells = []
        ns_names = []
        # ss_names = [] # meshio has no side set analog
        # eb_names = []
        ns = []
        # ss = []
        info = []

        point_data_keys = []
        cell_data_keys = []
        for key, value in nc.variables.items():
            if key == "info_records":
                value.set_auto_mask(False)
                for c in value[:]:
                    try:
                        info += [b"".join(c).decode("UTF-8").strip("\x00")]
                    except UnicodeDecodeError:
                        # https://github.com/nschloe/meshio/issues/983
                        pass
            elif key == "qa_records":
                value.set_auto_mask(False)
                for val in value:
                    info += [b"".join(c).decode("UTF-8").strip("\x00") for c in val[:]]
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
                point_data_names = [
                    b"".join(c).decode("UTF-8").strip("\x00") for c in value[:]
                ]
            elif key[:12] == "vals_nod_var":
                point_data_keys.append(key)
            elif key == "name_elem_var":
                value.set_auto_mask(False)
                cell_data_names = [
                    b"".join(c).decode("UTF-8").strip("\x00") for c in value[:]
                ]
            elif key[:13] == "vals_elem_var":
                cell_data_keys.append(key)
            elif key == "name_glo_var":
                value.set_auto_mask(False)
                field_data_names = [
                    b"".join(c).decode("UTF-8").strip("\x00") for c in value[:]
                ]
            elif key == "ns_names":
                value.set_auto_mask(False)
                ns_names = [b"".join(c).decode("UTF-8").strip("\x00") for c in value[:]]
            # elif key == "eb_names":
            #     value.set_auto_mask(False)
            #     eb_names = [b"".join(c).decode("UTF-8").strip("\x00") for c in value[:]]
            elif key.startswith("node_ns"):  # Expected keys: node_ns1, node_ns2
                ns.append(value[:] - 1)  # Exodus is 1-based

        point_sets = {name: dat for name, dat in zip(ns_names, ns)}
        single, double, triple = categorize(point_data_names)
        single_c, double_c, triple_c = categorize(cell_data_names)

        # time logic
        num_steps = len(nc.dimensions["time_step"])
        timestep = kwargs.get("timestep", 0)
        if timestep == "all":
            indices = range(num_steps)
        elif isinstance(timestep, int):
            if (timestep < 0) or (timestep >= num_steps):
                raise ValueError(
                    f"Only {num_steps} available, timestep={timestep} is invalid"
                )
            indices = [timestep]
        else:
            raise TypeError("timestep must be either 'all' or an integer")

        # pre-fetch point and cell data variable objects
        pd_vars = {key: nc.variables[key] for key in point_data_keys}
        for v in pd_vars.values():
            v.set_auto_mask(False)
        cd_vars = {key: nc.variables[key] for key in cell_data_keys}
        for v in cd_vars.values():
            v.set_auto_mask(False)

        # data extraction at each timestep
        meshes = []
        for ts_idx in indices:
            pd = {}
            cd = {}

            for key in point_data_keys:
                index = 0 if len(key) == 12 else int(key[12:]) - 1

                pd[index] = pd_vars[key][ts_idx]

            for key in cell_data_keys:
                m = re.match("vals_elem_var(\\d+)?(?:eb(\\d+))?", key)
                var_idx = 0 if m.group(1) is None else int(m.group(1)) - 1
                eb_idx = 0 if m.group(2) is None else int(m.group(2)) - 1

                if var_idx not in cd:
                    cd[var_idx] = {}
                cd[var_idx][eb_idx] = cd_vars[key][ts_idx]

            # assign correct names to point data and keep linked variables (e.g. for vectors) together
            point_data = {}
            for name, idx in single:
                point_data[name] = pd[idx]
            for name, idx0, idx1 in double:
                point_data[name] = np.column_stack([pd[idx0], pd[idx1]])
            for name, idx0, idx1, idx2 in triple:
                point_data[name] = np.column_stack([pd[idx0], pd[idx1], pd[idx2]])

            # accounting for separate element blocks in cell data (though not currently taking block names)
            cell_data_raw = {}
            for var_idx, name in enumerate(cell_data_names):
                block_data_list = []
                for eb_idx in range(len(cells)):
                    data = cd.get(var_idx, {}).get(eb_idx)
                    if data is None:
                        num_cells_in_block = cells[eb_idx][1].shape[0]
                        data = np.zeros(num_cells_in_block)
                    block_data_list.append(data)

                cell_data_raw[var_idx] = block_data_list

            # now actually construct cell_data with the same linked variables/vectors functionality
            cell_data = {}
            for name, idx in single_c:
                cell_data[name] = cell_data_raw[idx]
            for name, idx0, idx1 in double_c:
                cell_data[name] = [
                    np.column_stack([cell_data_raw[idx0][eb], cell_data_raw[idx1][eb]])
                    for eb in range(len(cells))
                ]
            for name, idx0, idx1, idx2 in triple_c:
                cell_data[name] = [
                    np.column_stack(
                        [
                            cell_data_raw[idx0][eb],
                            cell_data_raw[idx1][eb],
                            cell_data_raw[idx2][eb],
                        ]
                    )
                    for eb in range(len(cells))
                ]

            # load field data, including time
            if "vals_glo_var" in nc.variables:
                values = nc.variables["vals_glo_var"][ts_idx]
                field_data = {name: val for name, val in zip(field_data_names, values)}
            else:
                field_data = {}
            if "time_whole" in nc.variables and "time" not in field_data:
                field_data["time"] = nc.variables["time_whole"][ts_idx]

            m = Mesh(
                points,
                cells,
                point_data=point_data,
                cell_data=cell_data,
                point_sets=point_sets,
                field_data=field_data,
                info=info,
            )
            meshes.append(m)

        return meshes if timestep == "all" else meshes[0]


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

            if iy is not None and iz is not None:
                triple.append((name[:-1], ix, iy, iz))
                is_accounted_for[ix] = is_accounted_for[iy] = is_accounted_for[iz] = True
            elif iy is not None:
                double.append((name[:-1], ix, iy))
                is_accounted_for[ix] = is_accounted_for[iy] = True
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
        field_data: list | dict
        time_whole: float
        time_step_size: float
        time_final: float
        timestep_arr: list[float]
        side_sets: dict[str, tuple[list, list]]
    If point/cell data are not specified, those contained in the mesh will be used.
    If data are specified with lists, each entry will be associated with a timestep.
        Otherwise, the data is treated as a single timestep.
    Timestep definition/inference works as follows:
        For single timestep:
            If time_whole is given, it is used.
            Otherwise, if "time" is a key in field_data, it is used.
            Otherwise, time_whole defaults to 0.0.
        For multiple timesteps:
            If timestep_arr (list of all time values) is given, it is used.
            Otherwise, if "time" is a key in field_data, it is used.
            Otherwise, if time_step_size (and optionally time_final) is (are) given,
                a linspace array of times is created based on this fixed timestep.
    To specify side sets, provide a dictionary of format {"set_name" : (elem_ids, side_ids)}
        Use pythonic 0-indexing; conversion to Exodus's 1-indexing will be done in the function
        Element indices should be global
    """

    import netCDF4

    # get data, standardize to support one or multiple timesteps
    point_data = kwargs.get("point_data", mesh.point_data)
    if isinstance(point_data, dict):
        point_data = [point_data]
    cell_data = kwargs.get("cell_data", mesh.cell_data)
    if isinstance(cell_data, dict):
        cell_data = [cell_data]
    # field data (typical for storing global variables)
    field_data = kwargs.get("field_data", mesh.field_data)
    if isinstance(field_data, dict):
        field_data = [field_data]

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
    if not isinstance(field_data, list):
        raise TypeError(
            f"Incompatible field data (type {type(field_data)}, expected list or dict)"
        )
    for i, d in enumerate(field_data):
        if not isinstance(d, dict):
            raise TypeError(
                f"Incompatible field data for timestep {i} (type {type(d)}, expected dict)"
            )

    # determine time steps
    time_step_size = kwargs.get("time_step_size", None)
    time_initial = kwargs.get("time_initial", None)
    time_final = kwargs.get("time_final", None)
    timestep_arr = kwargs.get("timestep_arr", None)
    num_timesteps = max(len(point_data), len(cell_data), len(field_data))
    if timestep_arr is None:
        if num_timesteps == 1:
            time_whole = kwargs.get("time_whole", None)
            if time_whole is None:
                if "time" in field_data[0]:
                    time_whole = field_data[0]["time"]
                else:
                    time_whole = 0.0
            elif not isinstance(time_whole, (int, float)):
                warn("Provided value of time_whole invalid, using 0.0")
                time_whole = 0.0
            timestep_arr = [float(time_whole)]
        else:
            if (
                time_step_size is None
                and time_initial is None
                and time_final is None
                and all("time" in fd for fd in field_data)
                and len(field_data) == num_timesteps
            ):
                timestep_arr = [fd["time"] for fd in field_data]
            else:
                if time_step_size is None:
                    time_step_size = 1.0
                if time_initial is None:
                    time_initial = 0.0
                if time_final is None:
                    time_final = time_initial + time_step_size * (num_timesteps - 1)
                timestep_arr = np.linspace(time_initial, time_final, num=num_timesteps)
    else:
        if (lt := len(timestep_arr)) != num_timesteps:
            raise ValueError(
                f"timestep_arr has length incompatible with data ({lt} != {num_timesteps})"
            )
    if num_timesteps > 1:
        if len(point_data) == 1:
            point_data = point_data * num_timesteps
        if len(cell_data) == 1:
            cell_data = cell_data * num_timesteps
        if len(field_data) == 1:
            field_data = field_data * num_timesteps

    # get side sets, if they exist
    side_sets = kwargs.get("side_sets")

    with netCDF4.Dataset(filename, "w") as rootgrp:
        # set global metadata
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
        data = rootgrp.createVariable("time_whole", "f8", "time_step")
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
                eb_id  # eb_prop1 should be 1-indexed for full Exodus compliance (e.g. for SEACAS tools)
            )
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
        # Assuming all timesteps have the same variables
        # fix order of variables once to avoid undefined dict order behavior
        pd_name_arr = list(point_data[0].keys())
        # split vectors into scalar components
        pd_names_split = []
        pd_dtypes = []
        for name in pd_name_arr:
            values = point_data[0][name]
            if len(values.shape) == 1:
                pd_names_split.append(name)
                pd_dtypes.append(values.dtype.name)
            else:
                suffixes = (
                    ["X", "Y", "Z"]
                    if values.shape[1] <= 3
                    else [str(i + 1) for i in range(values.shape[1])]
                )
                for i in range(values.shape[1]):
                    pd_names_split.append(f"{name}{suffixes[i]}")
                    pd_dtypes.append(values.dtype.name)
        # now with the split components, write names and data
        num_nod_var = len(pd_names_split)
        if num_nod_var > 0:
            rootgrp.createDimension("num_nod_var", num_nod_var)

            point_data_names = rootgrp.createVariable(
                "name_nod_var", "S1", ("num_nod_var", "len_string")
            )
            point_data_names.set_auto_mask(False)

            # set names and create variables
            for var_idx, name in enumerate(pd_names_split):
                var_id = var_idx + 1
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    point_data_names[var_idx, i] = letter.encode("ascii")
                dtype = pd_dtypes[var_idx]
                rootgrp.createVariable(
                    f"vals_nod_var{var_id}",
                    numpy_to_exodus_dtype[dtype],
                    (
                        "time_step",
                        "num_nodes",
                    ),
                    fill_value=False,
                )

            # fill data values
            for ts, point_data_ts in enumerate(point_data):
                var_idx = 0
                for var_name in pd_name_arr:
                    values = point_data_ts[var_name]
                    if len(values.shape) == 1:
                        var = rootgrp.variables[f"vals_nod_var{var_idx+1}"]
                        var[ts] = values
                        var_idx += 1
                    else:
                        for i in range(values.shape[1]):
                            var = rootgrp.variables[f"vals_nod_var{var_idx+1}"]
                            var[ts] = values[:, i]
                            var_idx += 1

        # cell data
        # similar procedure to point data
        cd_name_arr = list(cell_data[0].keys())
        cd_names_split = []
        cd_dtypes = []
        for name in cd_name_arr:
            blocks = cell_data[0][name]
            if len(blocks[0].shape) == 1:
                cd_names_split.append(name)
                cd_dtypes.append(blocks[0].dtype.name)
            else:
                suffixes = (
                    ["X", "Y", "Z"]
                    if blocks[0].shape[1] <= 3
                    else [str(i + 1) for i in range(blocks[0].shape[1])]
                )
                for i in range(blocks[0].shape[1]):
                    cd_names_split.append(f"{name}{suffixes[i]}")
                    cd_dtypes.append(blocks[0].dtype.name)
        num_cell_var = len(cd_names_split)
        if num_cell_var > 0:
            rootgrp.createDimension("num_elem_var", num_cell_var)

            cell_data_names = rootgrp.createVariable(
                "name_elem_var", "S1", ("num_elem_var", "len_string")
            )
            cell_data_names.set_auto_mask(False)

            for var_idx, name in enumerate(cd_names_split):
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    cell_data_names[var_idx, i] = letter.encode("ascii")
                dtype = cd_dtypes[var_idx]
                for block_idx in range(len(mesh.cells)):
                    var_id = var_idx + 1
                    eb_id = block_idx + 1
                    rootgrp.createVariable(
                        f"vals_elem_var{var_id}eb{eb_id}",
                        numpy_to_exodus_dtype[dtype],
                        (
                            "time_step",
                            f"num_el_in_blk{eb_id}",
                        ),
                        fill_value=False,
                    )

            # fill data values
            for ts, cell_data_ts in enumerate(cell_data):
                var_idx = 0
                for var_name in cd_name_arr:
                    blocks = cell_data_ts[var_name]
                    if len(blocks[0].shape) == 1:
                        for block_idx, values in enumerate(blocks):
                            var = rootgrp.variables[
                                f"vals_elem_var{var_idx+1}eb{block_idx+1}"
                            ]
                            var[ts] = values
                        var_idx += 1
                    else:
                        for i in range(blocks[0].shape[1]):
                            for block_idx, values in enumerate(blocks):
                                var = rootgrp.variables[
                                    f"vals_elem_var{var_idx+1}eb{block_idx+1}"
                                ]
                                var[ts] = values[:, i]
                            var_idx += 1

        # field data
        num_globals = len(field_data[0])
        if num_globals > 0:
            rootgrp.createDimension("num_glo_var", num_globals)
            global_names = list(field_data[0].keys())
            name_var = rootgrp.createVariable(
                "name_glo_var", "S1", ("num_glo_var", "len_string")
            )
            name_var.set_auto_mask(False)

            for k, name in enumerate(global_names):
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    name_var[k, i] = letter.encode("ascii")

            data = rootgrp.createVariable(
                "vals_glo_var", "f8", ("time_step", "num_glo_var")
            )
            for ts, field_data_ts in enumerate(field_data):
                row_values = [field_data_ts[name] for name in global_names]
                data[ts] = row_values

        # node sets
        num_point_sets = len(mesh.point_sets)
        if num_point_sets > 0:
            ns_prop = rootgrp.createVariable("ns_prop1", "i4", "num_node_sets")
            ns_names = rootgrp.createVariable(
                "ns_names", "S1", ("num_node_sets", "len_string")
            )
            ns_names.set_auto_mask(False)
            for ns_idx, (name, indices) in enumerate(mesh.point_sets.items()):
                ns_id = ns_idx + 1
                ns_prop[ns_idx] = ns_id
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    ns_names[ns_idx, i] = letter.encode("ascii")

                dim = f"num_nod_ns{ns_id}"
                rootgrp.createDimension(dim, indices.shape[0])
                data = rootgrp.createVariable(
                    f"node_ns{ns_id}", numpy_to_exodus_dtype[indices.dtype.name], dim
                )
                data[:] = indices + 1

        # side sets
        # there is no exact analog to this in meshio; this must be specified by the user
        if side_sets:
            rootgrp.createDimension("num_side_sets", len(side_sets))
            ss_prop = rootgrp.createVariable("ss_prop1", "i4", "num_side_sets")
            ss_names = rootgrp.createVariable(
                "ss_names", "S1", ("num_side_sets", "len_string")
            )
            ss_names.set_auto_mask(False)
            for ss_idx, (name, (elem_ids, side_ids)) in enumerate(side_sets.items()):
                ss_id = ss_idx + 1
                ss_prop[ss_idx] = ss_id
                for i, letter in enumerate(name.ljust(33, "\x00")):
                    ss_names[ss_idx, i] = letter.encode("ascii")

                dim = f"num_side_ss{ss_id}"
                rootgrp.createDimension(dim, len(elem_ids))

                # write parent element IDs
                el_var = rootgrp.createVariable(f"elem_ss{ss_id}", "i4", dim)
                el_var[:] = np.asarray(elem_ids) + 1

                # write local side IDs
                si_var = rootgrp.createVariable(f"side_ss{ss_id}", "i4", dim)
                si_var[:] = np.asarray(side_ids) + 1

        # Exodus does not have a common analog to meshio's cell_sets (there are Element Sets, but these are rarely used); typically, element blocks are used instead
        # Here, however, the element blocks are topology-based, so they cannot be used for other types of labels
        # As a workaround, one may use the mesh's cell_sets and call cell_sets_to_data() on the mesh to instead store cell labels in cell_data


register_format("exodus", [".e", ".exo", ".ex2"], read, {"exodus": write})
