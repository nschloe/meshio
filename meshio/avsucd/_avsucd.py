"""
I/O for AVS-UCD format, cf.
<https://lanl.github.io/LaGriT/pages/docs/read_avs.html>.
"""
import logging

import numpy

from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


meshio_data = {"avsucd:material", "flac3d:zone", "gmsh:physical", "medit:ref"}


meshio_to_avsucd_type = {
    "vertex": "pt",
    "line": "line",
    "triangle": "tri",
    "quad": "quad",
    "tetra": "tet",
    "pyramid": "pyr",
    "wedge": "prism",
    "hexahedron": "hex",
}
avsucd_to_meshio_type = {v: k for k, v in meshio_to_avsucd_type.items()}


def read(filename):
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Skip comments and unpack first line
    num_nodes, num_cells, num_node_data, num_cell_data, _ = numpy.genfromtxt(
        f, max_rows=1, dtype=int, comments="#"
    )

    # Read nodes
    point_ids, points = _read_nodes(f, num_nodes)

    # Read cells
    cell_ids, cells, cell_data = _read_cells(f, num_cells, point_ids)

    # Read node data
    if num_node_data:
        point_data = _read_node_data(f, num_nodes, point_ids)
    else:
        point_data = {}

    # Read cell data
    if num_cell_data:
        cell_data.update(_read_cell_data(f, num_cells, cells, cell_ids))

    # Convert cell_data to array
    for k, v in cell_data.items():
        for kk, vv in v.items():
            cell_data[k][kk] = numpy.array(vv)

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def _read_nodes(f, num_nodes):
    data = numpy.genfromtxt(f, max_rows=num_nodes)
    points_ids = {int(pid): i for i, pid in enumerate(data[:, 0])}
    return points_ids, data[:, 1:]


def _read_cells(f, num_cells, point_ids):
    cells = {}
    cell_ids = {}
    cell_data = {"avsucd:material": {}}
    count = {k: 0 for k in meshio_to_avsucd_type.keys()}
    for _ in range(num_cells):
        line = f.readline().strip().split()
        cell_id, cell_mat = int(line[0]), int(line[1])
        cell_type = avsucd_to_meshio_type[line[2]]
        corner = [point_ids[int(pid)] for pid in line[3:]]

        if cell_type not in cells:
            cells[cell_type] = [corner]
            cell_data["avsucd:material"][cell_type] = [cell_mat]
        else:
            cells[cell_type].append(corner)
            cell_data["avsucd:material"][cell_type].append(cell_mat)

        cell_ids[cell_id] = (cell_type, count[cell_type])
        count[cell_type] += 1

    for k, v in cells.items():
        cells[k] = numpy.array(v)
    return cell_ids, cells, cell_data


def _read_node_data(f, num_nodes, point_ids):
    line = f.readline().strip().split()
    node_data_size = [int(i) for i in line[1:]]

    labels = {}
    point_data = {}
    for i, dsize in enumerate(node_data_size):
        line = f.readline().strip().split(",")
        labels[i] = line[0].strip().replace(" ", "_")
        point_data[labels[i]] = (
            numpy.empty(num_nodes) if dsize == 1 else numpy.empty((num_nodes, dsize))
        )

    for _ in range(num_nodes):
        line = f.readline().strip().split()
        pid = point_ids[int(line[0])]
        j = 0
        for i, dsize in enumerate(node_data_size):
            if dsize == 1:
                point_data[labels[i]][pid] = float(line[j + 1])
            else:
                point_data[labels[i]][pid] = [
                    float(val) for val in line[j + 1 : j + 1 + dsize]
                ]
            j += dsize
    return point_data


def _read_cell_data(f, num_cells, cells, cell_ids):
    line = f.readline().strip().split()
    cell_data_size = [int(i) for i in line[1:]]

    labels = {}
    cell_data = {}
    for i, dsize in enumerate(cell_data_size):
        line = f.readline().strip().split(",")
        labels[i] = line[0].strip().replace(" ", "_")
        cell_data[labels[i]] = {
            k: numpy.empty(len(v)) if dsize == 1 else numpy.empty((len(v), dsize))
            for k, v in cells.items()
        }

    for _ in range(num_cells):
        line = f.readline().strip().split()
        cell_type, cid = cell_ids[int(line[0])]
        j = 0
        for i, dsize in enumerate(cell_data_size):
            if dsize == 1:
                cell_data[labels[i]][cell_type][cid] = float(line[j + 1])
            else:
                cell_data[labels[i]][cell_type][cid] = [
                    float(val) for val in line[j + 1 : j + 1 + dsize]
                ]
            j += dsize
    return cell_data


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        logging.warning(
            "AVS-UCD requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    with open_file(filename, "w") as f:
        # Write first line
        num_nodes = len(mesh.points)
        num_cells = sum(len(v) for v in mesh.cells.values())
        num_node_data = [
            1 if v.ndim == 1 else v.shape[1] for v in mesh.point_data.values()
        ]
        num_cell_data = [
            1 if vv.ndim == 1 else vv.shape[1]
            for k, v in mesh.cell_data.items()
            for vv in v.values()
            if k not in meshio_data
        ]
        num_node_data_sum = sum(num_node_data)
        num_cell_data_sum = sum(num_cell_data)
        f.write(
            f"{num_nodes} {num_cells} {num_node_data_sum} {num_cell_data_sum} 0\n".format()
        )

        # Write nodes
        _write_nodes(f, mesh.points)

        # Write cells
        _write_cells(f, mesh.cells, mesh.cell_data, num_cells)

        # Write node data
        if num_node_data_sum:
            _write_node_data(
                f, mesh.point_data, num_nodes, num_node_data, num_node_data_sum
            )

        # Write cell data
        if num_cell_data_sum:
            _write_cell_data(
                f, mesh.cell_data, num_cells, num_cell_data, num_cell_data_sum
            )


def _write_nodes(f, points):
    for i, (x, y, z) in enumerate(points):
        f.write(f"{i+1} {x} {y} {z}\n")


def _write_cells(f, cells, cell_data, num_cells):
    # Interoperability with other formats
    mat_data = None
    for k in cell_data.keys():
        if k in meshio_data:
            mat_data = k
            break

    # Material array
    if mat_data:
        material = numpy.concatenate([v for v in cell_data[mat_data].values()])
    else:
        material = numpy.zeros(num_cells, dtype = int)

    # Loop over cells
    i = 0
    for k, v in cells.items():
        for cell in v:
            cell_str = " ".join(str(c + 1) for c in cell)
            f.write(f"{i+1} {int(material[i])} {meshio_to_avsucd_type[k]} {cell_str}\n")
            i += 1


def _write_node_data(f, point_data, num_nodes, num_node_data, num_node_data_sum):
    num_node_data_str = " ".join(str(i) for i in num_node_data)
    f.write(f"{len(num_node_data)} {num_node_data_str}\n")

    for label in point_data.keys():
        f.write(f"{label}, real\n")

    point_data_array = numpy.hstack(
        [v[:, None] if v.ndim == 1 else v for v in point_data.values()]
    )
    point_data_array = numpy.hstack(
        (numpy.arange(1, num_nodes + 1)[:, None], point_data_array)
    )
    numpy.savetxt(
        f, point_data_array, delimiter=" ", fmt=["%d"] + ["%.14e"] * num_node_data_sum
    )


def _write_cell_data(f, cell_data, num_cells, num_cell_data, num_cell_data_sum):
    num_cell_data_str = " ".join(str(i) for i in num_cell_data)
    f.write(f"{len(num_cell_data)} {num_cell_data_str}\n")

    for label in cell_data.keys():
        if label not in meshio_data:
            f.write(f"{label}, real\n")

    cell_data_array = numpy.hstack(
        [
            vv[:, None] if vv.ndim == 1 else vv
            for k, v in cell_data.items()
            for vv in v.values()
            if k not in meshio_data
        ]
    )
    cell_data_array = numpy.hstack(
        (numpy.arange(1, num_cells + 1)[:, None], cell_data_array)
    )
    numpy.savetxt(
        f, cell_data_array, delimiter=" ", fmt=["%d"] + ["%.14e"] * num_cell_data_sum
    )


register("avsucd", [".inp"], read, {"avsucd": write})
