"""
I/O for AVS-UCD format, cf.
<https://lanl.github.io/LaGriT/pages/docs/read_avs.html>.
"""
import numpy

from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


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


avsucd_to_meshio_order = {
    "vertex": [0],
    "line": [0, 1],
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 3, 2],
    "pyramid": [4, 0, 1, 2, 3],
    "wedge": [3, 4, 5, 0, 1, 2],
    "hexahedron": [4, 5, 6, 7, 0, 1, 2, 3],
}


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
        point_data = _read_node_data(f, num_nodes, num_node_data, point_ids)
    else:
        point_data = {}

    # Read cell data
    if num_cell_data:
        cell_data.update(_read_cell_data(f, num_cells, num_cell_data, cells, cell_ids))

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data={
            k: {kk: numpy.array(vv) for kk, vv in v.items()}
            for k, v in cell_data.items()
        },
    )


def _read_nodes(f, num_nodes):
    data = numpy.genfromtxt(f, max_rows=num_nodes)
    points_ids = {int(pid): i for i, pid in enumerate(data[:, 0])}
    return points_ids, data[:, 1:]


def _read_cells(f, num_cells, point_ids):
    cells = {}
    cell_ids = {}
    cell_data = {"avsucd:mat": {}}
    count = {k: 0 for k in meshio_to_avsucd_type.keys()}
    for _ in range(num_cells):
        line = f.readline().strip().split()
        cell_id, cell_mat = int(line[0]), int(line[1])
        cell_type = avsucd_to_meshio_type[line[2]]
        corner = [point_ids[int(pid)] for pid in line[3:]]

        if cell_type not in cells:
            cells[cell_type] = [corner]
            cell_data["avsucd:mat"][cell_type] = [cell_mat]
        else:
            cells[cell_type].append(corner)
            cell_data["avsucd:mat"][cell_type].append(cell_mat)

        cell_ids[cell_id] = (cell_type, count[cell_type])
        count[cell_type] += 1

    for k, v in cells.items():
        cells[k] = numpy.array(v)[:, avsucd_to_meshio_order[k]]
    return cell_ids, cells, cell_data


def _read_node_data(f, num_nodes, num_node_data, point_ids):
    line = f.readline()  # Not quite sure what to do with this line...

    labels = {}
    point_data = {}
    for i in range(num_node_data):
        line = f.readline().strip().split(",")
        labels[i] = line[0].strip()
        point_data[labels[i]] = numpy.empty(num_nodes)

    for _ in range(num_nodes):
        line = f.readline().strip().split()
        pid = point_ids[int(line[0])]
        for i, val in enumerate(line[1:]):
            point_data[labels[i]][pid] = float(val)
    return point_data


def _read_cell_data(f, num_cells, num_cell_data, cells, cell_ids):
    line = f.readline()  # Not quite sure what to do with this line...

    labels = {}
    cell_data = {}
    for i in range(num_cell_data):
        line = f.readline().strip().split(",")
        labels[i] = line[0].strip()
        cell_data[labels[i]] = {k: numpy.empty(len(v)) for k, v in cells.items()}

    for _ in range(num_cells):
        line = f.readline().strip().split()
        cell_type, cid = cell_ids[int(line[0])]
        for i, val in enumerate(line[1:]):
            cell_data[labels[i]][cell_type][cid] = float(val)
    return cell_data


def write(filename, mesh):
    pass


register("avsucd", [".inp"], read, {"avsucd": write})
