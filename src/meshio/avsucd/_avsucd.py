"""
I/O for AVS-UCD format, cf.
<https://lanl.github.io/LaGriT/pages/docs/read_avs.html>.
"""
import numpy as np

from ..__about__ import __version__ as version
from .._common import _pick_first_int_data, warn
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

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


meshio_to_avsucd_order = {
    "vertex": [0],
    "line": [0, 1],
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 3, 2],
    "pyramid": [4, 0, 1, 2, 3],
    "wedge": [3, 4, 5, 0, 1, 2],
    "hexahedron": [4, 5, 6, 7, 0, 1, 2, 3],
}


avsucd_to_meshio_order = {
    k: (v if k != "pyramid" else [1, 2, 3, 4, 0])
    for k, v in meshio_to_avsucd_order.items()
}


def read(filename):
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Skip comments and unpack first line
    num_nodes, num_cells, num_node_data, num_cell_data, _ = np.genfromtxt(
        f, max_rows=1, dtype=int, comments="#"
    )

    # Read nodes
    point_ids, points = _read_nodes(f, num_nodes)

    # Read cells
    cell_ids, cells, cell_data = _read_cells(f, num_cells, point_ids)

    # Read node data
    if num_node_data:
        point_data = _read_data(f, num_nodes, point_ids)
    else:
        point_data = {}

    # Read cell data
    if num_cell_data:
        cdata = _read_data(f, num_cells, cell_ids)
        sections = np.cumsum([len(c[1]) for c in cells[:-1]])
        for k, v in cdata.items():
            cell_data[k] = np.split(v, sections)

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def _read_nodes(f, num_nodes):
    if num_nodes > 0:
        data = np.genfromtxt(f, max_rows=num_nodes)
    else:
        data = np.empty((0, 3))
    points_ids = {int(pid): i for i, pid in enumerate(data[:, 0])}
    return points_ids, data[:, 1:]


def _read_cells(f, num_cells, point_ids):
    cells = []
    cell_ids = {}
    cell_data = {"avsucd:material": []}
    count = 0
    for _ in range(num_cells):
        line = f.readline().strip().split()
        cell_id = int(line[0])
        cell_mat = int(line[1])
        cell_type = avsucd_to_meshio_type[line[2]]
        corner = [point_ids[int(pid)] for pid in line[3:]]

        if len(cells) > 0 and cells[-1][0] == cell_type:
            cells[-1][1].append(corner)
            cell_data["avsucd:material"][-1].append(cell_mat)
        else:
            cells.append((cell_type, [corner]))
            cell_data["avsucd:material"].append([cell_mat])

        cell_ids[cell_id] = count
        count += 1

    # Convert to numpy arrays
    for k, (cell_type, cdata) in enumerate(cells):
        cells[k] = CellBlock(
            cell_type, np.array(cdata)[:, avsucd_to_meshio_order[cell_type]]
        )
        cell_data["avsucd:material"][k] = np.array(cell_data["avsucd:material"][k])
    return cell_ids, cells, cell_data


def _read_data(f, num_entities, entity_ids):
    line = f.readline().strip().split()
    data_size = [int(i) for i in line[1:]]

    labels = {}
    data = {}
    for i, dsize in enumerate(data_size):
        line = f.readline().strip().split(",")
        labels[i] = line[0].strip().replace(" ", "_")
        data[labels[i]] = (
            np.empty(num_entities) if dsize == 1 else np.empty((num_entities, dsize))
        )

    for _ in range(num_entities):
        line = f.readline().strip().split()
        eid = entity_ids[int(line[0])]
        j = 0
        for i, dsize in enumerate(data_size):
            if dsize == 1:
                data[labels[i]][eid] = float(line[j + 1])
            else:
                data[labels[i]][eid] = [
                    float(val) for val in line[j + 1 : j + 1 + dsize]
                ]
            j += dsize
    return data


def write(filename, mesh):
    if len(mesh.points.shape) > 1 and mesh.points.shape[1] == 2:
        warn(
            "AVS-UCD requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])

    with open_file(filename, "w") as f:
        # Write meshio version
        f.write(f"# Written by meshio v{version}\n")

        # Write first line
        num_nodes = len(mesh.points)
        num_cells = sum(len(c.data) for c in mesh.cells)

        # Try to find an appropriate materials array
        key, other = _pick_first_int_data(mesh.cell_data)
        if key and other:
            other_string = ", ".join(other)
            warn(
                "AVS-UCD can only write one cell data array. "
                f"Picking {key}, skipping {other_string}."
            )
        material = (
            np.concatenate(mesh.cell_data[key])
            if key
            else np.zeros(num_cells, dtype=int)
        )

        num_node_data = [
            1 if v.ndim == 1 else v.shape[1] for v in mesh.point_data.values()
        ]
        num_cell_data = [
            1 if np.concatenate(v).ndim == 1 else np.concatenate(v).shape[1]
            for k, v in mesh.cell_data.items()
            if k != key
        ]
        num_node_data_sum = sum(num_node_data)
        num_cell_data_sum = sum(num_cell_data)
        f.write(f"{num_nodes} {num_cells} {num_node_data_sum} {num_cell_data_sum} 0\n")

        # Write nodes
        _write_nodes(f, mesh.points)

        # Write cells
        _write_cells(f, mesh.cells, material)

        # Write node data
        if num_node_data_sum:
            labels = mesh.point_data.keys()
            data_array = np.column_stack([v for v in mesh.point_data.values()])
            _write_data(
                f, labels, data_array, num_nodes, num_node_data, num_node_data_sum
            )

        # Write cell data
        if num_cell_data_sum:
            labels = [k for k in mesh.cell_data.keys() if k != key]
            data_array = np.column_stack(
                [np.concatenate(v) for k, v in mesh.cell_data.items() if k != key]
            )
            _write_data(
                f, labels, data_array, num_cells, num_cell_data, num_cell_data_sum
            )


def _write_nodes(f, points):
    for i, (x, y, z) in enumerate(points):
        f.write(f"{i + 1} {x} {y} {z}\n")


def _write_cells(f, cells, material):
    i = 0
    for cell_block in cells:
        cell_type = cell_block.type
        v = cell_block.data
        for cell in v[:, meshio_to_avsucd_order[cell_type]]:
            cell_str = " ".join(str(c) for c in cell + 1)
            f.write(
                f"{i + 1} {material[i]} {meshio_to_avsucd_type[cell_type]} {cell_str}\n"
            )
            i += 1


def _write_data(f, labels, data_array, num_entities, num_data, num_data_sum):
    num_data_str = " ".join(str(i) for i in num_data)
    f.write(f"{len(num_data)} {num_data_str}\n")

    for label in labels:
        f.write(f"{label}, real\n")

    data_array = np.column_stack((np.arange(1, num_entities + 1), data_array))
    np.savetxt(f, data_array, delimiter=" ", fmt=["%d"] + ["%.14e"] * num_data_sum)


register_format("avsucd", [".avs"], read, {"avsucd": write})
