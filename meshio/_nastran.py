"""
I/O for Nastran bulk data.

See <http://help.autodesk.com/view/NSTRN/2019/ENU/?guid=GUID-42B54ACB-FBE3-47CA-B8FE-475E7AD91A00>
"""
import logging

import numpy

from .__about__ import __version__
from ._common import num_nodes_per_cell
from ._mesh import Mesh

nastran_to_meshio_type = {
    "CELAS1": "vertex",
    "CBEAM": "line",
    "CBUSH": "line",
    "CBUSH1D": "line",
    "CROD": "line",
    "CGAP": "line",
    "CBAR": "line",
    "CTRIAR": "triangle",
    "CTRIA3": "triangle",
    "CTRIAX6": "triangle6",
    "CTRIA6": "triangle6",
    "CQUADR": "quad",
    "CSHEAR": "quad",
    "CQUAD4": "quad",
    "CQUAD8": "quad8",
    "CTETRA": "tetra",
    "CTETRA_": "tetra10",  # fictive
    "CPYRA": "pyramid",
    "CPYRA_": "pyramid13",  # fictive
    "CPENTA": "wedge",
    "CPENTA_": "wedge15",  # fictive
    "CHEXA": "hexahedron",
    "CHEXA_": "hexahedron20",  # fictive
}
nastran_solid_types = ["CTETRA", "CPYRA", "CPENTA", "CHEXA"]
meshio_to_nastran_type = {v: k for k, v in nastran_to_meshio_type.items()}


def read(filename):
    with open(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Skip until BEGIN BULK
    begin_bulk = False
    while not begin_bulk:
        line = f.readline()
        if line.strip().startswith("BEGIN BULK"):
            begin_bulk = True
            break
    else:
        raise RuntimeError('"BEGIN BULK" statement not found')

    # Reading data
    points = []
    points_id = []
    points_id_dict = None
    cells = {}
    cells_id = {}
    while True:
        line = f.readline()
        if not line:
            break

        # End loop when ENDDATA detected
        line = line.strip()
        if line.startswith("ENDDATA"):
            break

        # Blank lines or comments
        if len(line) < 4 or line.startswith(("$", "//", "#")):
            continue

        chunks = _chunk_string(line)
        keyword = chunks[0]

        # Points
        if keyword == "GRID":
            point_id = int(chunks[1])
            points_id.append(point_id)
            points.append([_nastran_float(i) for i in chunks[3:6]])

        # Cells
        elif keyword in nastran_to_meshio_type:
            if points_id_dict is None:
                points_id_dict = dict(zip(points_id, range(len(points))))
            cell_type = nastran_to_meshio_type[keyword]

            # For solid elements, check if it corresponds to a 2nd-order element
            if keyword in nastran_solid_types:
                cell_type = _determine_solid_2nd(f, cell_type)

            cell_id = int(chunks[1])
            if cell_type in cells_id:
                cells_id[cell_type].append(cell_id)
            else:
                cells_id[cell_type] = [cell_id]

            n_nodes = num_nodes_per_cell[cell_type]
            if n_nodes <= 6:
                cell = [points_id_dict[int(i)] for i in chunks[3 : 3 + n_nodes]]
            else:
                cell = [points_id_dict[int(i)] for i in chunks[3:]]
                chunks = _chunk_string(f.readline())
                cell.extend([points_id_dict[int(i)] for i in chunks[1:]])
                if n_nodes >= 15:
                    chunks = _chunk_string(f.readline())
                    cell.extend([points_id_dict[int(i)] for i in chunks[1:]])

            if cell_type in cells:
                cells[cell_type].append(cell)
            else:
                cells[cell_type] = [cell]

    points = numpy.array(points)
    points_id = numpy.array(points_id, dtype=int)
    for key in cells:
        cells[key] = numpy.array(cells[key], dtype=int)
        cells_id[key] = numpy.array(cells_id[key], dtype=int)

    # Construct the mesh object
    mesh = Mesh(points, cells)
    mesh.points_id = points_id
    mesh.cells_id = cells_id
    return mesh


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        logging.warning(
            "Nastran requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.hstack([mesh.points, numpy.zeros((len(mesh.points), 1))])

    with open(filename, "w") as f:
        f.write("$ Nastran file written by meshio v{}\n".format(__version__))
        f.write("BEGIN BULK\n")

        # Points
        for point_id, x in enumerate(mesh.points):
            f.write(
                "GRID, {:d},, {:.15e}, {:.15e}, {:.15e}\n".format(
                    point_id + 1, x[0], x[1], x[2]
                )
            )

        # Cells
        cell_id = 0
        for cell_type, cells in mesh.cells.items():
            nastran_type = meshio_to_nastran_type[cell_type].replace("_", "")
            for cell in cells:
                cell_id += 1
                cell_info = "{}, {:d},, ".format(nastran_type, cell_id)
                conn = ", ".join(str(nid + 1) for nid in cell[:6])
                f.write(cell_info + conn + "\n")
                if len(cell) >= 7:
                    conn = ", ".join(str(nid + 1) for nid in cell[6:14])
                    f.write("+, " + conn + "\n")
                    if len(cell) >= 15:
                        conn = ", ".join(str(nid + 1) for nid in cell[14:])
                        f.write("+, " + conn + "\n")

        f.write("ENDDATA\n")


def _nastran_float(string):
    string = string.strip()
    try:
        return float(string)
    except ValueError:
        return float(string[0] + string[1:].replace("+", "e+").replace("-", "e-"))


def _chunk_string(string):
    string = string.strip()
    if "," in string:  # free format
        chunks = [chunk.strip() for chunk in string.split(",")]
    else:  # fixed format
        chunk_size = 8
        chunks = [
            string[0 + i : chunk_size + i].strip()
            for i in range(0, len(string), chunk_size)
        ]
    return chunks[:9]  # the 10-th chunk is ignored


def _determine_solid_2nd(f, cell_type):
    element_lines = 1
    curr_pos = f.tell()
    if f.readline()[0] == "+":
        element_lines += 1
        if f.readline()[0] == "+":
            element_lines += 1
    f.seek(curr_pos)

    if element_lines == 2:
        if cell_type == "tetra":
            cell_type = "tetra10"
        elif cell_type == "pyramid":
            cell_type = "pyramid13"
    else:
        if cell_type == "wedge":
            cell_type = "wedge15"
        elif cell_type == "hexahedron":
            cell_type = "hexahedron20"
    return cell_type
