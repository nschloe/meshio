"""
I/O for Nastran bulk data.

See <http://help.autodesk.com/view/NSTRN/2019/ENU/?guid=GUID-42B54ACB-FBE3-47CA-B8FE-475E7AD91A00>
"""
import logging

import numpy

from ..__about__ import __version__
from .._common import num_nodes_per_cell
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh

CHUNK_SIZE = 8
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
    "CTRAX6": "triangle6",
    "CTRIAX6": "triangle6",
    "CTRIA6": "triangle6",
    "CQUADR": "quad",
    "CSHEAR": "quad",
    "CQUAD4": "quad",
    "CQUAD8": "quad8",
    "CQUAD9": "quad9",
    "CTETRA": "tetra",
    "CTETRA_": "tetra10",  # fictive
    "CPYRAM": "pyramid",
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
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Skip until BEGIN BULK
    begin_bulk = False
    while not begin_bulk:
        line = f.readline()
        if not line:
            raise RuntimeError('"BEGIN BULK" statement not found')
            break
        if line.strip().startswith("BEGIN BULK"):
            begin_bulk = True
            break

    # Reading data
    points = []
    points_id = []
    cells = {}
    cells_id = {}
    cell = None
    cell_type = None
    keyword_prev = None

    def add_cell(cell, cell_type, keyword_prev):
        cell = list(map(int, cell))
        cell = _convert_to_vtk_ordering(cell, keyword_prev)

        # Treat 2nd order CTETRA, CPYRA, CPENTA, CHEXA elements
        if len(cell) > num_nodes_per_cell[cell_type]:
            assert cell_type in ["tetra", "pyramid", "wedge", "hexahedron"]
            if cell_type == "tetra":
                cell_type = "tetra10"
            elif cell_type == "pyramid":
                cell_type = "pyramid13"
            elif cell_type == "wedge":
                cell_type = "wedge15"
            elif cell_type == "hexahedron":
                cell_type = "hexahedron20"

        try:
            cells[cell_type].append(cell)
            cells_id[cell_type].append(cell_id)
        except KeyError:
            cells[cell_type] = [cell]
            cells_id[cell_type] = [cell_id]

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
        keyword = chunks[0].strip()

        # Points
        if keyword == "GRID":
            point_id = int(chunks[1])
            points_id.append(point_id)
            points.append([_nastran_float(i) for i in chunks[3:6]])

        # Cells
        elif keyword in nastran_to_meshio_type:
            # Add previous cell and cell_id
            if cell is not None:
                add_cell(cell, cell_type, keyword_prev)

            # Current cell
            cell_type = nastran_to_meshio_type[keyword]
            cell_id = int(chunks[1])

            n_nodes = num_nodes_per_cell[cell_type]
            if keyword in nastran_solid_types or n_nodes > 6:
                cell = chunks[3:]
            else:
                cell = chunks[3 : 3 + n_nodes]

            keyword_prev = keyword

        # Cells card continuation for 2nd order CTETRA, CPYRA, CPENTA, CHEXA elements
        elif keyword[0] == "+":
            assert cell is not None
            cell.extend(chunks[1:])

    # Add the last cell
    add_cell(cell, cell_type, keyword_prev)

    # Convert to numpy arrays
    points = numpy.array(points)
    points_id = numpy.array(points_id, dtype=int)
    for cell_type in cells:
        cells[cell_type] = numpy.array(cells[cell_type], dtype=int)
        cells_id[cell_type] = numpy.array(cells_id[cell_type], dtype=int)

    # Convert to natural point ordering
    # https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    points_id_dict = dict(zip(points_id, numpy.arange(len(points), dtype=int)))
    points_id_get = numpy.vectorize(points_id_dict.__getitem__)
    for cell_type in cells:
        cells[cell_type] = points_id_get(cells[cell_type])

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
        points = numpy.column_stack([mesh.points, numpy.zeros(mesh.points.shape[0])])
    else:
        points = mesh.points

    with open_file(filename, "w") as f:
        f.write(f"$ Nastran file written by meshio v{__version__}\n")
        f.write("BEGIN BULK\n")

        # Points
        for point_id, x in enumerate(points):
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
                cell_info = f"{nastran_type}, {cell_id:d},, "
                cell1 = cell + 1
                cell1 = _convert_to_nastran_ordering(cell1, nastran_type)
                conn = ", ".join(str(nid) for nid in cell1[:6])
                f.write(cell_info + conn + "\n")
                if len(cell1) > 6:
                    conn = ", ".join(str(nid) for nid in cell1[6:14])
                    f.write("+, " + conn + "\n")
                    if len(cell1) > 14:
                        conn = ", ".join(str(nid) for nid in cell1[14:])
                        f.write("+, " + conn + "\n")

        f.write("ENDDATA\n")


def _nastran_float(string):
    try:
        return float(string)
    except ValueError:
        string = string.strip()
        return float(string[0] + string[1:].replace("+", "e+").replace("-", "e-"))


def _chunk_string(string):
    string = string.strip()
    if "," in string:  # free format
        chunks = string.split(",")
    else:  # fixed format
        chunks = [
            string[0 + i : CHUNK_SIZE + i] for i in range(0, len(string), CHUNK_SIZE)
        ]
    return chunks[:9]  # the 10-th chunk is ignored


def _convert_to_vtk_ordering(cell, nastran_type):
    if nastran_type in ["CTRAX6", "CTRIAX6"]:
        cell = [cell[i] for i in [0, 2, 4, 1, 3, 5]]
    return cell


def _convert_to_nastran_ordering(cell, nastran_type):
    if nastran_type in ["CTRAX6", "CTRIAX6"]:
        cell = [cell[i] for i in [0, 3, 1, 4, 2, 5]]
    return cell


register("nastran", [".bdf", ".fem", ".nas"], read, {"nastran": write})
