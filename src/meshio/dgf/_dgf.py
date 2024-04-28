"""
I/O for DUNE grid format, cf.
https://www.dune-project.org/doxygen/2.9.0/group__DuneGridFormatParser.html#details
"""

import numpy as np

from ..__about__ import __version__ as version
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

#: Maps DGF cell types to meshio cell types.
dgf_to_meshio_type = {
    ("SIMPLEX", 2): "line",
    ("SIMPLEX", 3): "triangle",
    ("CUBE", 4): "quad",
}

#: Maps DGF cell types to meshio cell types.
meshio_to_dgf_type = {
    "line": ("SIMPLEX", 2),
    "triangle": ("SIMPLEX", 3),
    "quad": ("CUBE", 4),
}

#: The DGF keywords that are supported by meshio.
dgf_keywords = [
    # Directly supported
    "VERTEX",
    "SIMPLEX",
    "CUBE",
    # TODO: Add support later!
    # Supported by storing the information on the mesh? No face information in meshio?
    # "BOUNDARYSEGMENTS",
    # "BOUNDARYDOMAIN",
    # Unsupported keywords
    # "SIMPLEXGENERATOR"
    # "GRIDPARAMETER",
    # "PERIODICFACETRANSFORMATION",
    # ...
]


def read(filename):
    """Reads a DGF file.

    Args:
        filename (PathLike): The path to the file.

    Returns:
        meshio.Mesh: The meshio mesh
    """
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def readline(f):
    """Reads a line from a dgf file and skips comments.

    Args:
       f: The file object.

    Returns:
         str: the first non-comment line, stripped of leading/trailing whitespace, and with end-of-line comment removed.
    """
    # check file is at end
    while True:
        line = f.readline()
        if not line:
            return None
        line = line.split("%")[0].strip()
        if line:
            return line


def skip_block(f):
    """Skips a block in a dgf file.

    Args:
        f: The file object.
    """
    line = readline(f)
    while line and not line.startswith("#"):
        line = readline(f)
    return


def append_cell_block(cells, cell_block, cell_data, block_data):
    """Appends a cell block to the mesh.

    Args:
        cells (list): The current list of cell blocks.
        cell_block (CellBlock): The cell block to append.
        cell_data (dict): The current cell data.
        block_data (dict): The block data to append.
    """
    cells.append(cell_block)

    for varname, variable in block_data.items():
        if varname in cell_data:
            cell_data[varname].append(variable)
        else:
            cell_data[varname] = [variable]


def read_buffer(f):
    points = []
    point_data = {}
    cells = []
    cell_data = {}

    line = readline(f)
    if not line.startswith("DGF"):
        raise ReadError("DGF Files must start with 'DGF'")

    line = readline(f)
    while line:
        if line.upper().startswith("VERTEX"):
            points, point_data = _read_vertex(f)
        elif line.upper().startswith("SIMPLEX"):
            cell_block, block_data = _read_cells(f, "SIMPLEX")
            append_cell_block(cells, cell_block, cell_data, block_data)
        elif line.upper().startswith("CUBE"):
            cell_block, block_data = _read_cells(f, "CUBE")
            append_cell_block(cells, cell_block, cell_data, block_data)
        elif line.upper().startswith("INTERVAL"):
            raise ReadError(
                "INTERVAL blocks are not supported by meshio, please use meshzoo to generate the grid."
            )
        elif line.startswith("#"):
            warn("Comment with # found. Skipping line.")
        else:
            warn(f"Unknown keyword in '{line}'. Skipping block.")
            skip_block(f)
        line = readline(f)

    return Mesh(points, cells, point_data, cell_data)


def _read_vertex(f):
    """Reads the vertex block of a DGF file.

    Args:
        f: The file object, already in the start of the vertex block.

    Returns:
        tuple[np.ndarray, dict]: The point coordinates and point data.
    """
    data = []
    num_parameters = 0
    points = []
    point_data = {}

    # Determine number of parameters defined on the points
    line = readline(f)
    if line.startswith("parameters"):
        num_parameters = int(line.split()[1])
        line = readline(f)
    else:
        num_parameters = 0

    entries_per_line = len(line.split())
    n_dim = entries_per_line - num_parameters

    # Read vertex block including point coordinates and point data
    while line:
        if line.startswith("#"):
            break
        data.append([float(x) for x in line.split()])
        line = readline(f)

    # Extract point coordinates and point data
    points = np.array([x[:n_dim] for x in data])
    if num_parameters > 0:
        point_data = {
            f"param_{i}": np.array([x[n_dim + i] for x in data], dtype=float)
            for i in range(num_parameters)
        }

    return points, point_data


def _read_cells(f, block_name):
    """Reads the simplex block of a DGF file and returns the cells and cell data.

    Args:
        f: The file object, already in the start of the simplex block.
        block_name (str): The name of the block.

    Returns:
        tuple[np.ndarray, dict]: The cells and cell data.
    """

    data = []
    num_parameters = 0

    # Determine number of parameters defined on the simplex elements
    line = readline(f)
    if line.startswith("parameters"):
        num_parameters = int(line.split()[1])
        line = readline(f)
    else:
        num_parameters = 0

    entries_per_cell_definition = len(line.split())
    n_indices = entries_per_cell_definition - num_parameters

    # Read simplex block including cell connectivity and cell data
    while line:
        if line.startswith("#"):
            break
        data.append([float(x) for x in line.split()])
        line = readline(f)

    # Determine cell type
    if block_name == "SIMPLEX":
        dgf_type = ("SIMPLEX", n_indices)
    elif block_name == "CUBE":
        dgf_type = ("CUBE", n_indices)

    # Extract cell connectivity and cell data
    cell_connectivity = np.array([x[:n_indices] for x in data], dtype=int)
    cell_block = CellBlock(dgf_to_meshio_type[dgf_type], cell_connectivity)
    cell_data = {}
    if num_parameters > 0:
        cell_data = {
            f"param_{i}": np.array([x[n_indices + i] for x in data], dtype=float)
            for i in range(num_parameters)
        }

    return cell_block, cell_data


def write(filename, mesh):
    """Writes a DGF file."""

    with open_file(filename, "w") as f:
        f.write("DGF\n")
        f.write(f'% "Written by meshio v{version}"\n')

        # Write vertex block
        f.write("VERTEX\n")
        if len(mesh.point_data) > 0:
            f.write(f"parameters {len(mesh.point_data)}\n")
            for point, data in zip(mesh.points, mesh.point_data.values()):
                f.write(
                    " ".join(str(c) for c in point)
                    + " "
                    + " ".join(str(d) for d in data)
                    + "\n"
                )
        else:
            for point in mesh.points:
                f.write(" ".join(str(c) for c in point) + "\n")
        f.write("#\n")

        # Write cell blocks
        number_of_parameters = len(mesh.cell_data)
        for cell_block_index, cell_block in enumerate(mesh.cells):
            dgf_cell_type, _ = meshio_to_dgf_type[cell_block.type]
            f.write(dgf_cell_type + "\n")
            if len(mesh.cell_data) > 0:
                f.write(f"parameters {number_of_parameters}\n")

                # grab all param values for the current cell block
                # we cant do it directly on the existing list, because it does not support slicing
                n_cells = len(cell_block.data)
                parameters = np.zeros((number_of_parameters, n_cells))
                for pi, (_, parameter_all_blocks) in enumerate(mesh.cell_data.items()):
                    parameters[pi, :] = parameter_all_blocks[cell_block_index]

                # write out all cell indices and
                for cell_index, cell in enumerate(cell_block.data):
                    f.write(
                        " ".join(str(id) for id in cell)
                        + " "
                        + " ".join(str(d) for d in parameters[:, cell_index])
                        + "\n"
                    )
            else:
                for cell in cell_block.data:
                    f.write(" ".join(str(id) for id in cell) + "\n")
            f.write("#\n")

        # Write default boundary domain
        f.write("BOUNDARYDOMAIN\n")
        f.write("default 1\n")
        f.write("#\n")


register_format("dgf", [".dgf"], read, {"dgf": write})
