"""
I/O for Tecplot ASCII data format, cf.
<https://github.com/su2code/SU2/raw/master/externals/tecio/360_data_format_guide.pdf>,
<http://paulbourke.net/dataformats/tp/>.
"""
import numpy as np

from ..__about__ import __version__ as version
from .._common import warn
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register_format
from .._mesh import Mesh

zone_key_to_type = {
    "T": str,
    "I": int,
    "J": int,
    "K": int,
    "N": int,
    "NODES": int,
    "E": int,
    "ELEMENTS": int,
    "F": str,
    "ET": str,
    "DATAPACKING": str,
    "ZONETYPE": str,
    "NV": int,
    "VARLOCATION": str,
}


# 0=ORDERED
# 1=FELINESEG
# 2=FETRIANGLE
# 3=FEQUADRILATERAL
# 4=FETETRAHEDRON
# 5=FEBRICK
# 6=FEPOLYGON
# 7=FEPOLYHEDRON
tecplot_to_meshio_type = {
    "LINESEG": "line",
    "FELINESEG": "line",
    "TRIANGLE": "triangle",
    "FETRIANGLE": "triangle",
    "QUADRILATERAL": "quad",
    "FEQUADRILATERAL": "quad",
    "TETRAHEDRON": "tetra",
    "FETETRAHEDRON": "tetra",
    "BRICK": "hexahedron",
    "FEBRICK": "hexahedron",
}


meshio_to_tecplot_type = {
    "line": "FELINESEG",
    "triangle": "FETRIANGLE",
    "quad": "FEQUADRILATERAL",
    "tetra": "FETETRAHEDRON",
    "pyramid": "FEBRICK",
    "wedge": "FEBRICK",
    "hexahedron": "FEBRICK",
}


meshio_only = set(meshio_to_tecplot_type.keys())


meshio_to_tecplot_order = {
    "line": [0, 1],
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 2, 3, 4, 4, 4, 4],
    "wedge": [0, 1, 4, 3, 2, 2, 5, 5],
    "hexahedron": [0, 1, 2, 3, 4, 5, 6, 7],
}


meshio_to_tecplot_order_2 = {
    "triangle": [0, 1, 2, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 2, 3, 3, 3, 3],
    "pyramid": [0, 1, 2, 3, 4, 4, 4, 4],
    "wedge": [0, 1, 4, 3, 2, 2, 5, 5],
    "hexahedron": [0, 1, 2, 3, 4, 5, 6, 7],
}


meshio_type_to_ndim = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "pyramid": 3,
    "wedge": 3,
    "hexahedron": 3,
}


def read(filename):
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def readline(f):
    line = f.readline().strip()
    while line.startswith("#"):
        line = f.readline().strip()

    return line


def read_buffer(f):
    variables = None
    num_data = None
    zone_format = None
    zone_type = None
    is_cell_centered = None
    data = None
    cells = None

    while True:
        line = readline(f)

        if line.upper().startswith("VARIABLES"):
            # Multilines for VARIABLES appears to work only if
            # variable name is double quoted
            lines = [line]
            i = f.tell()
            line = readline(f).upper()
            while True:
                if line.startswith('"'):
                    lines += [line]
                    i = f.tell()
                    line = readline(f).upper()
                else:
                    f.seek(i)
                    break
            line = " ".join(lines)
            variables = _read_variables(line)

        elif line.upper().startswith("ZONE"):
            # ZONE can be defined on several lines e.g.
            # ```
            # ZONE NODES = 62533, ELEMENTS = 57982
            # , DATAPACKING = BLOCK, ZONETYPE = FEQUADRILATERAL
            # , VARLOCATION = ([1-2] = NODAL, [3-7] = CELLCENTERED)
            # ```
            # is valid (and understood by ParaView and VisIt).
            info_lines = [line]
            i = f.tell()
            line = readline(f).upper()
            while True:
                # check if the first entry can be converted to a float
                try:
                    float(line.split()[0])
                except ValueError:
                    info_lines += [line]
                    i = f.tell()
                    line = readline(f).upper()
                else:
                    f.seek(i)
                    break
            line = " ".join(info_lines)

            assert variables is not None

            zone = _read_zone(line)
            (
                num_nodes,
                num_cells,
                zone_format,
                zone_type,
                is_cell_centered,
            ) = _parse_fezone(zone, variables)

            num_data = [num_cells if i else num_nodes for i in is_cell_centered]
            data, cells = _read_zone_data(
                f,
                sum(num_data) if zone_format == "FEBLOCK" else num_nodes,
                num_cells,
                zone_format,
            )

            break  # Only support one zone, no need to read the rest

        elif not line:
            break

    assert num_data is not None
    assert zone_format is not None
    assert zone_type is not None
    assert variables is not None
    assert is_cell_centered is not None
    assert data is not None
    assert cells is not None

    data = (
        np.split(np.concatenate(data), np.cumsum(num_data[:-1]))
        if zone_format == "FEBLOCK"
        else np.transpose(data)
    )
    data = {k: v for k, v in zip(variables, data)}

    point_data, cell_data = {}, {}
    for i, variable in zip(is_cell_centered, variables):
        if i:
            cell_data[variable] = [data[variable]]
        else:
            point_data[variable] = data[variable]

    x = "X" if "X" in point_data.keys() else "x"
    y = "Y" if "Y" in point_data.keys() else "y"
    z = "Z" if "Z" in point_data.keys() else "z" if "z" in point_data.keys() else ""
    points = np.column_stack((point_data.pop(x), point_data.pop(y)))
    if z:
        points = np.column_stack((points, point_data.pop(z)))
    cells = [(tecplot_to_meshio_type[zone_type], cells - 1)]

    return Mesh(points, cells, point_data, cell_data)


def _read_variables(line):
    # Gather variables in a list
    line = line.split("=")[1]
    line = [x for x in line.replace(",", " ").split()]
    variables = []

    i = 0
    while i < len(line):
        if '"' in line[i] and not (line[i].startswith('"') and line[i].endswith('"')):
            var = f"{line[i]}_{line[i + 1]}"
            i += 1
        else:
            var = line[i]

        variables.append(var.replace('"', ""))
        i += 1

    # Check that at least X and Y are defined
    if "X" not in variables and "x" not in variables:
        raise ReadError("Variable 'X' not found")
    if "Y" not in variables and "y" not in variables:
        raise ReadError("Variable 'Y' not found")

    return variables


def _read_zone(line):
    # Gather zone entries in a dict
    line = line[5:]
    zone = {}

    # Look for zone title
    ivar = line.find('"')

    # If zone contains a title, process it and save the title
    if ivar >= 0:
        i1, i2 = ivar, ivar + line[ivar + 1 :].find('"') + 2
        zone_title = line[i1 + 1 : i2 - 1]
        line = line.replace(line[i1:i2], "PLACEHOLDER")
    else:
        zone_title = None

    # Look for VARLOCATION (problematic since it contains both ',' and '=')
    ivar = line.find("VARLOCATION")

    # If zone contains VARLOCATION, process it and remove the key/value pair
    if ivar >= 0:
        i1, i2 = line.find("("), line.find(")")
        zone["VARLOCATION"] = line[i1 : i2 + 1].replace(" ", "")
        line = line[:ivar] + line[i2 + 1 :]

    # Split remaining key/value pairs separated by '='
    line = [x for x in line.replace(",", " ").split() if x != "="]
    i = 0
    while i < len(line):
        if "=" in line[i]:
            if not (line[i].startswith("=") or line[i].endswith("=")):
                key, value = line[i].split("=")
            else:
                key = line[i].replace("=", "")
                value = line[i + 1]
                i += 1
        else:
            key = line[i]
            value = line[i + 1].replace("=", "")
            i += 1

        zone[key] = zone_key_to_type[key](value)
        i += 1

    # Add zone title to zone dict
    if zone_title:
        zone["T"] = zone_title

    return zone


def _parse_fezone(zone, variables):
    # Check that the grid is unstructured
    if "F" in zone.keys():
        if zone["F"] not in {"FEPOINT", "FEBLOCK"}:
            raise ReadError("Tecplot reader can only read finite-element type grids")
        if "ET" not in zone.keys():
            raise ReadError("Element type 'ET' not found")
        zone_format = zone.pop("F")
        zone_type = zone.pop("ET")
    elif "DATAPACKING" in zone.keys():
        if "ZONETYPE" not in zone.keys():
            raise ReadError("Zone type 'ZONETYPE' not found")
        zone_format = "FE" + zone.pop("DATAPACKING")
        zone_type = zone.pop("ZONETYPE")
    else:
        raise ReadError("Data format 'F' or 'DATAPACKING' not found")

    # Number of nodes
    if "N" in zone.keys():
        num_nodes = zone.pop("N")
    elif "NODES" in zone.keys():
        num_nodes = zone.pop("NODES")
    else:
        raise ReadError("Number of nodes not found")

    # Number of elements
    if "E" in zone.keys():
        num_cells = zone.pop("E")
    elif "ELEMENTS" in zone.keys():
        num_cells = zone.pop("ELEMENTS")
    else:
        raise ReadError("Number of elements not found")

    # Variable locations
    is_cell_centered = np.zeros(len(variables), dtype=int)
    if zone_format == "FEBLOCK":
        if "NV" in zone.keys():
            node_value = zone.pop("NV")
            is_cell_centered[node_value:] = 1
        elif "VARLOCATION" in zone.keys():
            varlocation = zone.pop("VARLOCATION")[1:-1].split(",")
            for location in varlocation:
                varrange, varloc = location.split("=")
                varloc = varloc.strip()
                if varloc == "CELLCENTERED":
                    varrange = varrange[1:-1].split("-")
                    if len(varrange) == 1:
                        i = int(varrange[0]) - 1
                        is_cell_centered[i] = 1
                    else:
                        imin = int(varrange[0]) - 1
                        imax = int(varrange[1]) - 1
                        for i in range(imin, imax + 1):
                            is_cell_centered[i] = 1

    return num_nodes, num_cells, zone_format, zone_type, is_cell_centered


def _read_zone_data(f, num_data, num_cells, zone_format):
    data, count = [], 0
    while count < num_data:
        line = readline(f).split()
        if line:
            data += [[float(x) for x in line]]
            count += len(line) if zone_format == "FEBLOCK" else 1

    cells, count = [], 0
    while count < num_cells:
        line = readline(f).split()
        if line:
            cells += [[[int(x) for x in line]]]
            count += 1

    return data, np.concatenate(cells)


def write(filename, mesh):
    # Check cell types
    cell_types = []
    cell_blocks = []
    for ic, c in enumerate(mesh.cells):
        if c.type in meshio_only:
            cell_types.append(c.type)
            cell_blocks.append(ic)
        else:
            warn(
                (
                    "Tecplot does not support cell type '{}'. "
                    "Skipping cell block {}."
                ).format(c.type, ic)
            )

    # Define cells and zone type
    cell_types = np.unique(cell_types)
    if len(cell_types) == 0:
        raise WriteError("No cell type supported by Tecplot in mesh")
    elif len(cell_types) == 1:
        # Nothing much to do except converting pyramids and wedges to hexahedra
        zone_type = meshio_to_tecplot_type[cell_types[0]]
        cells = np.concatenate(
            [
                mesh.cells[ic].data[:, meshio_to_tecplot_order[mesh.cells[ic].type]]
                for ic in cell_blocks
            ]
        )
    else:
        # Check if the mesh contains 2D and 3D cells
        num_dims = [meshio_type_to_ndim[mesh.cells[ic].type] for ic in cell_blocks]

        # Skip 2D cells if it does
        if len(np.unique(num_dims)) == 2:
            warn("Mesh contains 2D and 3D cells. Skipping 2D cells.")
            cell_blocks = [ic for ic, ndim in zip(cell_blocks, num_dims) if ndim == 3]

        # Convert 2D cells to quads / 3D cells to hexahedra
        zone_type = "FEQUADRILATERAL" if num_dims[0] == 2 else "FEBRICK"
        cells = np.concatenate(
            [
                mesh.cells[ic].data[:, meshio_to_tecplot_order_2[mesh.cells[ic].type]]
                for ic in cell_blocks
            ]
        )

    # Define variables
    variables = ["X", "Y"]
    data = [mesh.points[:, 0], mesh.points[:, 1]]
    varrange = [3, 0]

    if mesh.points.shape[1] == 3:
        variables += ["Z"]
        data += [mesh.points[:, 2]]
        varrange[0] += 1

    for k, v in mesh.point_data.items():
        if k not in {"X", "Y", "Z", "x", "y", "z"}:
            if v.ndim == 1:
                variables += [k]
                data += [v]
                varrange[0] += 1
            elif v.ndim == 2:
                for i, vv in enumerate(v.T):
                    variables += [f"{k}_{i}"]
                    data += [vv]
                    varrange[0] += 1
        else:
            warn(f"Skipping point data '{k}'.")

    if mesh.cell_data:
        varrange[1] = varrange[0] - 1
        for k, v in mesh.cell_data.items():
            if k not in {"X", "Y", "Z", "x", "y", "z"}:
                v = np.concatenate([v[ic] for ic in cell_blocks])
                if v.ndim == 1:
                    variables += [k]
                    data += [v]
                    varrange[1] += 1
                elif v.ndim == 2:
                    for i, vv in enumerate(v.T):
                        variables += [f"{k}_{i}"]
                        data += [vv]
                        varrange[1] += 1
            else:
                warn(f"Skipping cell data '{k}'.")

    with open_file(filename, "w") as f:
        # Title
        f.write(f'TITLE = "Written by meshio v{version}"\n')

        # Variables
        variables_str = ", ".join(f'"{var}"' for var in variables)
        f.write(f"VARIABLES = {variables_str}\n")

        # Zone record
        num_nodes = len(mesh.points)
        num_cells = sum(len(mesh.cells[ic].data) for ic in cell_blocks)
        f.write(f"ZONE NODES = {num_nodes}, ELEMENTS = {num_cells},\n")
        f.write(f"DATAPACKING = BLOCK, ZONETYPE = {zone_type}")
        if varrange[0] <= varrange[1]:
            f.write(",\n")
            varlocation_str = (
                f"{varrange[0]}"
                if varrange[0] == varrange[1]
                else f"{varrange[0]}-{varrange[1]}"
            )
            f.write(f"VARLOCATION = ([{varlocation_str}] = CELLCENTERED)\n")
        else:
            f.write("\n")

        # Zone data
        for arr in data:
            _write_table(f, arr)

        # CellBlock
        for cell in cells:
            f.write(" ".join(str(c) for c in cell + 1) + "\n")


def _write_table(f, data, ncol=20):
    nrow = len(data) // ncol
    lines = np.split(data, np.full(nrow, ncol).cumsum())
    for line in lines:
        if len(line):
            f.write(" ".join(str(l) for l in line) + "\n")


register_format("tecplot", [".dat", ".tec"], read, {"tecplot": write})
