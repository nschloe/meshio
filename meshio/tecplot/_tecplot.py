"""
I/O for Tecplot ASCII data format, cf.
<http://paulbourke.net/dataformats/tp/>.
"""
import numpy

from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


zone_key_to_type = {
    "N": int,
    "NODES": int,
    "E": int,
    "ELEMENTS": int,
    "F": str,
    "ET": str,
    "DATAPACKING": str,
    "ZONETYPE": str,
    "VARLOCATION": str,
}


tecplot_to_meshio_type = {
    "TRIANGLE": "triangle",
    "FETRIANGLE": "triangle",
    "QUADRILATERAL": "quad",
    "FEQUADRILATERAL": "quad",
    "TETRAHEDRON": "tetra",
    "FETETRAHEDRON": "tetra",
    "BRICK": "hexahedron",
    "FEBRICK": "hexahedron",
}


def read(filename):
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    while True:
        line = f.readline().rstrip().upper()

        if line.startswith("VARIABLES"):
            variables = _read_variables(line)
        elif line.startswith("ZONE"):
            # ZONE can be defined on several lines e.g.
            # ````
            # ZONE NODES = 62533 , ELEMENTS = 57982
            # , DATAPACKING=BLOCK, ZONETYPE=FEQUADRILATERAL
            # , VARLOCATION=([1-2]=NODAL, [3-7]=CELLCENTERED)
            # ````
            # is valid (and understood by Paraview and VisIt).
            lines = [line]
            i = f.tell()
            line = f.readline().rstrip().upper()
            while True:
                if not line[0].isdigit():
                    lines += [line]
                    i = f.tell()
                    line = f.readline().rstrip().upper()
                else:
                    f.seek(i)
                    break
            line = "".join(lines)

            num_nodes, num_cells, zone_format, zone_type, is_cell_centered = _read_zone(line, variables)

            if zone_format == "FEBLOCK":
                num_data = [num_cells if i else num_nodes for i in is_cell_centered]
                data, cells = _read_zone_block(f, sum(num_data), num_cells)
            elif zone_format == "FEPOINT":
                raise ReadError(
                    "Tecplot FEPOINT format reader is not implemented yet"
                )

            break   # Only support one zone, no need to read the rest

    data = numpy.split(numpy.concatenate(data), numpy.cumsum(num_data[:-1]))
    data = {k: v for k, v in zip(variables, data)}

    point_data, cell_data = {}, {}
    for i, variable in zip(is_cell_centered, variables):
        if i:
            cell_data[variable] = [data[variable]]
        else:
            point_data[variable] = data[variable]

    points = numpy.column_stack((point_data.pop("X"), point_data.pop("Y")))
    if "Z" in point_data.keys():
        points = numpy.column_stack((points, point_data.pop("Z")))
    cells = [(tecplot_to_meshio_type[zone_type], cells-1)]

    return Mesh(points, cells, point_data, cell_data)
        
        
def _read_variables(line):
    # Gather variables in a list
    line = line.split("=")[1].split(",")
    variables = [str(var).replace('"', "").strip() for var in line]

    # Check that at least X and Y are defined
    if "X" not in variables:
        raise ReadError("Variable 'X' not found")
    if "Y" not in variables:
        raise ReadError("Variable 'Y' not found")

    return variables


def _read_zone(line, variables):
    # Gather zone entries in a dict
    # We can only process the zone record character by character due to
    # value of VARLOCATION containing both comma and equality characters.
    line = line[5:]
    zone = {}
    i = 0
    key, value, read_key = "", "", True
    is_varlocation, is_end = False, False
    while True:
        char = line[i] if line[i] != " " else ""

        if char == "=":
            read_key = False
            is_varlocation = key == "VARLOCATION"

        if is_varlocation:
            i += 1
            while True:
                char = line[i] if line[i] != " " else ""
                value += char
                if line[i] == ")":
                    break
                else:
                    i += 1
            is_varlocation, is_end = False, True
            i += 1  # Skip comma
        else:
            if char != ",":
                if char != "=":
                    if read_key:
                        key += char
                    else:
                        value += char
            else:
                is_end = True
        
        if is_end:
            if key in zone_key_to_type.keys():
                zone[key] = zone_key_to_type[key](value)
            key, value, read_key = "", "", True
            is_end = False

        if i >= len(line)-1:
            zone[key] = value
            key, value, read_key = "", "", True
            break
        else:
            i += 1

    # Check that the grid is unstructured
    if "F" in zone.keys():
        if zone["F"] not in {"FEPOINT", "FEBLOCK"}:
            raise ReadError(
                "Tecplot reader can only read finite-element type grids"
            )
        if "ET" not in zone.keys():
            raise ReadError(
                "Element type 'ET' not found"
            )
        zone_format = zone.pop("F")
        zone_type = zone.pop("ET")
    elif "DATAPACKING" in zone.keys():
        if "ZONETYPE" not in zone.keys():
            raise ReadError(
                "Zone type 'ZONETYPE' not found"
            )
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
    is_cell_centered = numpy.zeros(len(variables), dtype=int)
    if "VARLOCATION" in zone.keys():
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
                    for i in range(imin, imax+1):
                        is_cell_centered[i] = 1

    return num_nodes, num_cells, zone_format, zone_type, is_cell_centered


def _read_zone_block(f, num_data, num_cells):
    data, count = [], 0
    while count < num_data:
        line = f.readline().rstrip().split()
        if line:
            data += [[float(x) for x in line]]
            count += len(line)

    cells, count = [], 0
    while count < num_cells:
        line = f.readline().rstrip().split()
        if line:
            cells += [[[int(x) for x in line]]]
            count += 1
    
    return data, numpy.concatenate(cells)


def write(filename, mesh):
    pass


register("tecplot", [".dat"], read, {"tecplot": write})
