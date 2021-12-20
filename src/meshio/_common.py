from xml.etree import ElementTree as ET

import numpy as np
from rich.console import Console

# See <https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells> for the node
# ordering.
num_nodes_per_cell = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "quad8": 8,
    "tetra": 4,
    "hexahedron": 8,
    "hexahedron20": 20,
    "hexahedron24": 24,
    "wedge": 6,
    "pyramid": 5,
    #
    "line3": 3,
    "triangle6": 6,
    "quad9": 9,
    "tetra10": 10,
    "hexahedron27": 27,
    "wedge15": 15,
    "wedge18": 18,
    "pyramid13": 13,
    "pyramid14": 14,
    #
    "line4": 4,
    "triangle10": 10,
    "quad16": 16,
    "tetra20": 20,
    "wedge40": 40,
    "hexahedron64": 64,
    #
    "line5": 5,
    "triangle15": 15,
    "quad25": 25,
    "tetra35": 35,
    "wedge75": 75,
    "hexahedron125": 125,
    #
    "line6": 6,
    "triangle21": 21,
    "quad36": 36,
    "tetra56": 56,
    "wedge126": 126,
    "hexahedron216": 216,
    #
    "line7": 7,
    "triangle28": 28,
    "quad49": 49,
    "tetra84": 84,
    "wedge196": 196,
    "hexahedron343": 343,
    #
    "line8": 8,
    "triangle36": 36,
    "quad64": 64,
    "tetra120": 120,
    "wedge288": 288,
    "hexahedron512": 512,
    #
    "line9": 9,
    "triangle45": 45,
    "quad81": 81,
    "tetra165": 165,
    "wedge405": 405,
    "hexahedron729": 729,
    #
    "line10": 10,
    "triangle55": 55,
    "quad100": 100,
    "tetra220": 220,
    "wedge550": 550,
    "hexahedron1000": 1000,
    "hexahedron1331": 1331,
    #
    "line11": 11,
    "triangle66": 66,
    "quad121": 121,
    "tetra286": 286,
}


def cell_data_from_raw(cells, cell_data_raw):
    cs = np.cumsum([len(c) for c in cells])[:-1]
    return {name: np.split(d, cs) for name, d in cell_data_raw.items()}


def raw_from_cell_data(cell_data):
    return {name: np.concatenate(value) for name, value in cell_data.items()}


def write_xml(filename, root):
    tree = ET.ElementTree(root)
    tree.write(filename)


def _pick_first_int_data(data):
    # pick out material
    keys = list(data.keys())
    candidate_keys = []
    for key in keys:
        # works for point_data and cell_data
        if data[key][0].dtype.kind in ["i", "u"]:  # int or uint
            candidate_keys.append(key)

    if len(candidate_keys) > 0:
        # pick the first
        key = candidate_keys[0]
        idx = keys.index(key)
        other = keys[:idx] + keys[idx + 1 :]
    else:
        key = None
        other = []

    return key, other


def warn(string, highlight: bool = True) -> None:
    Console(stderr=True).print(
        f"[yellow][bold]Warning:[/bold] {string}[/yellow]", highlight=highlight
    )


def error(string, highlight: bool = True) -> None:
    Console(stderr=True).print(
        f"[red][bold]Error:[/bold] {string}[/red]", highlight=highlight
    )
