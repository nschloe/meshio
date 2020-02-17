import xml.etree.ElementTree as ET

import numpy

num_nodes_per_cell = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "quad8": 8,
    "tetra": 4,
    "hexahedron": 8,
    "hexahedron20": 20,
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
    #
    "line11": 11,
    "triangle66": 66,
    "quad121": 121,
    "tetra286": 286,
}


def cell_data_from_raw(cells, cell_data_raw):
    cs = numpy.cumsum([len(c[1]) for c in cells])[:-1]
    return {name: numpy.split(d, cs) for name, d in cell_data_raw.items()}


def raw_from_cell_data(cell_data):
    return {name: numpy.concatenate(value) for name, value in cell_data.items()}


# https://stackoverflow.com/a/30019607/353337
def CDATA(text=None):
    element = ET.Element("![CDATA[")
    element.text = text
    return element


ET._original_serialize_xml = ET._serialize_xml


def _serialize_xml(write, elem, qnames, namespaces, short_empty_elements, **kwargs):
    if elem.tag == "![CDATA[":
        write("\n<{}{}]]>\n".format(elem.tag, elem.text))
        if elem.tail:
            write(ET._escape_cdata(elem.tail))
    else:
        return ET._original_serialize_xml(
            write, elem, qnames, namespaces, short_empty_elements, **kwargs
        )


ET._serialize_xml = ET._serialize["xml"] = _serialize_xml


def write_xml(filename, root):
    tree = ET.ElementTree(root)
    tree.write(filename)


def _pick_first_int_data(data):
    # pick out material
    keys = list(data.keys())
    candidate_keys = []
    for key in keys:
        # works for point_data and cell_data
        if data[key][0].dtype in [
            numpy.int8,
            numpy.int16,
            numpy.int32,
            numpy.int64,
            numpy.uint8,
            numpy.uint16,
            numpy.uint32,
            numpy.uint64,
        ]:
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


# https://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
vtk_to_meshio_type = {
    0: "empty",
    1: "vertex",
    # 2: 'poly_vertex',
    3: "line",
    # 4: 'poly_line',
    5: "triangle",
    # 6: 'triangle_strip',
    7: "polygon",
    # 8: 'pixel',
    9: "quad",
    10: "tetra",
    # 11: 'voxel',
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
    15: "penta_prism",
    16: "hexa_prism",
    21: "line3",
    22: "triangle6",
    23: "quad8",
    24: "tetra10",
    25: "hexahedron20",
    26: "wedge15",
    27: "pyramid13",
    28: "quad9",
    29: "hexahedron27",
    30: "quad6",
    31: "wedge12",
    32: "wedge18",
    33: "hexahedron24",
    34: "triangle7",
    35: "line4",
    #
    # 60: VTK_HIGHER_ORDER_EDGE,
    # 61: VTK_HIGHER_ORDER_TRIANGLE,
    # 62: VTK_HIGHER_ORDER_QUAD,
    # 63: VTK_HIGHER_ORDER_POLYGON,
    # 64: VTK_HIGHER_ORDER_TETRAHEDRON,
    # 65: VTK_HIGHER_ORDER_WEDGE,
    # 66: VTK_HIGHER_ORDER_PYRAMID,
    # 67: VTK_HIGHER_ORDER_HEXAHEDRON,
}
meshio_to_vtk_type = {v: k for k, v in vtk_to_meshio_type.items()}
