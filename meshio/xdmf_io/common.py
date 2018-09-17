# -*- coding: utf-8 -*-
#
import numpy

numpy_to_xdmf_dtype = {
    "int32": ("Int", "4"),
    "int64": ("Int", "8"),
    "uint32": ("UInt", "4"),
    "uint64": ("UInt", "8"),
    "float32": ("Float", "4"),
    "float64": ("Float", "8"),
}
xdmf_to_numpy_type = {v: k for k, v in numpy_to_xdmf_dtype.items()}

dtype_to_format_string = {
    "int32": "%d",
    "int64": "%d",
    "unit32": "%d",
    "uint64": "%d",
    "float32": "%.7e",
    "float64": "%.15e",
}


# See
# <http://www.xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
# for XDMF types.
# There appears to be no particular consistency, so allow for different
# alternatives as well.
meshio_to_xdmf_type = {
    "vertex": ["Polyvertex"],
    "line": ["Polyline"],
    "triangle": ["Triangle"],
    "quad": ["Quadrilateral"],
    "tetra": ["Tetrahedron"],
    "pyramid": ["Pyramid"],
    "wedge": ["Wedge"],
    "hexahedron": ["Hexahedron"],
    "line3": ["Edge_3"],
    "triangle6": ["Triangle_6", "Tri_6"],
    "quad8": ["Quadrilateral_8", "Quad_8"],
    "tetra10": ["Tetrahedron_10", "Tet_10"],
    "pyramid13": ["Pyramid_13"],
    "wedge15": ["Wedge_15"],
    "hexahedron20": ["Hexahedron_20", "Hex_20"],
}
xdmf_to_meshio_type = {v: k for k, vals in meshio_to_xdmf_type.items() for v in vals}


# Check out
# <https://gitlab.kitware.com/xdmf/xdmf/blob/master/XdmfTopologyType.cpp>
# for the list of indices.
xdmf_idx_to_meshio_type = {
    0x1: "vertex",
    0x2: "line",
    0x4: "triangle",
    0x5: "quad",
    0x6: "tetra",
    0x7: "pyramid",
    0x8: "wedge",
    0x9: "hexahedron",
    0x22: "line3",
    0x23: "quad9",
    0x24: "triangle6",
    0x25: "quad8",
    0x26: "tetra10",
    0x27: "pyramid13",
    0x28: "wedge15",
    0x29: "wedge18",
    0x30: "hexahedron20",
    0x31: "hexahedron24",
    0x32: "hexahedron27",
    0x33: "hexahedron64",
    0x34: "hexahedron125",
    0x35: "hexahedron216",
    0x36: "hexahedron343",
    0x37: "hexahedron512",
    0x38: "hexahedron729",
    0x39: "hexahedron1000",
    0x40: "hexahedron1331",
    # 0x41: 'hexahedron_spectral_64',
    # 0x42: 'hexahedron_spectral_125',
    # 0x43: 'hexahedron_spectral_216',
    # 0x44: 'hexahedron_spectral_343',
    # 0x45: 'hexahedron_spectral_512',
    # 0x46: 'hexahedron_spectral_729',
    # 0x47: 'hexahedron_spectral_1000',
    # 0x48: 'hexahedron_spectral_1331',
}
meshio_type_to_xdmf_index = {v: k for k, v in xdmf_idx_to_meshio_type.items()}


def translate_mixed_cells(data):
    # Translate it into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (cell_type1, p0, p1, ... ,pk, cell_type2, p10, p11, ..., p1k, ...

    # http://www.xdmf.org/index.php/XDMF_Model_and_Format#Topology
    # https://gitlab.kitware.com/xdmf/xdmf/blob/master/XdmfTopologyType.hpp#L394
    xdmf_idx_to_num_nodes = {
        1: 1,  # vertex
        4: 3,  # triangle
        5: 4,  # quad
        6: 4,  # tet
        7: 5,  # pyramid
        8: 6,  # wedge
        9: 8,  # hex
        11: 6,  # triangle6
    }

    # collect types and offsets
    types = []
    offsets = []
    r = 0
    while r < len(data):
        types.append(data[r])
        offsets.append(r)
        r += xdmf_idx_to_num_nodes[data[r]] + 1

    offsets = numpy.array(offsets)

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

    cells = {}
    for tpe, b in bins.items():
        meshio_type = xdmf_idx_to_meshio_type[tpe]
        assert (data[offsets[b]] == tpe).all()
        n = xdmf_idx_to_num_nodes[tpe]
        indices = numpy.array([numpy.arange(1, n + 1) + o for o in offsets[b]])
        cells[meshio_type] = data[indices]

    return cells
