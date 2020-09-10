"""
I/O for DOLFIN's XML format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/dolfin_xml/dolfin_xml.html>.
"""
import logging
import os
import pathlib
import re
from xml.etree import ElementTree as ET

import numpy

from .._exceptions import ReadError, WriteError
from .._helpers import register
from .._mesh import Mesh


def _read_mesh(filename):
    dolfin_to_meshio_type = {"triangle": ("triangle", 3), "tetrahedron": ("tetra", 4)}

    # Use iterparse() to avoid loading the entire file via parse(). iterparse()
    # allows to discard elements (via clear()) after they have been processed.
    # See <https://stackoverflow.com/a/326541/353337>.
    for event, elem in ET.iterparse(filename, events=("start", "end")):
        if event == "end":
            continue

        if elem.tag == "dolfin":
            # Don't be too strict with the assertion. Some mesh files don't have the
            # proper tags.
            # assert elem.attrib['nsmap'] \
            #     == '{\'dolfin\': \'https://fenicsproject.org/\'}'
            pass
        elif elem.tag == "mesh":
            dim = int(elem.attrib["dim"])
            cell_type, num_nodes_per_cell = dolfin_to_meshio_type[
                elem.attrib["celltype"]
            ]
            cell_tags = [f"v{i}" for i in range(num_nodes_per_cell)]
        elif elem.tag == "vertices":
            points = numpy.empty((int(elem.attrib["size"]), dim))
            keys = ["x", "y"]
            if dim == 3:
                keys += ["z"]
        elif elem.tag == "vertex":
            k = int(elem.attrib["index"])
            points[k] = [elem.attrib[key] for key in keys]
        elif elem.tag == "cells":
            cells = [
                (
                    cell_type,
                    numpy.empty(
                        (int(elem.attrib["size"]), num_nodes_per_cell), dtype=int
                    ),
                )
            ]
        elif elem.tag in ["triangle", "tetrahedron"]:
            k = int(elem.attrib["index"])
            cells[0][1][k] = [elem.attrib[t] for t in cell_tags]
        else:
            logging.warning("Unknown entry %s. Ignoring.", elem.tag)

        elem.clear()

    return points, cells, cell_type


def _read_cell_data(filename, cell_type):
    dolfin_type_to_numpy_type = {
        "int": numpy.dtype("int"),
        "float": numpy.dtype("float"),
        "uint": numpy.dtype("uint"),
    }

    cell_data = {}
    dir_name = pathlib.Path(filename).resolve().parent

    # Loop over all files in the same directory as `filename`.
    basename = os.path.splitext(os.path.basename(filename))[0]
    # TODO remove .as_posix when requiring Python 3.6
    for f in os.listdir(dir_name.as_posix()):
        # Check if there are files by the name "<filename>_*.xml"; if yes,
        # extract the * pattern and make it the name of the data set.
        out = re.match(f"{basename}_([^\\.]+)\\.xml", f)
        if not out:
            continue
        name = out.group(1)

        parser = ET.XMLParser()
        tree = ET.parse((dir_name / f).as_posix(), parser)
        root = tree.getroot()

        mesh_functions = list(root)
        if len(mesh_functions) != 1:
            raise ReadError("Can only handle one mesh function")
        mesh_function = mesh_functions[0]

        if mesh_function.tag != "mesh_function":
            raise ReadError()
        size = int(mesh_function.attrib["size"])
        dtype = dolfin_type_to_numpy_type[mesh_function.attrib["type"]]
        data = numpy.empty(size, dtype=dtype)
        for child in mesh_function:
            if child.tag != "entity":
                raise ReadError()
            idx = int(child.attrib["index"])
            data[idx] = child.attrib["value"]

        if name not in cell_data:
            cell_data[name] = []
        cell_data[name].append(data)

    return cell_data


def read(filename):
    points, cells, cell_type = _read_mesh(filename)
    cell_data = _read_cell_data(filename, cell_type)
    return Mesh(points, cells, cell_data=cell_data)


def _write_mesh(filename, points, cell_type, cells):
    stripped_cells = [c for c in cells if c.type == cell_type]

    meshio_to_dolfin_type = {"triangle": "triangle", "tetra": "tetrahedron"}

    if any(c.type != cell_type for c in cells):
        discarded_cell_types = {c.type for c in cells if c.type != cell_type}
        logging.warning(
            "DOLFIN XML can only handle one cell type at a time. "
            "Using %s, discarding %s.",
            cell_type,
            ", ".join(discarded_cell_types),
        )

    dim = points.shape[1]
    if dim not in [2, 3]:
        raise WriteError(f"Can only write dimension 2, 3, got {dim}.")

    coord_names = ["x", "y"]
    if dim == 3:
        coord_names += ["z"]

    with open(filename, "w") as f:
        f.write("<dolfin nsmap=\"{'dolfin': 'https://fenicsproject.org/'}\">\n")
        ct = meshio_to_dolfin_type[cell_type]
        f.write(f'  <mesh celltype="{ct}" dim="{dim}">\n')

        num_points = len(points)
        f.write(f'    <vertices size="{num_points}">\n')
        for idx, point in enumerate(points):
            s = " ".join(f'{xyz}="{p}"' for xyz, p in zip("xyz", point))
            f.write(f'      <vertex index="{idx}" {s} />\n')
        f.write("    </vertices>\n")

        num_cells = 0
        for c in stripped_cells:
            num_cells += len(c.data)

        f.write(f'    <cells size="{num_cells}">\n')
        idx = 0
        for ct, cls in stripped_cells:
            type_string = meshio_to_dolfin_type[ct]
            for cell in cls:
                s = " ".join(f'v{k}="{c}"' for k, c in enumerate(cell))
                f.write(f'      <{type_string} index="{idx}" {s} />\n')
                idx += 1
        f.write("    </cells>\n")
        f.write("  </mesh>\n")
        f.write("</dolfin>")


def _numpy_type_to_dolfin_type(dtype):
    types = {
        "int": [numpy.int8, numpy.int16, numpy.int32, numpy.int64],
        "uint": [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64],
        "float": [numpy.float16, numpy.float32, numpy.float64],
    }
    for key, numpy_types in types.items():
        for numpy_type in numpy_types:
            if numpy.issubdtype(dtype, numpy_type):
                return key

    raise WriteError("Could not convert NumPy data type to DOLFIN data type.")


def _write_cell_data(filename, dim, cell_data):
    dolfin = ET.Element("dolfin", nsmap={"dolfin": "https://fenicsproject.org/"})

    mesh_function = ET.SubElement(
        dolfin,
        "mesh_function",
        type=_numpy_type_to_dolfin_type(cell_data.dtype),
        dim=str(dim),
        size=str(len(cell_data)),
    )

    for k, value in enumerate(cell_data):
        ET.SubElement(mesh_function, "entity", index=str(k), value=repr(value))

    tree = ET.ElementTree(dolfin)
    tree.write(filename)


def write(filename, mesh):
    logging.warning("DOLFIN XML is a legacy format. Consider using XDMF instead.")

    if any("tetra" == c.type for c in mesh.cells):
        cell_type = "tetra"
    elif any("triangle" == c.type for c in mesh.cells):
        cell_type = "triangle"
    else:
        raise WriteError(
            "DOLFIN XML only supports triangles and tetrahedra. "
            "Consider using XDMF instead."
        )

    _write_mesh(filename, mesh.points, cell_type, mesh.cells)

    for name, lst in mesh.cell_data.items():
        for data in lst:
            fname = os.path.splitext(filename)[0]
            cell_data_filename = f"{fname}_{name}.xml"
            dim = 2 if mesh.points.shape[1] == 2 or all(mesh.points[:, 2] == 0) else 3
            _write_cell_data(cell_data_filename, dim, numpy.array(data))


register("dolfin-xml", [".xml"], read, {"dolfin-xml": write})
