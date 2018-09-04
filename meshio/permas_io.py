# -*- coding: utf-8 -*-
#
"""
I/O for PERMAS dat format
"""
import gzip
import logging
import re

import numpy

from .__about__ import __version__, __website__
from .mesh import Mesh


def read(filename):
    """Reads a (compressed) PERMAS dato or post file.
    """
    # The format is specified at
    # <http://www.intes.de>.

    cells = {}
    meshio_to_permas_type = {
        "vertex": (1, "PLOT1"),
        "line": (2, "PLOTL2"),
        "triangle": (3, "TRIA3"),
        "quad": (4, "QUAD4"),
        "tetra": (4, "TET4"),
        "hexahedron": (8, "HEXE8"),
        "wedge": (6, "PENTA6"),
        "pyramid": (5, "PYRA5"),
    }

    opener = gzip.open if filename.endswith(".gz") else open

    with opener(filename, "rb") as f:
        while True:
            line = f.readline().decode("utf-8")
            if not line or re.search("\\$END STRUCTURE", line):
                break
            for meshio_type, permas_ele in meshio_to_permas_type.items():
                num_nodes = permas_ele[0]
                permas_type = permas_ele[1]

                if re.search("\\$ELEMENT TYPE = {}".format(permas_type), line):
                    while True:
                        line = f.readline().decode("utf-8")
                        if not line or line.startswith("!"):
                            break
                        data = numpy.array(line.split(), dtype=int)
                        if meshio_type in cells:
                            cells[meshio_type].append(data[-num_nodes:])
                        else:
                            cells[meshio_type] = [data[-num_nodes:]]

            if re.search("\\$COOR", line):
                points = []
                while True:
                    line = f.readline().decode("utf-8")
                    if not line or line.startswith("!"):
                        break
                    for r in numpy.array(line.split(), dtype=float)[1:]:
                        points.append(r)

    points = numpy.array(points)
    points = numpy.reshape(points, newshape=(len(points) // 3, 3))
    for key in cells:
        # Subtract one to account for the fact that python indices
        # are 0-based.
        cells[key] = numpy.array(cells[key], dtype=int) - 1

    return Mesh(points, cells)


def write(filename, mesh):
    """Writes PERMAS dat files, cf.
    http://www.intes.de # PERMAS-ASCII-file-format
    """
    if mesh.points.shape[1] == 2:
        logging.warning(
            "PERMAS requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    opener = gzip.open if filename.endswith(".gz") else open

    with opener(filename, "wb") as fh:
        fh.write(
            "\n".join(
                [
                    "!",
                    "! File written by meshio version {}".format(__version__),
                    "! Further information available at {}".format(__website__),
                    "!",
                    "$ENTER COMPONENT NAME = DFLT_COMP DOFTYPE = DISP MATH",
                    "    $SITUATION NAME = REAL_MODES",
                    "        DFLT_COMP SYSTEM = NSV CONSTRAINTS = SPCVAR_1 ! LOADING = LOADVAR_1",
                    "    $END SITUATION" "!",
                    "!",
                    "    $STRUCTURE",
                    "!" "\n",
                ]
            ).encode("utf-8")
        )

        # Write nodes
        fh.write("        $COOR NSET = ALL_NODES\n".encode("utf-8"))
        for k, x in enumerate(mesh.points):
            fh.write(
                "        {:8d} {:+.15f} {:+.15f} {:+.15f}\n".format(
                    k + 1, x[0], x[1], x[2]
                ).encode("utf-8")
            )

        meshio_to_permas_type = {
            "vertex": (1, "PLOT1"),
            "line": (2, "PLOTL2"),
            "triangle": (3, "TRIA3"),
            "quad": (4, "QUAD4"),
            "tetra": (4, "TET4"),
            "hexahedron": (8, "HEXE8"),
            "wedge": (6, "PENTA6"),
            "pyramid": (5, "PYRA5"),
        }
        #
        # Avoid non-unique element numbers in case of multiple element types by
        # num_ele !!!
        #
        num_ele = 0

        for meshio_type, cell in mesh.cells.items():
            numcells, num_local_nodes = cell.shape
            permas_type = meshio_to_permas_type[meshio_type]
            fh.write("!\n".encode("utf-8"))
            fh.write(
                "        $ELEMENT TYPE = {} ESET = {}\n".format(
                    permas_type[1], permas_type[1]
                ).encode("utf-8")
            )
            for k, c in enumerate(cell):
                form = "        {:8d} " + " ".join(num_local_nodes * ["{:8d}"]) + "\n"
                fh.write(form.format(k + num_ele + 1, *(c + 1)).encode("utf-8"))
            num_ele += numcells

        fh.write("!\n".encode("utf-8"))
        fh.write("    $END STRUCTURE\n".encode("utf-8"))
        fh.write("!\n".encode("utf-8"))
        elem_3D = ["HEXE8", "TET4", "PENTA6", "PYRA5"]
        elem_2D = ["TRIA3", "QUAD4"]
        elem_1D = ["PLOT1", "PLOTL2"]
        fh.write("    $SYSTEM NAME = NSV\n".encode("utf-8"))
        fh.write("!\n".encode("utf-8"))
        fh.write("        $ELPROP\n".encode("utf-8"))
        for meshio_type, cell in mesh.cells.items():
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type[1] in elem_3D:
                fh.write(
                    "            {} MATERIAL = DUMMY_MATERIAL\n".format(
                        permas_type[1]
                    ).encode("utf-8")
                )
            elif permas_type[1] in elem_2D:
                fh.write(
                    (
                        12 * " "
                        + "{} GEODAT = GD_{} MATERIAL = DUMMY_MATERIAL\n".format(
                            permas_type[1], permas_type[1]
                        )
                    ).encode("utf-8")
                )
            else:
                assert permas_type[1] in elem_1D
        fh.write("!\n".encode("utf-8"))
        fh.write("        $GEODAT SHELL  CONT = THICK  NODES = ALL\n".encode("utf-8"))
        for meshio_type, cell in mesh.cells.items():
            permas_type = meshio_to_permas_type[meshio_type]
            if permas_type[1] in elem_2D:
                fh.write(
                    (12 * " " + "GD_{} 1.0\n".format(permas_type[1])).encode("utf-8")
                )
        fh.write(
            """!
!
    $END SYSTEM
!
    $CONSTRAINTS NAME = SPCVAR_1
    $END CONSTRAINTS
!
    $LOADING NAME = LOADVAR_1
    $END LOADING
!
$EXIT COMPONENT
!
$ENTER MATERIAL
!
    $MATERIAL NAME = DUMMY_MATERIAL TYPE = ISO
!
        $ELASTIC  GENERAL  INPUT = DATA
            2.0E+05 0.3
!
        $DENSITY  GENERAL  INPUT = DATA
            7.8E-09
!
        $THERMEXP  GENERAL  INPUT = DATA
            1.2E-05
!
    $END MATERIAL
!
$EXIT MATERIAL
!
$FIN
""".encode(
                "utf-8"
            )
        )
    return
