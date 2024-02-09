"""
I/O for PERMAS dat files.
"""

import numpy as np
import re
#
from ._permdat_line import eldef
from . import _permas_read as prd
from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

meshio_to_permas_type = {v: k for k, v in eldef.permas_to_meshio_type.items()}

def read(filename):
    """Reads a PERMAS dat file."""
    with open_file(filename, "r") as f:
        out = prd.read_buffer(f)
    return out

def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        warn(
            "PERMAS requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    with open_file(filename, "wt") as f:
        f.write("!PERMAS DataFile Version 19.0\n")
        f.write(f"!written by meshio v{__version__}\n")
        f.write("$ENTER COMPONENT NAME=DFLT_COMP\n")
        f.write("$STRUCTURE\n")
        f.write("$COOR\n")
        for k, x in enumerate(points):
            f.write(f"{k + 1} {x[0]} {x[1]} {x[2]}\n")
        eid = 0
        tria6_order = [0, 3, 1, 4, 2, 5]
        tet10_order = [0, 4, 1, 5, 2, 6, 7, 8, 9, 3]
        quad9_order = [0, 4, 1, 7, 8, 5, 3, 6, 2]
        wedge15_order = [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14]
        for cell_block in mesh.cells:
            node_idcs = cell_block.data
            f.write("!\n")
            f.write("$ELEMENT TYPE=" + meshio_to_permas_type[cell_block.type] + "\n")
            if cell_block.type == "tetra10":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in tet10_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            elif cell_block.type == "triangle6":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in tria6_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            elif cell_block.type == "quad9":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in quad9_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            elif cell_block.type == "wedge15":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in wedge15_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            else:
                for row in node_idcs:
                    eid += 1
                    nids_strs = (str(nid + 1) for nid in row.tolist())
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
        f.write("$END STRUCTURE\n")
        f.write("$EXIT COMPONENT\n")
        f.write("$FIN\n")


register_format(
    "permas", [".post", ".post.gz", ".dato", ".dato.gz", ".dat"], read, {"permas": write}
)
