"""
I/O for AFLR's UGRID format
TODO document :
<http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/3d_grid_file_type_ugrid.html>.
http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/3d_input_output_grids.html
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
<https://www.math.u-bordeaux.fr/~dobrzyns/logiciels/RT-422/node58.html>
for something like a specification.
"""
import logging
from ctypes import c_double, c_float

import numpy

from ._exceptions import ReadError
from ._files import open_file
from ._mesh import Mesh


def read(filename):
    with open_file(filename) as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    cells = {}
    cell_data = {}

    line = f.readline()
    if not line:
        # EOF
        raise ReadError()

    line = line.strip()
    items = line.split()
    if not len(items) == 7:
        raise ReadError()

    nnodes = int(items[0])
    ntria  = int(items[1])
    nquad  = int(items[2])
    ntet   = int(items[3])
    npyra  = int(items[4])
    nprism = int(items[5])
    nhex   = int(items[6])

    points = numpy.fromfile( f, count = nnodes * 3, dtype=c_double, sep=" ").reshape(nnodes, 3) 
    if ntria > 0 :
        out = numpy.fromfile( f, count=ntria * 3 , dtype=int, sep=" ").reshape(ntria,3)
        # adapt for 0-base
        cells["triangle"] = out - 1
    if nquad > 0 :
        out = numpy.fromfile( f, count=nquad * 4 , dtype=int, sep=" ").reshape(nquad,4)
        # adapt for 0-base
        cells["quad"] = out - 1
    if ntria > 0 :
        out = numpy.fromfile( f, count=ntria , dtype=int, sep=" ")
        cell_data["triangle"] = {"ugrid:ref": out}
    if nquad > 0 :
        out = numpy.fromfile( f, count=nquad, dtype=int, sep=" ")
        cell_data["quad"] = {"ugrid:ref": out}
    if ntet > 0 :
        out = numpy.fromfile( f, count=ntet * 4, dtype=int, sep=" ").reshape(ntet,4)
        # adapt for 0-base
        cells["tetra"] = out - 1
        # TODO check if we can avoid that
        out = numpy.zeros(ntet)
        cell_data["tetra"] = {"ugrid:ref": out}
    if npyra > 0 :
        out = numpy.fromfile( f, count=npyra * 5, dtype=int, sep=" ").reshape(npyra,5)
        # adapt for 0-base
        cells["pyramid"] = out - 1
        # reorder
        cells["pyramid"] = cells["pyramid"][ : , [3,4,1,0,2] ]
        # TODO check if we can avoid that
        out = numpy.zeros(npyra)
        cell_data["pyramid"] = {"ugrid:ref": out}
    if nprism > 0 :
        out = numpy.fromfile( f, count=nprism * 6, dtype=int, sep=" ").reshape(nprism,6)
        # adapt for 0-base
        cells["wedge"] = out - 1
        # reorder
        # TODO check if we can avoid that
        out = numpy.zeros(nprism)
        cell_data["wedge"] = {"ugrid:ref": out}
    if nhex > 0 :
        out = numpy.fromfile( f, count=nhex * 8, dtype=int, sep=" ").reshape(nhex,8)
        # adapt for 0-base
        cells["hexahedron"] = out - 1
        # reorder
        # TODO check if we can avoid that
        out = numpy.zeros(nhex)
        cell_data["hexahedron"] = {"ugrid:ref": out}

    return Mesh(points, cells, cell_data=cell_data)


def write(filename, mesh):
    with open_file(filename, "wb") as fh:
        version = {numpy.dtype(c_float): 1, numpy.dtype(c_double): 2}[mesh.points.dtype]
        """
        # N. B.: PEP 461 Adding % formatting to bytes and bytearray
        fh.write(b"MeshVersionFormatted %d\n" % version)

        n, d = mesh.points.shape

        fh.write(b"Dimension %d\n" % d)

        # vertices
        fh.write(b"\nVertices\n")
        fh.write("{}\n".format(n).encode("utf-8"))
        if "medit:ref" in mesh.point_data:
            labels = mesh.point_data["medit:ref"]
        elif "gmsh:physical" in mesh.point_data:
            # Translating gmsh data to medit is an important case, so treat it
            # explicitly here.
            labels = mesh.point_data["gmsh:physical"]
        else:
            labels = numpy.ones(n, dtype=int)
        data = numpy.c_[mesh.points, labels]
        fmt = " ".join(["%r"] * d) + " %d"
        numpy.savetxt(fh, data, fmt)

        medit_from_meshio = {
            "line": ("Edges", 2),
            "triangle": ("Triangles", 3),
            "quad": ("Quadrilaterals", 4),
            "tetra": ("Tetrahedra", 4),
            "hexahedron": ("Hexahedra", 8),
        }

        for key, data in mesh.cells.items():
            try:
                medit_name, num = medit_from_meshio[key]
            except KeyError:
                msg = ("MEDIT's mesh format doesn't know {} cells. Skipping.").format(
                    key
                )
                logging.warning(msg)
                continue
            fh.write(b"\n")
            fh.write("{}\n".format(medit_name).encode("utf-8"))
            fh.write("{}\n".format(len(data)).encode("utf-8"))

            if key in mesh.cell_data and "medit:ref" in mesh.cell_data[key]:
                labels = mesh.cell_data[key]["medit:ref"]
            elif key in mesh.cell_data and "gmsh:physical" in mesh.cell_data[key]:
                # Translating gmsh data to medit is an important case, so treat it
                # explicitly here.
                labels = mesh.cell_data[key]["gmsh:physical"]
            elif key in mesh.cell_data and "flac3d:zone" in mesh.cell_data[key]:
                labels = mesh.cell_data[key]["flac3d:zone"]
            else:
                labels = numpy.ones(len(data), dtype=int)

            # adapt 1-base
            data_with_label = numpy.c_[data + 1, labels]
            fmt = " ".join(["%d"] * (num + 1))
            numpy.savetxt(fh, data_with_label, fmt)

        fh.write(b"\nEnd\n")

    """
    return
