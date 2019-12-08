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

# Float size and endianess are recorded by these suffixes
# http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/ugc_file_formats.html
file_types = {
     "ascii" : { "type" : "ascii" , "float_type" : "f" ,  "int_type": "i" },
        "b8" : { "type" : "binary", "float_type" : ">f8" ,  "int_type": ">i4" },
        "b4" : { "type" : "binary", "float_type" : ">f4" ,  "int_type": ">i4" },
       "lb8" : { "type" : "binary", "float_type" : "<f8" ,  "int_type": "<i4" },
       "lb4" : { "type" : "binary", "float_type" : "<f4" ,  "int_type": "<i4" },
}

file_type = None

def read(filename):

    global file_type

    file_type = file_types["ascii"]
    
    filename_parts = filename.split('.')
    if len(filename_parts) > 1 :
        type_suffix = filename_parts[-2] 
        if type_suffix in ["r8","r4","lr8","lr4"]:
            raise ReadError("FORTRAN unformatted file format is not supported yet")
        elif type_suffix in file_types.keys():
            file_type = file_types[type_suffix]

    with open_file(filename) as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    cells = {}
    cell_data = {}

    itype = file_type["int_type"]
    ftype = file_type["float_type"]

    def read_section(count, dtype):
        print(file_type)
        if file_type["type"] == "ascii":
            return numpy.fromfile(f, count=count, dtype=dtype, sep = ' ')
        else:
            return numpy.fromfile(f, count=count, dtype=dtype)

    nitems = read_section(count=7, dtype=itype)
    print(nitems)
    if not nitems.size == 7:
        raise ReadError("Header of ugrid file is ill-formed")

    nnodes = nitems[0]
    ntria  = nitems[1]
    nquad  = nitems[2]
    ntet   = nitems[3]
    npyra  = nitems[4]
    nprism = nitems[5]
    nhex   = nitems[6]

    points = read_section(count=nnodes * 3, dtype=ftype).reshape(nnodes, 3) 
    print(points)
    if ntria > 0 :
        out = read_section(count=ntria * 3, dtype=itype).reshape(ntria,3)
        # adapt for 0-base
        cells["triangle"] = out - 1
    if nquad > 0 :
        out = read_section(count=nquad * 4, dtype=itype).reshape(nquad,4)
        # adapt for 0-base
        cells["quad"] = out - 1
    if ntria > 0 :
        out = read_section(count=ntria , dtype=itype)
        cell_data["triangle"] = {"ugrid:ref": out}
    print(cells["triangle"])
    if nquad > 0 :
        out = read_section(count=nquad, dtype=itype)
        cell_data["quad"] = {"ugrid:ref": out}
    if ntet > 0 :
        out = read_section(count=ntet * 4, dtype=itype).reshape(ntet,4)
        # adapt for 0-base
        cells["tetra"] = out - 1
        # TODO check if we can avoid that
        out = numpy.zeros(ntet)
        cell_data["tetra"] = {"ugrid:ref": out}
    if npyra > 0 :
        out = read_section(count=npyra * 5, dtype=itype).reshape(npyra,5)
        # adapt for 0-base
        cells["pyramid"] = out - 1
        # reorder
        cells["pyramid"] = cells["pyramid"][ : , [3,4,1,0,2] ]
        # TODO check if we can avoid that
        out = numpy.zeros(npyra)
        cell_data["pyramid"] = {"ugrid:ref": out}
    if nprism > 0 :
        out = read_section(count=nprism * 6, dtype=itype).reshape(nprism,6)
        # adapt for 0-base
        cells["wedge"] = out - 1
        # reorder
        # TODO check if we can avoid that
        out = numpy.zeros(nprism)
        cell_data["wedge"] = {"ugrid:ref": out}
    if nhex > 0 :
        out = read_section(count=nhex * 8, dtype=itype).reshape(nhex,8)
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
