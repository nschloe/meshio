"""
I/O for Medit's format, cf.
<https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html>.
Check out
<https://hal.inria.fr/inria-00069921/fr/>
<https://www.ljll.math.upmc.fr/frey/publications/RT-0253.pdf>
<https://www.math.u-bordeaux.fr/~dobrzyns/logiciels/RT-422/node58.html>
for something like a specification.
"""
import logging
from ctypes import c_double, c_float

import numpy

from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


def read(filename):

    with open_file(filename) as f:
        if filename[-1] == "b":
            mesh = read_binary_buffer(f)
        else:
            mesh = read_buffer(f)
    return mesh

def read_binary_buffer(f):
    dim = 0
    cells = []
    point_data = {}
    cell_data = {"medit:ref": []}

    meshio_from_medit = {
        "GmfVertices":("point",None),
        "GmfEdges": ("line", 2),
        "GmfTriangles": ("triangle", 3),
        "GmfQuadrilaterals": ("quad", 4),
        "GmfTetrahedra": ("tetra", 4),
        "GmfPrisms": ("wedge",6),
        "GmfHexahedra": ("hexahedron", 8),
    }

    medit_codes = {
            3:"GmfDimension",
            4:"GmfVertices",
            5:"GmfEdges",
            6:"GmfTriangles",
            7:"GmfQuadrilaterals",
            8:"GmfTetrahedra",
            9:"GmfPrisms",
            10:"GmfHexahedra",
            54:"GmfEnd"
            }

    code = numpy.fromfile(f, count=1, dtype="i4")
    if code != 1 and code != 16777216:
        raise ReadError("Invalid code")
    #TODO endianess
    
    version = numpy.fromfile(f, count=1, dtype="i4")
    if version < 1 or version > 4:
        raise ReadError("Invalid version")
    print("vertsion ",version)

    #TODO size
    itype="i4"
    ftype="f8"
    #append endianess
    points = None
    cells = []
    point_data = dict()
    celldata = None
    postype="i4"

    print("code " ,code)

    field = numpy.fromfile(f, count=1, dtype="i4")  

    if field != 3 :# =  GmfDimension
        raise ReadError("Invalid dimension code")

    pos = numpy.fromfile(f, count=1, dtype="i4")  

    dim = numpy.fromfile(f, count=1, dtype="i4")[0]
    
    if dim != 2 and dim != 3:
        raise ReadError("Invalid mesh dimension")

    while True:
        field = numpy.fromfile(f, count=1, dtype="i4")[0]
        if field not in medit_codes.keys():
            pos = numpy.fromfile(f, count=1, dtype="i4")  
            print("skipping")
            continue
        if medit_codes[field] == "GmfEnd":
            break
        pos = numpy.fromfile(f, count=1, dtype="i4")  

        meshio_type,ncols = meshio_from_medit[medit_codes[field]]
        nitems = numpy.fromfile(f, count=1, dtype=itype)[0]

        if meshio_type == "point":
            ncols = dim + 1
            print(nitems)
            print(nitems*ncols)
            print([dim*ftype,itype])
            dtype= "".join([dim * (ftype +","),itype])
            dtype= numpy.dtype(dtype)
            out = numpy.fromfile(f,count=nitems, dtype=dtype)
            points = numpy.column_stack((out['f0'],out['f1'],out['f2']))
            point_data["medit:ref"] = out['f3']
        else:
            ncols = ncols+1 # add reference
            out = numpy.fromfile(f,count=nitems * ncols, dtype=itype).reshape(nitems,ncols)
            # adapt for 0-base
            cells.append((meshio_type, out[:, :ncols - 1] - 1))
            cell_data["medit:ref"].append(out[:, -1])

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


    

def read_buffer(f):
    dim = 0
    cells = []
    point_data = {}
    cell_data = {"medit:ref": []}

    meshio_from_medit = {
        "Edges": ("line", 2),
        "Triangles": ("triangle", 3),
        "Quadrilaterals": ("quad", 4),
        "Tetrahedra": ("tetra", 4),
        "Hexahedra": ("hexahedron", 8),  # Frey
        "Hexaedra": ("hexahedron", 8),  # Dobrzynski
    }

    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0 or line[0] == "#":
            continue

        items = line.split()

        if not items[0].isalpha():
            raise ReadError()

        if items[0] == "MeshVersionFormatted":
            version = items[1]
            dtype = {"1": c_float, "2": c_double}[version]
        elif items[0] == "Dimension":
            dim = int(items[1])
        elif items[0] == "Vertices":
            if dim <= 0:
                raise ReadError()
            num_verts = int(f.readline())
            out = numpy.fromfile(
                f, count=num_verts * (dim + 1), dtype=dtype, sep=" "
            ).reshape(num_verts, dim + 1)
            points = out[:, :dim]
            point_data["medit:ref"] = out[:, dim].astype(int)
        elif items[0] in meshio_from_medit:
            meshio_type, points_per_cell = meshio_from_medit[items[0]]
            # The first value is the number of elements
            num_cells = int(f.readline())

            out = numpy.fromfile(
                f, count=num_cells * (points_per_cell + 1), dtype=int, sep=" "
            ).reshape(num_cells, points_per_cell + 1)

            # adapt for 0-base
            cells.append((meshio_type, out[:, :points_per_cell] - 1))
            cell_data["medit:ref"].append(out[:, -1])
        elif items[0] == "Normals":
            # those are just discarded
            num_normals = int(f.readline())
            numpy.fromfile(f, count=num_normals * dim, dtype=dtype, sep=" ").reshape(
                num_normals, dim
            )
        elif items[0] == "NormalAtVertices":
            # those are just discarded
            num_normal_at_vertices = int(f.readline())
            numpy.fromfile(
                f, count=num_normal_at_vertices * 2, dtype=int, sep=" "
            ).reshape(num_normal_at_vertices, 2)
        else:
            if items[0] != "End":
                raise ReadError("Unknown keyword '{}'.".format(items[0]))

    return Mesh(points, cells, point_data=point_data, cell_data=cell_data)


def write(filename, mesh):
    with open_file(filename, "wb") as fh:
        version = {numpy.dtype(c_float): 1, numpy.dtype(c_double): 2}[mesh.points.dtype]
        # N. B.: PEP 461 Adding % formatting to bytes and bytearray
        fh.write(b"MeshVersionFormatted %d\n" % version)

        n, d = mesh.points.shape

        fh.write(b"Dimension %d\n" % d)

        # vertices
        fh.write(b"\nVertices\n")
        fh.write(f"{n}\n".encode("utf-8"))
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

        for icg, (key, data) in enumerate(mesh.cells):
            try:
                medit_name, num = medit_from_meshio[key]
            except KeyError:
                msg = ("MEDIT's mesh format doesn't know {} cells. Skipping.").format(
                    key
                )
                logging.warning(msg)
                continue
            fh.write(b"\n")
            fh.write(f"{medit_name}\n".encode("utf-8"))
            fh.write("{}\n".format(len(data)).encode("utf-8"))

            if "medit:ref" in mesh.cell_data:
                labels = mesh.cell_data["medit:ref"][icg]
            elif "gmsh:physical" in mesh.cell_data:
                # Translating gmsh data to medit is an important case, so treat it
                # explicitly here.
                labels = mesh.cell_data["gmsh:physical"][icg]
            elif "flac3d:zone" in mesh.cell_data:
                labels = mesh.cell_data["flac3d:zone"][icg]
            elif "avsucd:material" in mesh.cell_data:
                labels = mesh.cell_data["avsucd:material"][icg]
            else:
                labels = numpy.ones(len(data), dtype=int)

            # adapt 1-base
            data_with_label = numpy.c_[data + 1, labels]
            fmt = " ".join(["%d"] * (num + 1))
            numpy.savetxt(fh, data_with_label, fmt)

        fh.write(b"\nEnd\n")


register("medit", [".mesh",".meshb"], read, {"medit": write})
