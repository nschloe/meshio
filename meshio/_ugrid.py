"""
I/O for AFLR's UGRID format
TODO document :
[1] <http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/3d_grid_file_type_ugrid.html>.
http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/3d_input_output_grids.html
Check out
[2] <http://www.simcenter.msstate.edu/software/downloads/ug_io/index_simsys_web.php?path=release>
for UG_IO C code able to read and convert UGRID files
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
        "b8" : { "type" : "C", "float_type" : ">f8" ,  "int_type": ">i4" },
        "b4" : { "type" : "C", "float_type" : ">f4" ,  "int_type": ">i4" },
       "lb8" : { "type" : "C", "float_type" : "<f8" ,  "int_type": "<i4" },
       "lb4" : { "type" : "C", "float_type" : "<f4" ,  "int_type": "<i4" },
       "r8" : { "type" : "F", "float_type" : ">f8" ,  "int_type": ">i4" },
       "r4" : { "type" : "F", "float_type" : ">f4" ,  "int_type": ">i4" },
       "lr8" : { "type" : "F", "float_type" : "<f8" ,  "int_type": "<i4" },
       "lr4" : { "type" : "F", "float_type" : "<f4" ,  "int_type": "<i4" },
}

file_type = None

def determine_file_type(filename) :
    file_type = file_types["ascii"]
    filename_parts = filename.split('.')
    if len(filename_parts) > 1 :
        type_suffix = filename_parts[-2] 
        if type_suffix in file_types.keys():
            file_type = file_types[type_suffix]

    return file_type


def read(filename):

    global file_type

    file_type = determine_file_type(filename)

    with open_file(filename) as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    cells = {}
    cell_data = {}

    itype = file_type["int_type"]
    ftype = file_type["float_type"]

    def read_section(count, dtype):
        if file_type["type"] == "ascii":
            return numpy.fromfile(f, count=count, dtype=dtype, sep = ' ')
        else :
            return numpy.fromfile(f, count=count, dtype=dtype)
    
    # FORTRAN type includes a number of bytes before and after
    # each record , according to documentation [1] there are
    # two records in the file
    # see also UG_IO freely available code at [2]
    if file_type["type"] == "F":
        nbytes = numpy.fromfile(f,count=1,dtype=itype)
    
    nitems = read_section(count=7, dtype=itype)

    if file_type["type"] == "F":
        nbytes = numpy.fromfile(f,count=1,dtype=itype)

    if not nitems.size == 7:
        raise ReadError("Header of ugrid file is ill-formed")

    nnodes = nitems[0]
    ntria  = nitems[1]
    nquad  = nitems[2]
    ntet   = nitems[3]
    npyra  = nitems[4]
    nprism = nitems[5]
    nhex   = nitems[6]

    if file_type["type"] == "F":
        nbytes = numpy.fromfile(f,count=1,dtype=itype)

    points = read_section(count=nnodes * 3, dtype=ftype).reshape(nnodes, 3) 
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
    if file_type["type"] == "F":
        nbytes = numpy.fromfile(f,count=1,dtype=itype)

    return Mesh(points, cells, cell_data=cell_data)


def write(filename, mesh):
    file_type = determine_file_type(filename)
    
    with open_file(filename, "w") as f:
        itype = file_type["int_type"]
        ftype = file_type["float_type"]

        def write_section(array,dtype):
            if file_type["type"] == "ascii":
                fmt = " ".join(["%{}".format(dtype)] * ( array.shape[1]))
                numpy.savetxt(f, array, fmt)
            else:
                array.astype(dtype).tofile(f)
        
        ugrid_counts = { "points" : 0, "triangle" : 0 , "quad" : 0, "tetra" : 0 , "pyramid" : 0, "wedge" : 0, "hexahedron" : 0 }

        ugrid_counts["points"] = mesh.points.shape[0]
        
        for key, data in mesh.cells.items():
            if key in ugrid_counts.keys():
                ugrid_counts[key] = data.shape[0]
            else:
                msg = ("UGRID mesh format doesn't know {} cells. Skipping.").format(
                    key
                )
                logging.warning(msg)
                continue

        nitems = numpy.array(
                [ [ 
                ugrid_counts["points"],
                ugrid_counts["triangle"],
                ugrid_counts["quad"],
                ugrid_counts["tetra"],
                ugrid_counts["pyramid"],
                ugrid_counts["wedge"],
                ugrid_counts["hexahedron"] 
                ] ])
        # header
        write_section(nitems,itype)
        write_section(mesh.points,ftype)

        for key in ["triangle","quad"]:
            if ugrid_counts[key] > 0 :
                # UGRID is one-based
                out = mesh.cells[key] + 1
                write_section(out,itype)
        
        # write boundary tags
        for key in ["triangle","quad"]:
            if ugrid_counts[key] == 0 :
                continue
            if "ugrid:ref" in mesh.cell_data[key]:
                labels = mesh.cell_data[key]["ugrid:ref"]
            elif "medit:ref" in mesh.cell_data[key]:
                labels = mesh.cell_data[key]["medit:ref"]
            elif "gmsh:physical" in mesh.cell_data[key]:
                labels = mesh.cell_data[key]["gmsh:physical"]
            elif key in mesh.cell_data and "flac3d:zone" in mesh.cell_data[key]:
                labels = mesh.cell_data[key]["flac3d:zone"]
            else:
                labels = numpy.ones(ugrid_counts[key], dtype=itype)
            
            labels = labels.reshape(ugrid_counts[key],1)
            write_section(labels,itype)

        # write volume elements
        for key in ["tetra","pyramid","wedge","hexahedron"]:
            if ugrid_counts[key] > 0 :
                # UGRID is one-based
                out = mesh.cells[key] + 1
                if key == "pyramid":
                    out = out[ : , [3,4,1,0,2] ]
                write_section(out,itype)
    return
