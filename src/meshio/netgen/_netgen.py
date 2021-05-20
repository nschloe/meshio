"""
I/O for Netgen mesh files.
"""
import logging

import numpy as np

from ..__about__ import __version__
from .._common import num_nodes_per_cell, _topological_dimension 
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register
from .._mesh import CellBlock, Mesh

def _fast_forward_over_blank_lines(f):
    is_eof = False
    while True:
        line = f.readline()
        if not line:
            is_eof = True
            break
        elif len(line.strip()) > 0:
            break
    return line, is_eof

netgen1d_to_meshio_type = {
    2 : "line",
}
netgen2d_to_meshio_type = {
    3 : "triangle",
    6 : "triangle6",
    4 : "quad",
    8 : "quad8",
}

netgen3d_to_meshio_type = {
    4 : "tetra",
    5 : "pyramid",
    6 : "wedge",
    8 : "hexahedron",
    10 : "tetra10",
    13 : "pyramid13",
    15 : "wedge15",
    20 : "hexahedron20",
}

netgen_to_meshio_type = {
    1 : netgen1d_to_meshio_type,
    2 : netgen2d_to_meshio_type,
    3 : netgen3d_to_meshio_type,
}

netgen_to_meshio_pmap = {
    "line" : [0,1],
    "triangle" : list(range(3)),
    "triangle6" : [0,1,2,5,3,4],
    "quad" : list(range(4)),
    "quad8" : [0,1,2,3,4,7,5,6],
    "tetra" : [0,2,1,3],
    "tetra10" : [0,2,1,3,5,7,4,6,9,8],
    "pyramid" : [0,3,2,1,4],
    "pyramid13" :  [0,3,2,1,4,7,6,8,5,9,12,11,10],
    "wedge" : [0,2,1,3,5,4],
    "wedge15" :  [0,2,1,3,5,4,7,8,6,13,14,12,9,11,10],
    "hexahedron" : [0,3,2,1,4,7,6,5],
    "hexahedron20" : [0,3,2,1,4,7,6,5,10,9,11,8, 16,19,18,17, 14,13,15,12]
}

meshio_to_netgen_pmap = {}
for t,pmap in netgen_to_meshio_pmap.items():
    n = len(pmap)
    ipmap = list(range(n))
    for i in range(n):
        ipmap[pmap[i]] = i
    meshio_to_netgen_pmap[t] = ipmap

def read(filename):
    if filename.endswith(".vol.gz"):
        import gzip
        with gzip.open(filename, "rt") as f:
            return read_buffer(f)

    with open_file(filename, "r") as f:
        return read_buffer(f)

def _read_cells( f, cells, dim ):
    ncells = int(f.readline())
    tmap = netgen_to_meshio_type[dim]

    for i in range(ncells):
        data = list(map(int, f.readline().strip().split(' ')))
        if dim==1:
            np = 2
            pi = data[1:3]
        elif dim==2:
            np = data[4]
            pi = data[5:5+np]
        elif dim==3:
            np = data[1]
            pi = data[2:2+np]

        t = tmap[np]
        pmap = netgen_to_meshio_pmap[t]
        pi = [ pi[pmap[i]] for i in range(len(pi)) ]
        if len(cells) == 0 or t != cells[-1][0]:
            cells.append((t, []))
        cells[-1][1].append( pi )

def _write_cells( f, block ):
    pmap = meshio_to_netgen_pmap[block.type]
    dim = _topological_dimension[block.type]
    post_data = []
    if dim==1:
        pre_data = [1, 0]
        post_data = [-1, -1, 0, 0, 1, 0, 1, 0]
    if dim==2:
        pre_data = [1,1,0,0, len(pmap)]
    if dim==3:
        pre_data = [1, len(pmap)]

    for i in range(len(block)):
        pi = block.data[i]
        pi = [pi[pmap[k]]+1 for k in range(len(pi))]
        print(*pre_data, *pi, *post_data, file=f)

def _skip_block(f):
    n = int(f.readline())
    for i in range(n):
        f.readline()

def read_buffer(f):
    points = []
    cells = []

    while True:
        # fast-forward over blank lines
        line, is_eof = _fast_forward_over_blank_lines(f)
        if is_eof:
            break

        line = line.strip()
        if line.startswith('#'):
            continue
        elif line == "dimension":
            f.readline()
        elif line == "geomtype":
            f.readline()
        elif line == "mesh3d":
            continue
        elif line == "points":
            npoints = int(f.readline())
            for i in range(npoints):
                p = filter(None, f.readline().strip().split(' '))
                points.append( list(map(float,  p)) )

        elif line == "edgesegments":
            _read_cells(f, cells, 1)

        elif line == "surfaceelements":
            _read_cells(f, cells, 2)

        elif line == "volumeelements":
            _read_cells(f, cells, 3)

        elif line == "endmesh":
            break

        else:
            _skip_block(f)

    # convert to numpy arrays
    # Subtract one to account for the fact that python indices are 0-based.
    for k, c in enumerate(cells):
        cells[k] = (c[0], np.array(c[1], dtype=int) - 1)

    # Construct the mesh object
    mesh = Mesh(points, cells)
    return mesh


def write(filename, mesh):
    point_format="fixed-large"
    cell_format="fixed-small"
    if filename.endswith(".vol.gz"):
        import gzip
        with gzip.open(filename, "wt") as f:
            return write_buffer(f, mesh, point_format, cell_format)

    with open_file(filename, "w") as f:
        return write_buffer(f, mesh, point_format, cell_format)

def write_buffer(f, mesh, point_format, cell_format):
    cells_per_dim = [0,0,0,0]
    for block in mesh.cells:
        cells_per_dim[_topological_dimension[block.type]] += len(block)

    print("""# Generated by meshio
mesh3d
dimension
3
geomtype
0

""", file=f)
    print("# surfnr    bcnr   domin  domout      np      p1      p2      p3", file=f)
    print("surfaceelements", file=f)
    print(cells_per_dim[2], file=f)
    for block in mesh.cells:
        if _topological_dimension[block.type] == 2:
            _write_cells(f, block)

    print("#  matnr      np      p1      p2      p3      p4", file=f)
    print("volumeelements", file=f)
    print(cells_per_dim[3], file=f)
    for block in mesh.cells:
        if _topological_dimension[block.type] == 3:
            _write_cells(f, block)

    print("# surfid  0   p1   p2   trignum1    trignum2   domin/surfnr1    domout/surfnr2   ednr1   dist1   ednr2   dist2", file=f)
    print("edgesegmentsgi2", file=f)
    print(cells_per_dim[1], file=f)
    for block in mesh.cells:
        if _topological_dimension[block.type] == 1:
            _write_cells(f, block)

    print("#          X             Y             Z", file=f)
    print("points", file=f)
    print(len(mesh.points), file=f)

    for p in mesh.points:
        if len(p)==2:
            print(*p, 0,file=f)
        else:
            print(*p, file=f)

    print("#          pnum             index", file=f)
    print("pointelements", file=f)
    print("0", file=f)

    print("""
materials
1
1 domain


bcnames
1
1 boundary

#   Surfnr     Red     Green     Blue
face_colours
1
       1   0.00000000   1.00000000   0.00000000
""", file=f)

    print("endmesh", file=f)


register("netgen", [".vol", ".vol.gz"], read, {"netgen": write})