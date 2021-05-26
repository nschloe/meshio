"""
I/O for Netgen mesh files.
"""
import numpy as np

from ..__about__ import __version__
from .._common import _topological_dimension
from .._files import open_file
from .._helpers import register
from .._mesh import Mesh


def _fast_forward_over_blank_lines(f):
    is_eof = False
    while True:
        line = f.readline()
        if not line:
            is_eof = True
            break
        elif len(line.strip()) > 0:
            break
    return line.strip(), is_eof


netgen0d_to_meshio_type = {
    1: "vertex",
}
netgen1d_to_meshio_type = {
    2: "line",
}
netgen2d_to_meshio_type = {
    3: "triangle",
    6: "triangle6",
    4: "quad",
    8: "quad8",
}

netgen3d_to_meshio_type = {
    4: "tetra",
    5: "pyramid",
    6: "wedge",
    8: "hexahedron",
    10: "tetra10",
    13: "pyramid13",
    15: "wedge15",
    20: "hexahedron20",
}

netgen_to_meshio_type = {
    0: netgen0d_to_meshio_type,
    1: netgen1d_to_meshio_type,
    2: netgen2d_to_meshio_type,
    3: netgen3d_to_meshio_type,
}

netgen_to_meshio_pmap = {
    "vertex": [0],
    "line": [0, 1],
    "triangle": list(range(3)),
    "triangle6": [0, 1, 2, 5, 3, 4],
    "quad": list(range(4)),
    "quad8": [0, 1, 2, 3, 4, 7, 5, 6],
    "tetra": [0, 2, 1, 3],
    "tetra10": [0, 2, 1, 3, 5, 7, 4, 6, 9, 8],
    "pyramid": [0, 3, 2, 1, 4],
    "pyramid13": [0, 3, 2, 1, 4, 7, 6, 8, 5, 9, 12, 11, 10],
    "wedge": [0, 2, 1, 3, 5, 4],
    "wedge15": [0, 2, 1, 3, 5, 4, 7, 8, 6, 13, 14, 12, 9, 11, 10],
    "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5],
    "hexahedron20": [
        0,
        3,
        2,
        1,
        4,
        7,
        6,
        5,
        10,
        9,
        11,
        8,
        16,
        19,
        18,
        17,
        14,
        13,
        15,
        12,
    ],
}

meshio_to_netgen_pmap = {}
for t, pmap in netgen_to_meshio_pmap.items():
    n = len(pmap)
    ipmap = list(range(n))
    for i in range(n):
        ipmap[pmap[i]] = i
    meshio_to_netgen_pmap[t] = ipmap


def read(filename):
    if str(filename).endswith(".vol.gz"):
        import gzip

        with gzip.open(filename, "rt") as f:
            return read_buffer(f)

    with open_file(filename, "r") as f:
        return read_buffer(f)


def _read_cells(f, netgen_cell_type, cells):
    if netgen_cell_type == "pointelements":
        dim = 0
        nump = 1
        pi0 = 0
    if netgen_cell_type.startswith("edgesegments"):
        dim = 1
        nump = 2
        pi0 = 1
    if netgen_cell_type == "surfaceelements":
        dim = 2
    if netgen_cell_type == "volumeelements":
        dim = 3

    ncells = int(f.readline())
    tmap = netgen_to_meshio_type[dim]

    for i in range(ncells):
        line, is_eof = _fast_forward_over_blank_lines(f)
        data = list(filter(None, line.split(" ")))
        if dim == 2:
            nump = int(data[4])
            pi0 = 5
        elif dim == 3:
            nump = int(data[1])
            pi0 = 2

        pi = list(map(int, data[pi0 : pi0 + nump]))

        t = tmap[nump]
        pmap = netgen_to_meshio_pmap[t]
        pi = [pi[pmap[i]] for i in range(len(pi))]
        if len(cells) == 0 or t != cells[-1][0]:
            cells.append((t, []))
        cells[-1][1].append(pi)


def _write_cells(f, block):
    if len(block) == 0:
        return
    pmap = np.array(meshio_to_netgen_pmap[block.type])
    dim = _topological_dimension[block.type]
    post_data = []
    if dim == 1:
        pre_data = [1, 0]
        post_data = [-1, -1, 0, 0, 1, 0, 1, 0]
    if dim == 2:
        pre_data = [1, 1, 0, 0, len(pmap)]
    if dim == 3:
        pre_data = [1, len(pmap)]

    col1 = len(pre_data)
    col2 = col1 + len(pmap)
    col3 = col2 + len(post_data)

    pi = np.zeros((len(block), col3), dtype=np.int32)
    pi[:, :col1] = np.repeat([pre_data], len(block), axis=0)
    pi[:, col1:col2] = block.data[:, pmap] + 1
    pi[:, col2:] = np.repeat([post_data], len(block), axis=0)
    np.savetxt(f, pi, "%i")


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
        if line.startswith("#"):
            continue
        elif line == "dimension":
            f.readline()
        elif line == "geomtype":
            f.readline()
        elif line == "mesh3d":
            continue
        elif line == "points":
            npoints = int(f.readline())
            if npoints > 0:
                points = np.fromfile(f, sep=" ", count=3 * npoints).reshape(npoints, 3)

        elif line in [
            "pointelements",
            "edgesegments",
            "surfaceelements",
            "volumeelements",
        ]:
            _read_cells(f, line, cells)

        elif line == "endmesh":
            break

        elif line in [
            "bcnames",
            "cd2names",
            "cd3names",
            "face_colours",
            "identifications",
            "identificationtypes",
            "materials",
            "singular_edge_left",
            "singular_edge_right",
            "singular_face_inside",
            "singular_face_outside",
            "singular_points",
        ]:
            _skip_block(f)
        else:
            continue

    # convert to numpy arrays
    # Subtract one to account for the fact that python indices are 0-based.
    for k, c in enumerate(cells):
        cells[k] = (c[0], np.array(c[1], dtype=int) - 1)

    # Construct the mesh object
    mesh = Mesh(points, cells)
    return mesh


def write(filename, mesh):
    point_format = "fixed-large"
    cell_format = "fixed-small"
    if str(filename).endswith(".vol.gz"):
        import gzip

        with gzip.open(filename, "wt") as f:
            return write_buffer(f, mesh, point_format, cell_format)

    with open_file(filename, "w") as f:
        return write_buffer(f, mesh, point_format, cell_format)


def write_buffer(f, mesh, point_format, cell_format):
    cells_per_dim = [0, 0, 0, 0]
    for block in mesh.cells:
        cells_per_dim[_topological_dimension[block.type]] += len(block)

    print(
        """# Generated by meshio {}
mesh3d

dimension
3

geomtype
0
""".format(
            __version__
        ),
        file=f,
    )
    print("\n# surfnr    bcnr   domin  domout      np      p1      p2      p3", file=f)
    print("surfaceelements", file=f)
    print(cells_per_dim[2], file=f)
    for block in mesh.cells:
        if _topological_dimension[block.type] == 2:
            _write_cells(f, block)

    print("\n#  matnr      np      p1      p2      p3      p4", file=f)
    print("volumeelements", file=f)
    print(cells_per_dim[3], file=f)
    for block in mesh.cells:
        if _topological_dimension[block.type] == 3:
            _write_cells(f, block)

    print(
        "\n# surfid  0   p1   p2   trignum1    trignum2   domin/surfnr1    domout/surfnr2   ednr1   dist1   ednr2   dist2",
        file=f,
    )
    print("edgesegmentsgi2", file=f)
    print(cells_per_dim[1], file=f)
    for block in mesh.cells:
        if _topological_dimension[block.type] == 1:
            _write_cells(f, block)

    print("\n#          X             Y             Z", file=f)
    print("points", file=f)
    print(len(mesh.points), file=f)

    points = mesh.points
    npoints, dim = points.shape
    if dim == 2:
        p = np.zeros((points.shape[0], 3), dtype=points.dtype)
        p[:, :2] = points
    else:
        p = points
    np.savetxt(f, p, "% 20.16f")

    print("\n#          pnum             index", file=f)
    print("pointelements", file=f)
    print("0", file=f)

    print(
        """
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
""",
        file=f,
    )

    print("endmesh", file=f)


register("netgen", [".vol", ".vol.gz"], read, {"netgen": write})
