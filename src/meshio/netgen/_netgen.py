"""
I/O for Netgen mesh files
<https://github.com/NGSolve/netgen/blob/master/libsrc/meshing/meshclass.cpp>.
"""
import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._files import open_file
from .._helpers import register_format
from .._mesh import Mesh


def _fast_forward_over_blank_lines(f):
    is_eof = False
    while True:
        line = f.readline()
        if not line:
            is_eof = True
            break

        line = line.strip()
        if len(line) > 0 and not line.startswith("#"):
            break
    return line, is_eof


netgen_codims = {"materials": 0, "bcnames": 1, "cd2names": 2, "cd3names": 3}


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


def _read_cells(f, netgen_cell_type, cells, cells_index, skip_every_other_line=False):
    if netgen_cell_type == "pointelements":
        dim = 0
        nump = 1
        pi0 = 0
        i_index = 1
    elif netgen_cell_type.startswith("edgesegments"):
        dim = 1
        nump = 2
        pi0 = 2
        i_index = 0
    elif netgen_cell_type.startswith("surfaceelements"):
        dim = 2
        pi0 = 5
        i_index = 1
    elif netgen_cell_type == "volumeelements":
        dim = 3
        pi0 = 2
        i_index = 0
    else:
        raise ValueError(f"Unknown Netgen cell section '{netgen_cell_type}'")

    num_cells = int(f.readline())
    tmap = netgen_to_meshio_type[dim]

    for _ in range(num_cells):
        line, _ = _fast_forward_over_blank_lines(f)
        data = list(filter(None, line.split(" ")))
        index = int(data[i_index])
        if dim == 2:
            nump = int(data[4])
        elif dim == 3:
            nump = int(data[1])

        pi = list(map(int, data[pi0 : pi0 + nump]))

        t = tmap[nump]
        if len(cells) == 0 or t != cells[-1][0]:
            cells.append((t, []))
            cells_index.append([])
        cells[-1][1].append(pi)
        cells_index[-1].append(index)
        if skip_every_other_line:
            line, _ = _fast_forward_over_blank_lines(f)


def _write_cells(f, cell_block, index=None):
    if len(cell_block) == 0:
        return
    pmap = np.array(meshio_to_netgen_pmap[cell_block.type])
    post_data = []
    pre_data = []
    i_index = 0
    if cell_block.dim == 0:
        post_data = [1]
        i_index = 1
    elif cell_block.dim == 1:
        pre_data = [1, 0]
        post_data = [-1, -1, 0, 0, 1, 0, 1, 0]
    elif cell_block.dim == 2:
        pre_data = [1, 1, 0, 0, len(pmap)]
        i_index = 1
    elif cell_block.dim == 3:
        pre_data = [1, len(pmap)]
    else:
        raise ValueError(f"Invalid cell dimension: {cell_block.dim}")

    col1 = len(pre_data)
    col2 = col1 + len(pmap)
    col3 = col2 + len(post_data)

    pi = np.zeros((len(cell_block), col3), dtype=np.int32)
    pi[:, :col1] = np.repeat([pre_data], len(cell_block), axis=0)
    pi[:, col1:col2] = cell_block.data[:, pmap] + 1
    pi[:, col2:] = np.repeat([post_data], len(cell_block), axis=0)
    if index is not None:
        pi[:, i_index] = index
    np.savetxt(f, pi, "%i")


def _skip_block(f):
    n = int(f.readline())
    for _ in range(n):
        f.readline()


def _write_codim_domain_data(f, mesh, cells_index, dim, codim):
    # assume format as read from gmsh 4.1 files
    data = {}
    for name, val in mesh.field_data.items():
        if val[1] == dim - codim:
            data[val[0]] = name

    # set generic default names (is this appropriate/useful?)
    if len(data) == 0:
        indices = set()
        for cell_block, index in zip(mesh.cells, cells_index):
            if index is None:
                continue
            if cell_block.dim == dim - codim:
                indices = indices.union(set(index))

        for idx in indices:
            data[idx] = f"cd{codim:d}_{idx:d}"

    if len(data) == 0:
        return

    codim_tag = [kk for kk, vv in netgen_codims.items() if vv == codim][0]
    f.write(f"\n{codim_tag:s}\n")

    ncd = max(data.keys())
    f.write(f"{ncd:d}\n")
    for idx in range(1, ncd + 1):
        f.write("{:d} {:s}\n".format(idx, data.get(idx, "")))


def read_buffer(f):
    points = []
    cells = []
    cells_index = []
    field_data = {}
    identifications = None
    identificationtypes = None

    have_edgesegmentsgi2_in_two_lines = False

    line, is_eof = _fast_forward_over_blank_lines(f)
    if line != "mesh3d":
        raise RuntimeError("Not a valid Netgen mesh")

    while True:
        line, is_eof = _fast_forward_over_blank_lines(f)
        if is_eof:
            break

        elif line == "dimension":
            dimension = int(f.readline())
        elif line == "geomtype":
            geomtype = int(f.readline())
            if geomtype not in [0, 1, 10, 11, 12, 13]:
                warn(f"Unknown geomtype in Netgen mesh: {geomtype}")

        elif line == "points":
            num_points = int(f.readline())
            if num_points > 0:
                points = np.loadtxt(f, max_rows=num_points)
                if dimension != 3:
                    points = points[:, :dimension]

        elif line in [
            "pointelements",
            "edgesegments",
            "edgesegmentsgi",
            "edgesegmentsgi2",
            "surfaceelements",
            "surfaceelementsgi",
            "surfaceelementsuv",
            "volumeelements",
        ]:
            _read_cells(f, line, cells, cells_index, have_edgesegmentsgi2_in_two_lines)
        elif line == "endmesh":
            break

        elif line.split() == ["surf1", "surf2", "p1", "p2"]:
            # if this line is present, the edgesegmentsgi2 info is split in two lines per data set
            have_edgesegmentsgi2_in_two_lines = True

        elif line in netgen_codims.keys():
            edim = dimension - netgen_codims[line]
            num_entries = int(f.readline())
            for _ in range(num_entries):
                line = f.readline().split()
                if len(line) != 2:
                    continue
                idx, name = line
                field_data[name] = [int(idx), edim]

        elif line == "identifications":
            num_entries = int(f.readline())
            if num_entries > 0:
                identifications = np.loadtxt(
                    f, max_rows=num_entries, dtype=int
                ).reshape(num_entries, 3)

        elif line == "identificationtypes":
            num_entries = int(f.readline())
            if num_entries > 0:
                identificationtypes = np.loadtxt(f, max_rows=1, dtype=int).reshape(
                    1, num_entries
                )

        elif line in [
            "face_colours",
            "singular_edge_left",
            "singular_edge_right",
            "singular_face_inside",
            "singular_face_outside",
            "singular_points",
        ]:
            _skip_block(f)
        else:
            raise RuntimeError(f"Unknown Netgen mesh token: {line}")

    # convert to numpy arrays
    # subtract one (netgen counts 1-based)
    # apply permutation of vertex numbers
    for k, (t, data) in enumerate(cells):
        pmap = netgen_to_meshio_pmap[t]
        d = np.array(data, dtype=np.uint32)
        d[:, :] = d[:, pmap] - 1
        cells[k] = (t, d)

    # currently, there is no better place for identification data
    kwargs = {}
    if identifications is not None:
        kwargs["info"] = {
            "netgen:identifications": identifications,
            "netgen:identificationtypes": identificationtypes,
        }

    mesh = Mesh(
        points,
        cells,
        cell_data={"netgen:index": cells_index},
        field_data=field_data,
        **kwargs,
    )
    return mesh


def write(filename, mesh, float_fmt=".16e"):
    if str(filename).endswith(".vol.gz"):
        import gzip

        with gzip.open(filename, "wt") as f:
            write_buffer(f, mesh, float_fmt)
        return

    with open_file(filename, "w") as f:
        write_buffer(f, mesh, float_fmt)


def write_buffer(f, mesh, float_fmt):
    _, dimension = mesh.points.shape
    cells_per_dim = [0, 0, 0, 0]

    # Netgen can store one cell_index, i.e., integer cell data. Pick one in
    # mesh.cell_data, and prefer "netgen:index" if present. Unfortunately, netgen cannot
    # store the name of the data; when reading, it will always be "netgen:index".
    # See also <https://github.com/nschloe/meshio/issues/1199>.
    if "netgen:index" in mesh.cell_data:
        cells_index = mesh.cell_data["netgen:index"]
    else:
        # any other integer cell data?
        cells_index = None
        for values in mesh.cell_data.values():
            if np.issubdtype(values[0].dtype, np.integer):
                cells_index = values
                break

        if cells_index is None:
            cells_index = [None] * len(mesh.cells)

    for cell_block in mesh.cells:
        cells_per_dim[cell_block.dim] += len(cell_block)

    f.write(f"# Generated by meshio {__version__}\n")
    f.write("mesh3d\n\n")

    f.write("dimension\n")
    f.write(f"{dimension}\n\n")

    f.write("geomtype\n")
    f.write("0\n")

    f.write("\n# surfnr    bcnr   domin  domout      np      p1      p2      p3\n")
    f.write("surfaceelements\n")
    f.write(f"{cells_per_dim[2]}\n")
    for cell_block, index in zip(mesh.cells, cells_index):
        if cell_block.dim == 2:
            _write_cells(f, cell_block, index)

    f.write("\n#  matnr      np      p1      p2      p3      p4\n")
    f.write("volumeelements\n")
    f.write(f"{cells_per_dim[3]}\n")
    for cell_block, index in zip(mesh.cells, cells_index):
        if cell_block.dim == 3:
            _write_cells(f, cell_block, index)

    f.write(
        "\n# surfid  0   p1   p2   trignum1    trignum2   domin/surfnr1    domout/surfnr2   ednr1   dist1   ednr2   dist2\n",
    )
    f.write("edgesegmentsgi2\n")
    f.write(f"{cells_per_dim[1]}\n")
    for cell_block, index in zip(mesh.cells, cells_index):
        if cell_block.dim == 1:
            _write_cells(f, cell_block, index)

    f.write("\n#          X             Y             Z\n")
    f.write("points\n")
    f.write(f"{len(mesh.points)}\n")

    points = mesh.points
    if dimension != 3:
        points = np.hstack(
            (points, np.zeros((points.shape[0], 3 - dimension), dtype=points.dtype))
        )
    np.savetxt(f, points, "%" + float_fmt)

    f.write("\n#          pnum             index\n")
    f.write("pointelements\n")
    f.write(f"{cells_per_dim[0]}\n")
    for cell_block, index in zip(mesh.cells, cells_index):
        if cell_block.dim == 0:
            _write_cells(f, cell_block, index)

    # currently, there is no better place for identification data
    if isinstance(mesh.info, dict):
        identifications = mesh.info.get("netgen:identifications")
        identificationtypes = mesh.info.get("netgen:identificationtypes")
        if identifications is not None and identificationtypes is not None:
            f.write("\nidentifications\n")
            f.write(f"{identifications.shape[0]}\n")
            np.savetxt(f, identifications, "%d")
            f.write("\nidentificationtypes\n")
            f.write(f"{identificationtypes.size}\n")
            np.savetxt(
                f, identificationtypes.reshape(1, identificationtypes.size), "%d"
            )

    for codim in range(dimension + 1):
        _write_codim_domain_data(f, mesh, cells_index, dimension, codim)

    f.write("\nendmesh\n")


register_format("netgen", [".vol", ".vol.gz"], read, {"netgen": write})
