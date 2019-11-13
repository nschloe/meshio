"""
CGNS <https://cgns.github.io/>

TODO link to specification?
"""
import numpy

from ._mesh import Mesh


def read(filename):
    import h5py

    f = h5py.File(filename, "r")

    x = f["Base"]["Zone1"]["GridCoordinates"]["CoordinateX"][" data"]
    y = f["Base"]["Zone1"]["GridCoordinates"]["CoordinateY"][" data"]
    z = f["Base"]["Zone1"]["GridCoordinates"]["CoordinateZ"][" data"]
    points = numpy.column_stack([x, y, z])

    # f["Base"]["Zone1"]["GridElements"]["ElementRange"][" data"])
    idx_min, idx_max = f["Base"]["Zone1"]["GridElements"]["ElementRange"][" data"]
    data = f["Base"]["Zone1"]["GridElements"]["ElementConnectivity"][" data"]
    cells = numpy.array(data).reshape(idx_max, -1) - 1

    # TODO how to distinguish cell types?
    assert cells.shape[1] == 4
    cells = {"tetra": cells}

    return Mesh(points, cells)


def write(filename, mesh):
    import h5py

    f = h5py.File(filename, "w")

    base = f.create_group("Base")

    # TODO something is missing here

    zone1 = base.create_group("Zone1")
    coords = zone1.create_group("GridCoordinates")

    # write points
    coord_x = coords.create_group("CoordinateX")
    coord_x.create_dataset(" data", data=mesh.points[:, 0])
    coord_y = coords.create_group("CoordinateY")
    coord_y.create_dataset(" data", data=mesh.points[:, 1])
    coord_z = coords.create_group("CoordinateZ")
    coord_z.create_dataset(" data", data=mesh.points[:, 2])

    # write cells
    # TODO write cells other than tetra
    elems = zone1.create_group("GridElements")
    rnge = elems.create_group("ElementRange")
    rnge.create_dataset(" data", data=[1, mesh.cells["tetra"].shape[0]])
    conn = elems.create_group("ElementConnectivity")
    conn.create_dataset(" data", data=mesh.cells["tetra"].reshape(-1) + 1)
    return
