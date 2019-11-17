"""
PLOT3D format is for structured grids only
For more info, check out: PLOT3D User's Manual, chapter 8: Data File Formats
Author: Samir OUCHENE
"""

import numpy as np
from _mesh import Mesh
import meshio


def read(filename):
    """
    Read the mesh in PLOT3D Format from filename.
    """
    with open(filename, "r") as p3dfile:
        first_line = p3dfile.readline().rstrip("\n").split()
        n = len(first_line)
        assert 2 <= n <= 3, "The first line must contain either 2 or 3 values"
        _2d = False
        if n == 2:
            _2d = True
            imax, jmax = map(int, first_line)
        else:
            imax, jmax, kmax = map(int, first_line)
    if _2d:
        pts_ = np.loadtxt(filename, skiprows=1).reshape((imax * jmax, 2), order="F")
        zero_vec = np.zeros_like(pts_[:, 0]).reshape((imax * jmax, -1))
        pts = np.hstack((pts_, zero_vec))

        cells_ = []
        for j in range(jmax - 1):
            cell = []
            for i in range(imax - 1):
                cell = [i + j * imax, i + 1 + j * imax]
                cell.append(i + (j + 1) * imax + 1)
                cell.append(i + (j + 1) * imax)
                cells_.append(cell)

        cells_ = np.array(cells_, dtype=np.int32)
        cells = {"quad": cells_}
        return Mesh(pts, cells)
    else:
        pts = np.loadtxt(filename, skiprows=1).reshape(
            (imax * jmax * kmax, 3), order="F"
        )
        cells_ = []
        for k in range(kmax - 1):
            cell = []
            for j in range(jmax - 1):
                for i in range(imax - 1):
                    i_ = i + j * imax + k * imax * jmax
                    i_j = i + (j + 1) * imax + k * imax * jmax
                    i_k = i + j * imax + (k + 1) * imax * jmax
                    i_j_k = i + (j + 1) * imax + (k + 1) * imax * jmax
                    cell = [i_, i_ + 1, i_j + 1, i_j, i_k, i_k + 1, i_j_k + 1, i_j_k]
                    cells_.append(cell)
        cells_ = np.array(cells_, dtype=np.int32)
        cells = {"hexahedron": cells_}

        return Mesh(pts, cells)


def write(filename, mesh):
    """
    Write the mesh to filename in PLOT3D format
    """
    points = mesh.points
    imax = np.unique(points[:, 0]).size
    jmax = np.unique(points[:, 1]).size
    kmax = np.unique(points[:, 2]).size
    _2d = True if kmax == 1 else False
    with open(filename, "w") as p3dfile:
        if not _2d:
            print(imax, jmax, kmax, file=p3dfile)
            for value in points.flatten(order="F"):
                print(value, file=p3dfile)
        else:
            print(imax, jmax, file=p3dfile)
            for value in points[:, 0:2].flatten(order="F"):
                print(value, file=p3dfile)


def test_plot3d():
    # Test write to p3d
    mesh = meshio.read("../test/meshes/cavity.vtk")
    write("../test/meshes/cavity.p3d", mesh)
    # test read p3d mesh file
    mesh = read("../test/meshes/naca0018_2d.p3d")
    meshio.write("../test/meshes/naca0018_2d.vtk", mesh)


if __name__ == "__main__":
    test_plot3d()
