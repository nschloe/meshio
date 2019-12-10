import os
import tempfile
import time

import matplotlib.pyplot as plt
import meshio
import numpy
import pygalmesh


def generate_mesh():
    """Generates a fairly large mesh.
    """
    # import meshzoo
    # points, cells = meshzoo.rectangle(nx=300, ny=300)
    # return meshio.Mesh(points, {"triangle": cells})
    s = pygalmesh.Ball([0, 0, 0], 1.0)
    mesh = pygalmesh.generate_mesh(s, cell_size=3.0e-2, verbose=True)
    # mesh = pygalmesh.generate_mesh(s, cell_size=1.0e-1, verbose=True)
    mesh.cells = {"tetra": mesh.cells["tetra"]}
    mesh.point_data = {}
    mesh.cell_data = {}
    return mesh


def plot_speed(names, elapsed_write, elapsed_read):
    names = numpy.asarray(names)
    elapsed_write = numpy.asarray(elapsed_write)
    elapsed_read = numpy.asarray(elapsed_read)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    idx = numpy.argsort(elapsed_write)[::-1]
    ax[0].barh(range(len(names)), elapsed_write[idx], align="center")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names[idx])
    ax[0].set_xlabel("time (s)")
    ax[0].set_title("write")
    ax[0].grid()

    idx = numpy.argsort(elapsed_read)[::-1]
    ax[1].barh(range(len(names)), elapsed_read[idx], align="center")
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names[idx])
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("read")
    ax[1].grid()

    fig.tight_layout()
    plt.show()
    # fig.savefig("performance.svg", transparent=True, bbox_inches="tight")


def plot_file_sizes(names, file_sizes, mem_size):
    idx = numpy.argsort(file_sizes)
    file_sizes = [file_sizes[i] for i in idx]
    names = [names[i] for i in idx]

    ax = plt.gca()
    y_pos = numpy.arange(len(file_sizes))
    ax.barh(y_pos, file_sizes, align="center")
    #
    ylim = ax.get_ylim()
    plt.plot([mem_size, mem_size], [-2, len(file_sizes) + 2], "C3", linewidth=2.0, zorder=0)
    ax.set_ylim(ylim)
    #
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("file size [MB]")
    ax.set_title("file sizes")
    plt.grid()
    plt.show()
    # plt.savefig("filesizes.svg", transparent=True, bbox_inches="tight")


def read_write(plot=False):
    mesh = generate_mesh()

    formats = {
        "VTU (binary)": (
            lambda f, m: meshio.vtu.write(f, m, binary=True),
            meshio.vtu.read,
            ["out.vtu"],
        ),
        "VTU (ASCII)": (
            lambda f, m: meshio.vtu.write(f, m, binary=False),
            meshio.vtu.read,
            ["out.vtu"],
        ),
        "VTK (binary)": (
            lambda f, m: meshio.vtk.write(f, m, binary=True),
            meshio.vtk.read,
            ["out.vtk"],
        ),
        "VTK (ASCII)": (
            lambda f, m: meshio.vtk.write(f, m, binary=False),
            meshio.vtk.read,
            ["out.vtk"],
        ),
        "Gmsh 4.1 (binary)": (
            lambda f, m: meshio.gmsh.write(f, m, binary=True),
            meshio.gmsh.read,
            ["out.msh"],
        ),
        "Gmsh 4.1 (ASCII)": (
            lambda f, m: meshio.gmsh.write(f, m, binary=False),
            meshio.gmsh.read,
            ["out.msh"],
        ),
        "Abaqus": (meshio.abaqus.write, meshio.abaqus.read, ["out.inp"]),
        # "ansys": ".ans",
        "CGNS": (meshio.cgns.write, meshio.cgns.read, ["out.cgns"]),
        "Dolfin-XML": (meshio.dolfin.write, meshio.dolfin.read, ["out.xml"]),
        "MDPA": (meshio.mdpa.write, meshio.mdpa.read, ["out.mdpa"]),
        "med": (meshio.med.write, meshio.med.read, ["out.med"]),
        "Medit": (meshio.medit.write, meshio.medit.read, ["out.mesh"]),
        "MOAB": (meshio.h5m.write, meshio.h5m.read, ["out.h5m"]),
        # # "obj": ".obj",
        # # "ply": ".ply",
        # # "stl": ".stl",
        "Nastran": (meshio.nastran.write, meshio.nastran.read, ["out.bdf"]),
        # # "off": ".off",
        # # "exodus": ".e",
        "FLAC3D": (meshio.flac3d.write, meshio.flac3d.read, ["out.f3grid"]),
        "Permas": (meshio.permas.write, meshio.permas.read, ["out.dato"]),
        # # "wkt": ".wkt",
        "XDMF (XML)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="XML"),
            meshio.xdmf.read,
            ["out.xdmf"],
        ),
        "XDMF (HDF, uncompressed)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression=None),
            meshio.xdmf.read,
            ["out.xdmf", "out.h5"],
        ),
        "XDMF (HDF, GZIP)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression="gzip"),
            meshio.xdmf.read,
            ["out.xdmf", "out.h5"],
        ),
        "XDMF (binary)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="Binary"),
            meshio.xdmf.read,
            ["out.xdmf", "out0.bin", "out1.bin"],
        ),
        "TetGen": (meshio.tetgen.write, meshio.tetgen.read, ["out.node", "out.ele"],),
    }

    # formats = {
    #     "Abaqus": (meshio.abaqus.write, meshio.abaqus.read),
    #     "ANSYS (ASCII)": (
    #         lambda f, m: meshio.ansys.write(f, m, binary=False),
    #         meshio.ansys.read,
    #     ),
    #     "ANSYS (binary)": (
    #         lambda f, m: meshio.ansys.write(f, m, binary=True),
    #         meshio.ansys.read,
    #     ),
    #     "Exodus": (meshio.exodus.write, meshio.exodus.read),
    #     "Dolfin XML": (meshio.dolfin.write, meshio.dolfin.read),
    #     "Gmsh 4.1 (ASCII)": (
    #         lambda f, m: meshio.gmsh.write(f, m, binary=False),
    #         meshio.gmsh.read,
    #     ),
    #     "Gmsh 4.1 (binary)": (
    #         lambda f, m: meshio.gmsh.write(f, m, binary=True),
    #         meshio.gmsh.read,
    #     ),
    #     "MDPA": (meshio.mdpa.write, meshio.mdpa.read),
    #     "MED": (meshio.med.write, meshio.med.read),
    #     "Medit": (meshio.medit.write, meshio.medit.read),
    #     "MOAB": (meshio.h5m.write, meshio.h5m.read),
    #     "Nastran": (meshio.nastran.write, meshio.nastran.read),
    #     # "OFF": (meshio.off.write, meshio.off.read),
    #     "Permas": (meshio.permas.write, meshio.permas.read),
    #     "PLY (ASCII)": (
    #         lambda f, m: meshio.ply.write(f, m, binary=False),
    #         meshio.ply.read,
    #     ),
    #     "PLY (binary)": (
    #         lambda f, m: meshio.ply.write(f, m, binary=True),
    #         meshio.ply.read,
    #     ),
    #     "STL (ASCII)": (
    #         lambda f, m: meshio.stl.write(f, m, binary=False),
    #         meshio.stl.read,
    #     ),
    #     "STL (binary)": (
    #         lambda f, m: meshio.stl.write(f, m, binary=True),
    #         meshio.stl.read,
    #     ),
    #     "VTK (ASCII)": (
    #         lambda f, m: meshio.vtk.write(f, m, binary=False),
    #         meshio.vtk.read,
    #     ),
    #     "VTK (binary)": (
    #         lambda f, m: meshio.vtk.write(f, m, binary=True),
    #         meshio.vtk.read,
    #     ),
    #     "VTU (ASCII)": (
    #         lambda f, m: meshio.vtu.write(f, m, binary=False),
    #         meshio.vtu.read,
    #     ),
    #     "VTU (binary)": (
    #         lambda f, m: meshio.vtu.write(f, m, binary=True),
    #         meshio.vtu.read,
    #     ),
    #     "XDMF (XML)": (
    #         lambda f, m: meshio.xdmf.write(f, m, data_format="XML"),
    #         meshio.xdmf.read,
    #     ),
    #     "XDMF (binary)": (
    #         lambda f, m: meshio.xdmf.write(f, m, data_format="Binary"),
    #         meshio.xdmf.read,
    #     ),
    #     "XDMF (HDF, uncompressed)": (
    #         lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression=None),
    #         meshio.xdmf.read,
    #     ),
    #     "XDMF (HDF, GZIP)": (
    #         lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression="gzip"),
    #         meshio.xdmf.read,
    #     ),
    # }

    elapsed_write = []
    elapsed_read = []
    file_sizes = []

    print()
    print("format                  write (s)    read(s)   file size")
    print()
    with tempfile.TemporaryDirectory() as directory:
        for name, (writer, reader, filenames) in formats.items():
            filename = os.path.join(directory, filenames[0])
            t = time.time()
            writer(filename, mesh)
            elapsed_write.append(time.time() - t)

            file_sizes.append(sum(os.stat(os.path.join(directory, f)).st_size for f in filenames))

            t = time.time()
            reader(filename)
            elapsed_read.append(time.time() - t)
            print("{:<22}  {:e} {:e}".format(name, elapsed_write[-1], elapsed_read[-1]))

    names = list(formats.keys())
    # convert to MB
    file_sizes = numpy.array(file_sizes)
    file_sizes = file_sizes / 1024.0 ** 2
    mem_size = mesh.points.nbytes + mesh.cells["tetra"].nbytes
    mem_size /= 1024.0 ** 2

    if plot:
        plot_speed(names, elapsed_write, elapsed_read)
        plot_file_sizes(names, file_sizes, mem_size)


if __name__ == "__main__":
    read_write(plot=True)
