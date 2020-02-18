import os
import tempfile
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy
import meshzoo

import meshio


def generate_triangular_mesh():
    if os.path.isfile("sphere.xdmf"):
        mesh = meshio.read("sphere.xdmf")
    else:
        points, cells = meshzoo.icosa_sphere(300)
        mesh = meshio.Mesh(points, {"triangle": cells})
        mesh.write("sphere.xdmf")
    return mesh


def generate_tetrahedral_mesh():
    """Generates a fairly large mesh.
    """
    if os.path.isfile("cache.xdmf"):
        mesh = meshio.read("cache.xdmf")
    else:
        import pygalmesh

        s = pygalmesh.Ball([0, 0, 0], 1.0)
        mesh = pygalmesh.generate_mesh(s, cell_size=2.0e-2, verbose=True)
        # mesh = pygalmesh.generate_mesh(s, cell_size=1.0e-1, verbose=True)
        mesh.cells = {"tetra": mesh.cells["tetra"]}
        mesh.point_data = []
        mesh.cell_data = {"tetra": {}}
        mesh.write("cache.xdmf")
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
    # plt.show()
    fig.savefig("performance.svg", transparent=True, bbox_inches="tight")
    plt.close()


def plot_file_sizes(names, file_sizes, mem_size):
    idx = numpy.argsort(file_sizes)
    file_sizes = [file_sizes[i] for i in idx]
    names = [names[i] for i in idx]

    ax = plt.gca()
    y_pos = numpy.arange(len(file_sizes))
    ax.barh(y_pos, file_sizes, align="center")
    #
    ylim = ax.get_ylim()
    plt.plot(
        [mem_size, mem_size], [-2, len(file_sizes) + 2], "C3", linewidth=2.0, zorder=0
    )
    ax.set_ylim(ylim)
    #
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("file size [MB]")
    ax.set_title("file sizes")
    plt.grid()
    # plt.show()
    plt.savefig("filesizes.svg", transparent=True, bbox_inches="tight")
    plt.close()


def plot_memory_usage(names, peak_memory_write, peak_memory_read, mem_size):
    names = numpy.asarray(names)
    peak_memory_write = numpy.asarray(peak_memory_write)
    peak_memory_read = numpy.asarray(peak_memory_read)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    idx = numpy.argsort(peak_memory_write)[::-1]
    ax[0].barh(range(len(names)), peak_memory_write[idx], align="center")
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names[idx])
    ax[0].set_xlabel("peak memory [MB]")
    ax[0].set_title("write")
    ax[0].grid()
    # plot memsize of mesh
    ylim = ax[0].get_ylim()
    ax[0].plot(
        [mem_size, mem_size], [-2, len(names) + 2], "C3", linewidth=2.0, zorder=0
    )
    ax[0].set_ylim(ylim)

    idx = numpy.argsort(peak_memory_read)[::-1]
    ax[1].barh(range(len(names)), peak_memory_read[idx], align="center")
    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names[idx])
    ax[1].set_xlabel("peak memory [MB]")
    ax[1].set_title("read")
    ax[1].grid()
    # plot memsize of mesh
    ylim = ax[1].get_ylim()
    ax[1].plot(
        [mem_size, mem_size], [-2, len(names) + 2], "C3", linewidth=2.0, zorder=0
    )
    ax[1].set_ylim(ylim)

    fig.tight_layout()
    # plt.show()
    fig.savefig("memory.svg", transparent=True, bbox_inches="tight")
    plt.close()


def read_write(plot=False):
    # mesh = generate_tetrahedral_mesh()
    mesh = generate_triangular_mesh()
    print(mesh)
    mem_size = mesh.points.nbytes + mesh.cells[0].data.nbytes
    mem_size /= 1024.0 ** 2
    print("mem_size: {:.2f} MB".format(mem_size))

    formats = {
        "Abaqus": (meshio.abaqus.write, meshio.abaqus.read, ["out.inp"]),
        "Ansys (ASCII)": (
            lambda f, m: meshio.ansys.write(f, m, binary=False),
            meshio.ansys.read,
            ["out.ans"],
        ),
        # "Ansys (binary)": (
        #     lambda f, m: meshio.ansys.write(f, m, binary=True),
        #     meshio.ansys.read,
        #     ["out.ans"],
        # ),
        "AVS-UCD": (meshio.avsucd.write, meshio.avsucd.read, ["out.ucd"]),
        # "CGNS": (meshio.cgns.write, meshio.cgns.read, ["out.cgns"]),
        "Dolfin-XML": (meshio.dolfin.write, meshio.dolfin.read, ["out.xml"]),
        "Exodus": (meshio.exodus.write, meshio.exodus.read, ["out.e"]),
        # "FLAC3D": (meshio.flac3d.write, meshio.flac3d.read, ["out.f3grid"]),
        "Gmsh 4.1 (ASCII)": (
            lambda f, m: meshio.gmsh.write(f, m, binary=False),
            meshio.gmsh.read,
            ["out.msh"],
        ),
        "Gmsh 4.1 (binary)": (
            lambda f, m: meshio.gmsh.write(f, m, binary=True),
            meshio.gmsh.read,
            ["out.msh"],
        ),
        "MDPA": (meshio.mdpa.write, meshio.mdpa.read, ["out.mdpa"]),
        "MED": (meshio.med.write, meshio.med.read, ["out.med"]),
        "Medit": (meshio.medit.write, meshio.medit.read, ["out.mesh"]),
        "MOAB": (meshio.h5m.write, meshio.h5m.read, ["out.h5m"]),
        "Nastran": (meshio.nastran.write, meshio.nastran.read, ["out.bdf"]),
        "OBJ": (meshio.obj.write, meshio.obj.read, ["out.obj"]),
        "OFF": (meshio.off.write, meshio.off.read, ["out.off"]),  # TODO add
        "Permas": (meshio.permas.write, meshio.permas.read, ["out.dato"]),
        "PLY (binary)": (
            lambda f, m: meshio.ply.write(f, m, binary=True),
            meshio.ply.read,
            ["out.ply"],
        ),
        "PLY (ASCII)": (
            lambda f, m: meshio.ply.write(f, m, binary=False),
            meshio.ply.read,
            ["out.ply"],
        ),
        "STL (binary)": (
            lambda f, m: meshio.stl.write(f, m, binary=True),
            meshio.stl.read,
            ["out.stl"],
        ),
        "STL (ASCII)": (
            lambda f, m: meshio.stl.write(f, m, binary=False),
            meshio.stl.read,
            ["out.stl"],
        ),
        # "TetGen": (meshio.tetgen.write, meshio.tetgen.read, ["out.node", "out.ele"],),
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
        "VTU (binary, uncompressed)": (
            lambda f, m: meshio.vtu.write(f, m, binary=True, compression=None),
            meshio.vtu.read,
            ["out.vtu"],
        ),
        "VTU (binary, zlib)": (
            lambda f, m: meshio.vtu.write(f, m, binary=True, compression="zlib"),
            meshio.vtu.read,
            ["out.vtu"],
        ),
        "VTU (binary, LZMA)": (
            lambda f, m: meshio.vtu.write(f, m, binary=True, compression="lzma"),
            meshio.vtu.read,
            ["out.vtu"],
        ),
        "VTU (ASCII)": (
            lambda f, m: meshio.vtu.write(f, m, binary=False),
            meshio.vtu.read,
            ["out.vtu"],
        ),
        "Wavefront .obj": (meshio.obj.write, meshio.obj.read, ["out.obj"]),
        # "wkt": ".wkt",
        "XDMF (binary)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="Binary"),
            meshio.xdmf.read,
            ["out.xdmf", "out0.bin", "out1.bin"],
        ),
        "XDMF (HDF, GZIP)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression="gzip"),
            meshio.xdmf.read,
            ["out.xdmf", "out.h5"],
        ),
        "XDMF (HDF, uncompressed)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression=None),
            meshio.xdmf.read,
            ["out.xdmf", "out.h5"],
        ),
        "XDMF (XML)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="XML"),
            meshio.xdmf.read,
            ["out.xdmf"],
        ),
    }

    # formats = {
    #     # "VTK (ASCII)": formats["VTK (ASCII)"],
    #     # "VTK (binary)": formats["VTK (binary)"],
    #     # "VTU (ASCII)": formats["VTU (ASCII)"],
    #     # "VTU (binary)": formats["VTU (binary)"],
    #     # "Gmsh 4.1 (binary)": formats["Gmsh 4.1 (binary)"],
    #     # "FLAC3D": formats["FLAC3D"],
    #     "MDPA": formats["MDPA"],
    # }

    # max_key_length = max(len(key) for key in formats)

    elapsed_write = []
    elapsed_read = []
    file_sizes = []
    peak_memory_write = []
    peak_memory_read = []

    print()
    print(
        "format                      "
        + "write (s)    "
        + "read(s)      "
        + "file size    "
        + "write mem    "
        + "read mem "
    )
    print()
    with tempfile.TemporaryDirectory() as directory:
        for name, (writer, reader, filenames) in formats.items():
            filename = os.path.join(directory, filenames[0])

            tracemalloc.start()
            t = time.time()
            writer(filename, mesh)
            # snapshot = tracemalloc.take_snapshot()
            elapsed_write.append(time.time() - t)
            peak_memory_write.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

            file_sizes.append(
                sum(os.stat(os.path.join(directory, f)).st_size for f in filenames)
            )

            tracemalloc.start()
            t = time.time()
            reader(filename)
            elapsed_read.append(time.time() - t)
            peak_memory_read.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()
            print(
                "{:<26}  {:e} {:e} {:e} {:e} {:e}".format(
                    name,
                    elapsed_write[-1],
                    elapsed_read[-1],
                    file_sizes[-1] / 1024.0 ** 2,
                    peak_memory_write[-1] / 1024.0 ** 2,
                    peak_memory_read[-1] / 1024.0 ** 2,
                )
            )

    names = list(formats.keys())
    # convert to MB
    file_sizes = numpy.array(file_sizes)
    file_sizes = file_sizes / 1024.0 ** 2
    peak_memory_write = numpy.array(peak_memory_write)
    peak_memory_write = peak_memory_write / 1024.0 ** 2
    peak_memory_read = numpy.array(peak_memory_read)
    peak_memory_read = peak_memory_read / 1024.0 ** 2

    if plot:
        plot_speed(names, elapsed_write, elapsed_read)
        plot_file_sizes(names, file_sizes, mem_size)
        plot_memory_usage(names, peak_memory_write, peak_memory_read, mem_size)


if __name__ == "__main__":
    read_write(plot=True)
