import tempfile
import time

import meshio
import meshzoo


def generate_mesh():
    """Generates a fairly large mesh.
    """
    points, cells = meshzoo.rectangle(nx=300, ny=300)
    return meshio.Mesh(points, {"triangle": cells})


def read_write(plot=False):
    mesh = generate_mesh()

    formats = {
        "Abaqus": (meshio.abaqus.write, meshio.abaqus.read),
        "ANSYS (ASCII)": (
            lambda f, m: meshio.ansys.write(f, m, binary=False),
            meshio.ansys.read,
        ),
        "ANSYS (binary)": (
            lambda f, m: meshio.ansys.write(f, m, binary=True),
            meshio.ansys.read,
        ),
        "Exodus": (meshio.exodus.write, meshio.exodus.read),
        "Dolfin XML": (meshio.dolfin.write, meshio.dolfin.read),
        "Gmsh 4.1 (ASCII)": (
            lambda f, m: meshio.gmsh.write(f, m, binary=False),
            meshio.gmsh.read,
        ),
        "Gmsh 4.1 (binary)": (
            lambda f, m: meshio.gmsh.write(f, m, binary=True),
            meshio.gmsh.read,
        ),
        "MDPA": (meshio.mdpa.write, meshio.mdpa.read),
        "MED": (meshio.med.write, meshio.med.read),
        "Medit": (meshio.medit.write, meshio.medit.read),
        "MOAB": (meshio.h5m.write, meshio.h5m.read),
        "Nastran": (meshio.nastran.write, meshio.nastran.read),
        # "OFF": (meshio.off.write, meshio.off.read),
        "Permas": (meshio.permas.write, meshio.permas.read),
        "PLY (ASCII)": (
            lambda f, m: meshio.ply.write(f, m, binary=False),
            meshio.ply.read,
        ),
        "PLY (binary)": (
            lambda f, m: meshio.ply.write(f, m, binary=True),
            meshio.ply.read,
        ),
        "STL (ASCII)": (
            lambda f, m: meshio.stl.write(f, m, binary=False),
            meshio.stl.read,
        ),
        "STL (binary)": (
            lambda f, m: meshio.stl.write(f, m, binary=True),
            meshio.stl.read,
        ),
        "VTK (ASCII)": (
            lambda f, m: meshio.vtk.write(f, m, binary=False),
            meshio.vtk.read,
        ),
        "VTK (binary)": (
            lambda f, m: meshio.vtk.write(f, m, binary=True),
            meshio.vtk.read,
        ),
        "VTU (ASCII)": (
            lambda f, m: meshio.vtu.write(f, m, binary=False),
            meshio.vtu.read,
        ),
        "VTU (binary)": (
            lambda f, m: meshio.vtu.write(f, m, binary=True),
            meshio.vtu.read,
        ),
        "XDMF (XML)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="XML"),
            meshio.xdmf.read,
        ),
        "XDMF (binary)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="Binary"),
            meshio.xdmf.read,
        ),
        "XDMF (HDF, uncompressed)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression=None),
            meshio.xdmf.read,
        ),
        "XDMF (HDF, GZIP)": (
            lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression="gzip"),
            meshio.xdmf.read,
        ),
    }

    elapsed_write = []
    elapsed_read = []

    print()
    print("format                  write (s)    read(s)")
    print()
    for name, (writer, reader) in formats.items():
        filename = tempfile.NamedTemporaryFile().name
        t = time.time()
        writer(filename, mesh)
        elapsed_write.append(time.time() - t)

        t = time.time()
        reader(filename)
        elapsed_read.append(time.time() - t)
        print("{:<22}  {:e} {:e}".format(name, elapsed_write[-1], elapsed_read[-1]))

    if plot:
        import numpy as np
        import matplotlib.pyplot as plt

        elapsed_write = np.asarray(elapsed_write)
        elapsed_read = np.asarray(elapsed_read)
        formats = np.asarray(list(formats.keys()))

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        idx = np.argsort(elapsed_write)[::-1]
        ax[0].barh(range(len(formats)), elapsed_write[idx], align="center")
        ax[0].set_yticks(range(len(formats)))
        ax[0].set_yticklabels(formats[idx])
        ax[0].set_xlabel("time (s)")
        ax[0].set_title("write")
        ax[0].grid()

        idx = np.argsort(elapsed_read)[::-1]
        ax[1].barh(range(len(formats)), elapsed_read[idx], align="center")
        ax[1].set_yticks(range(len(formats)))
        ax[1].set_yticklabels(formats[idx])
        ax[1].set_xlabel("time (s)")
        ax[1].set_title("read")
        ax[1].grid()

        fig.tight_layout()
        # plt.show()
        fig.savefig("performance.svg", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    read_write(plot=True)
