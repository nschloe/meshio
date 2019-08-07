import tempfile
import time

import meshio


def generate_mesh():
    """Generates a fairly large mesh.
    """
    import pygmsh

    geom = pygmsh.built_in.Geometry()

    geom.add_circle(
        [0.0, 0.0, 0.0],
        1.0,
        5.0e-3,
        # 1.0e-2,
        num_sections=4,
        # If compound==False, the section borders have to be points of the
        # discretization. If using a compound circle, they don't; gmsh can
        # choose by itself where to point the circle points.
        # compound=True,
    )
    mesh = pygmsh.generate_mesh(geom)
    for key in ["vertex", "line"]:  # some formats do not treat 0d/1d elements
        mesh.cells.pop(key, None)
    return mesh


def read_write(plot=False):
    mesh = generate_mesh()

    formats = [
        # "abaqus",
        # "ansys-ascii",
        "ansys-binary",
        "exodus",
        # "dolfin-xml",
        # "gmsh2-ascii",
        "gmsh2-binary",
        "mdpa",
        "med",
        "medit",
        # "moab",
        "nastran",
        "off",
        "permas",
        # "stl-ascii",
        "stl-binary",
        # "vtk-ascii",
        "vtk-binary",
        # "vtu-ascii",
        "vtu-binary",
        "xdmf",
    ]

    elapsed_write = []
    elapsed_read = []

    print()
    print("format        write (s)    read(s)")
    print()
    for fmt in formats:
        filename = tempfile.NamedTemporaryFile().name
        t = time.time()
        meshio.write(filename, mesh, file_format=fmt)
        elapsed_write.append(time.time() - t)

        t = time.time()
        meshio.read(filename, file_format=fmt)
        elapsed_read.append(time.time() - t)
        print("{0: <12}  {1:e} {2:e}".format(fmt, elapsed_write[-1], elapsed_read[-1]))

    if plot:
        import numpy as np
        import matplotlib.pyplot as plt

        elapsed_write = np.asarray(elapsed_write)
        elapsed_read = np.asarray(elapsed_read)
        formats = np.asarray(formats)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        idx = np.argsort(elapsed_write)
        ax[0].bar(range(len(formats)), elapsed_write[idx], align="center")
        ax[0].set_xticks(range(len(formats)))
        ax[0].set_xticklabels(formats[idx], rotation=90)
        ax[0].set_ylabel("time (s)")
        ax[0].set_title("write")

        idx = np.argsort(elapsed_read)
        ax[1].bar(range(len(formats)), elapsed_read[idx], align="center")
        ax[1].set_xticks(range(len(formats)))
        ax[1].set_xticklabels(formats[idx], rotation=90)
        ax[1].set_ylabel("time (s)")
        ax[1].set_title("read")

        fig.tight_layout()
        fig.savefig("performance.png")


if __name__ == "__main__":
    read_write(plot=True)
