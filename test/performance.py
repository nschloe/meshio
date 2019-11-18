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

    formats = [
        "abaqus",
        "ansys-ascii",
        "ansys-binary",
        "exodus",
        "dolfin-xml",
        "gmsh2-ascii",
        "gmsh2-binary",
        "gmsh4-ascii",
        "gmsh4-binary",
        "mdpa",
        "med",
        "medit",
        # "moab",
        "nastran",
        "off",
        "permas",
        "ply-ascii",
        "ply-binary",
        "stl-ascii",
        "stl-binary",
        "vtk-ascii",
        "vtk-binary",
        "vtu-ascii",
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
        ax[0].grid()

        idx = np.argsort(elapsed_read)
        ax[1].bar(range(len(formats)), elapsed_read[idx], align="center")
        ax[1].set_xticks(range(len(formats)))
        ax[1].set_xticklabels(formats[idx], rotation=90)
        ax[1].set_ylabel("time (s)")
        ax[1].set_title("read")
        ax[1].grid()

        fig.tight_layout()
        fig.savefig("performance.svg", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    read_write(plot=True)
