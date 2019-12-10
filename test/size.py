import os

import meshio
import matplotlib.pyplot as plt
import numpy
import pygalmesh

s = pygalmesh.Ball([0, 0, 0], 1.0)
mesh = pygalmesh.generate_mesh(s, cell_size=3.0e-2, verbose=True)
mesh.cells = {"tetra": mesh.cells["tetra"]}
mesh.point_data = {}
mesh.cell_data = {}

print("num points: {}".format(mesh.points.shape[0]))

formats = {
    "VTU (binary)": (lambda f, m: meshio.vtu.write(f, m, binary=True), ["out.vtu"]),
    "VTU (ASCII)": (lambda f, m: meshio.vtu.write(f, m, binary=False), ["out.vtu"]),
    "VTK (binary)": (lambda f, m: meshio.vtk.write(f, m, binary=True), ["out.vtk"]),
    "VTK (ASCII)": (lambda f, m: meshio.vtk.write(f, m, binary=False), ["out.vtk"]),
    "Gmsh 4.1 (binary)": (
        lambda f, m: meshio.gmsh.write(f, m, binary=True),
        ["out.msh"],
    ),
    "Gmsh 4.1 (ASCII)": (
        lambda f, m: meshio.gmsh.write(f, m, binary=False),
        ["out.msh"],
    ),
    "Abaqus": (meshio.abaqus.write, ["out.inp"]),
    # "ansys": ".ans",
    "CGNS": (meshio.cgns.write, ["out.cgns"]),
    "Dolfin-XML": (meshio.dolfin.write, ["out.xml"]),
    "MDPA": (meshio.mdpa.write, ["out.mdpa"]),
    "med": (meshio.med.write, ["out.med"]),
    "Medit": (meshio.medit.write, ["out.mesh"]),
    "MOAB": (meshio.h5m.write, ["out.h5m"]),
    # # "obj": ".obj",
    # # "ply": ".ply",
    # # "stl": ".stl",
    "Nastran": (meshio.nastran.write, ["out.bdf"]),
    # # "off": ".off",
    # # "exodus": ".e",
    "FLAC3D": (meshio.flac3d.write, ["out.f3grid"]),
    "Permas": (meshio.permas.write, ["out.dato"]),
    # # "wkt": ".wkt",
    "XDMF (XML)": (
        lambda f, m: meshio.xdmf.write(f, m, data_format="XML"),
        ["out.xdmf"],
    ),
    "XDMF (HDF, uncompressed)": (
        lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression=None),
        ["out.xdmf", "out.h5"],
    ),
    "XDMF (HDF, GZIP)": (
        lambda f, m: meshio.xdmf.write(f, m, data_format="HDF", compression="gzip"),
        ["out.xdmf", "out.h5"],
    ),
    "XDMF (binary)": (
        lambda f, m: meshio.xdmf.write(f, m, data_format="Binary"),
        ["out.xdmf", "out0.bin", "out1.bin"],
    ),
    "TetGen": (meshio.tetgen.write, ["out.node", "out.ele"],),
}

file_sizes = {}
for name, (fun, filenames) in formats.items():
    fun(filenames[0], mesh)
    file_sizes[name] = sum(os.stat(f).st_size for f in filenames)

mem_size = mesh.points.nbytes + mesh.cells["tetra"].nbytes

labels = list(file_sizes.keys())
file_sizes = list(file_sizes.values())
file_sizes = numpy.array(file_sizes)
# convert to MB
file_sizes = file_sizes / 1024.0 ** 2
mem_size /= 1024.0 ** 2

idx = numpy.argsort(file_sizes)
file_sizes = [file_sizes[i] for i in idx]
labels = [labels[i] for i in idx]

plt.bar(range(len(file_sizes)), file_sizes, align="center")
xlim = plt.gca().get_xlim()
plt.plot([-2, len(file_sizes) + 2], [mem_size, mem_size], "C3", linewidth=2.0, zorder=0)
plt.gca().set_xlim(xlim)
plt.gca().set_xticks(range(len(file_sizes)))
plt.gca().set_xticklabels(labels, rotation=90)
plt.gca().set_ylabel("file size [MB]")
plt.gca().set_title("file sizes")
plt.grid()
# plt.show()
plt.savefig("filesizes.svg", transparent=True, bbox_inches="tight")
