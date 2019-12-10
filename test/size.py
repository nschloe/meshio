import os

import meshio
import matplotlib.pyplot as plt
import numpy
import pygalmesh

s = pygalmesh.Ball([0, 0, 0], 1.0)
# mesh = pygalmesh.generate_mesh(s, cell_size=3.0e-2, verbose=True)
mesh = pygalmesh.generate_mesh(s, cell_size=5.0e-2, verbose=True)
mesh.cells = {"tetra": mesh.cells["tetra"]}
mesh.point_data = {}
mesh.cell_data = {}

print("num points: {}".format(mesh.points.shape[0]))

formats = {
    "VTU": (meshio.vtu.write, ".vtu"),
    "VTK": (meshio.vtk.write, ".vtk"),
    "Gmsh 4.1 (binary)": (lambda f, m: meshio.gmsh.write(f, m, binary=True), ".msh"),
    "Gmsh 4.1 (ASCII)": (lambda f, m: meshio.gmsh.write(f, m, binary=False), ".msh"),
    "Abaqus": (meshio.abaqus.write, ".inp"),
    # "ansys": ".ans",
    "CGNS": (meshio.cgns.write, ".cgns"),
    "Dolfin-XML": (meshio.dolfin.write, ".xml"),
    "MDPA": (meshio.mdpa.write, ".mdpa"),
    "med": (meshio.med.write, ".med"),
    "Medit": (meshio.medit.write, ".mesh"),
    "MOAB": (meshio.h5m.write, ".h5m"),
    # # "obj": ".obj",
    # # "ply": ".ply",
    # # "stl": ".stl",
    "Nastran": (meshio.nastran.write, ".bdf"),
    # # "off": ".off",
    # # "exodus": ".e",
    "FLAC3D": (meshio.flac3d.write, ".f3grid"),
    "Permas": (meshio.permas.write, ".dato"),
    # # "wkt": ".wkt",
    "XDMF (XML)": (lambda f, m: meshio.xdmf.write(f, m, data_format="XML"), ".xdmf"),
    "XDMF (binary)": (lambda f, m: meshio.xdmf.write(f, m, data_format="Binary"), ".xdmf"),
}

file_sizes = {}
for name, (fun, ext) in formats.items():
    fun(f"out{ext}", mesh)
    file_sizes[name] = os.stat(f"out{ext}").st_size


meshio.xdmf.write("out.xdmf", mesh, data_format="HDF", compression=None)
file_sizes["XDMF (HDF, uncompressed)"] = os.stat("out.xdmf").st_size + os.stat("out.h5").st_size
meshio.xdmf.write("out.xdmf", mesh, data_format="HDF", compression="gzip")
file_sizes["XDMF (HDF, GZIP)"] = os.stat("out.xdmf").st_size + os.stat("out.h5").st_size
mesh.write("out.node")
file_sizes["tetgen"] = os.stat("out.node").st_size + os.stat("out.ele").st_size

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
plt.show()
# plt.savefig("filesizes.svg", transparent=True, bbox_inches="tight")
