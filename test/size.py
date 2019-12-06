import os

import matplotlib.pyplot as plt
import numpy
import pygalmesh

s = pygalmesh.Ball([0, 0, 0], 1.0)
mesh = pygalmesh.generate_mesh(s, cell_size=1.0e-1, verbose=True)
mesh.cells = {"tetra": mesh.cells["tetra"]}
mesh.point_data = {}
mesh.cell_data = {}

print("num points: {}".format(mesh.points.shape[0]))

formats = {
    "vtu": ".vtu",
    "vtk": ".vtk",
    "gmsh": ".msh",
    "abaqus": ".inp",
    # "ansys": ".ans",
    "cgns": ".cgns",
    "dolfin-xml": ".xml",
    "mdpa": ".mdpa",
    "med": ".med",
    "medit": ".mesh",
    "moab": ".h5m",
    # "obj": ".obj",
    # "ply": ".ply",
    # "stl": ".stl",
    "nastran": ".bdf",
    # "off": ".off",
    # "exodus": ".e",
    "flac3d": ".f3grid",
    "permas": ".dato",
    # "wkt": ".wkt",
}

file_sizes = {}
for name, ext in formats.items():
    mesh.write(f"out{ext}")
    file_sizes[name] = os.stat(f"out{ext}").st_size


mesh.write("out.xdmf")
file_sizes["xdmf"] = os.stat("out.xdmf").st_size + os.stat("out.h5").st_size
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
# plt.show()
plt.savefig("filesizes.svg", transparent=True, bbox_inches="tight")
