"""
I/O for the Wavefront .obj file format, cf.
<https://en.wikipedia.org/wiki/Wavefront_.obj_file>.
"""
import numpy

from ._mesh import Mesh


def read(filename):
    with open(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    points = []
    faces = []
    while True:
        line = f.readline()

        if not line:
            # EOF
            break

        strip = line.strip()

        if len(strip) == 0 or strip[0] == "#":
            continue

        split = strip.split()

        if split[0] == "v":
            # vertex
            points.append([numpy.float(item) for item in split[1:]])
        elif split[0] == "vn":
            # skip vertex normals
            pass
        elif split[0] == "s":
            # "s 1" or "s off" controls smooth shading
            pass
        elif split[0] == "f":
            faces.append([int(item.split("//")[0]) for item in split[1:]])
        else:
            # who knows
            pass

    return Mesh(numpy.array(points), {"triangle": numpy.array(faces)})


def write(filename, mesh):
    # TODO
    return
