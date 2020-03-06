import numpy as np
from .._exceptions import ReadError
from .ovm_mesh import OpenVolumeMesh


def ovm_ascii_write(ovm: OpenVolumeMesh, fh, float_fmt: str):
    vertex_fmt = " ".join(["{:" + float_fmt + "}"] * 3)

    def writeline(line):
        fh.write(str(line) + "\n")

    def encode_vertex(v):
        return vertex_fmt.format(*v)

    def encode_edge(e):
        return "{} {}".format(*e)

    def encode_face(f):
        return str(len(f)) + " " + " ".join(str(he) for he in f)

    def encode_polyhedron(p):
        return str(len(p)) + " " + " ".join(str(hf) for hf in p)

    fh.write("OVM ASCII\n")
    fh.write("Vertices\n")
    writeline(len(ovm.vertices))
    fh.writelines(encode_vertex(v) + "\n" for v in ovm.vertices)

    fh.write("Edges\n")
    writeline(len(ovm.edges))
    fh.writelines(encode_edge(e) + "\n" for e in ovm.edges)
    fh.write("Faces\n")
    writeline(len(ovm.faces))
    fh.writelines(encode_face(f) + "\n" for f in ovm.faces)
    fh.write("Polyhedra\n")
    writeline(len(ovm.polyhedra))
    fh.writelines(encode_polyhedron(p) + "\n" for p in ovm.polyhedra)


class OVMReader:
    def __init__(self, f):
        self.f = f
        self.mesh = OpenVolumeMesh()

    def getline(self):
        return self.f.readline().strip()

    def section_header(self, kind):
        line = self.getline()
        if line.lower() != kind:
            raise ReadError("missing '{}' section".format(kind))

        count = int(self.getline())
        if count < 0:
            raise ReadError("Negative {} count".format(kind))

        return count

    def read_vertices(self):
        DIM = 3
        n_vertices = self.section_header("vertices")

        return np.fromfile(
            self.f, count=n_vertices * DIM, dtype=float, sep=" "
        ).reshape(n_vertices, DIM)

    def read_edges(self):
        n_edges = self.section_header("edges")

        return np.fromfile(self.f, count=n_edges * 2, dtype=int, sep=" ").reshape(
            n_edges, 2
        )

    def read_faces(self):
        def read_face():
            line = [int(x) for x in self.getline().split(" ")]
            n_halfedges = int(line[0])
            if len(line) - 1 != n_halfedges:
                raise ReadError(
                    "Encountered face which should have {} halfedges, but {} halfedge indices specified.".format(
                        len(line) - 1, n_halfedges
                    )
                )

            return line[1:]

        n_faces = self.section_header("faces")
        return [read_face() for _ in range(n_faces)]

    def read_polyhedra(self):
        def read_polyhedron():
            line = [int(x) for x in self.getline().split(" ")]
            n_halffaces = int(line[0])
            if len(line) - 1 != n_halffaces:
                raise ReadError(
                    "Encountered polyhedron which should have {} halffaces, but {} halfface indices specified.".format(
                        len(line) - 1, n_halffaces
                    )
                )

            return line[1:]

        n_polyhedra = self.section_header("polyhedra")

        return [read_polyhedron() for _ in range(n_polyhedra)]

    def read(self):
        line = self.f.readline().strip()
        if line != "OVM ASCII":
            raise ReadError("Not a OVM file, unknown header line {!r}.".format(line))

        self.mesh.vertices = self.read_vertices()
        self.mesh.edges = self.read_edges()
        self.mesh.faces = self.read_faces()
        self.mesh.polyhedra = self.read_polyhedra()

        return self.mesh


def ovm_ascii_read(fh):
    return OVMReader(fh).read().to_meshio()
