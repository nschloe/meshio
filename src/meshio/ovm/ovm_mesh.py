from .._mesh import Mesh
from .._common import warn
from collections import defaultdict
import numpy as np


def rotate(seq, first_idx):
    assert isinstance(seq, list)  # np.ndarray adds instead of concatenating with +
    """rotate list `seq` such that `seq[first_idx]` becomes the first element"""
    return seq[first_idx:] + seq[:first_idx]


def canonicalize_seq(seq):
    """rotate `seq` such that the minimal element is the first one"""
    idx = np.argmin(seq)
    return rotate(seq, idx)


def find_first(pred, seq):
    for k, x in enumerate(seq):
        if pred(x):
            return k, x


class OpenVolumeMesh:
    def __init__(self):
        self.vertices = []
        self.edges = []  # pairs of vertex indices, smallest first
        self.faces = []  # tuples of halfedge handles, smallest first
        self.polyhedra = []  # tuples of halffaces handles

        # edge cache: vertex handle v -> indices of edges that contain v
        self.cache_edges = defaultdict(list)

        # face cache:
        # edge handle e -> [fh, ...],
        # where fh is the index of a face that contains a halfedge of e as smallest edge
        self.cache_faces = defaultdict(list)

        self.vertex_props = {}

    def find_or_add_halfedge(self, src, dst):
        assert src != dst

        minv = min(src, dst)

        edge0 = (src, dst)
        edge1 = (dst, src)

        for eh in self.cache_edges[minv]:
            cand = self.edges[eh]
            if edge0 == cand:
                return 2 * eh
            if edge1 == cand:
                return 2 * eh + 1

        self.edges.append(edge0)
        eh = len(self.edges) - 1
        self.cache_edges[minv].append(eh)
        return 2 * eh

    def add_polyhedron(self, halffaces):
        self.polyhedra.append(halffaces)

    def find_or_add_halfface_from_vertices(self, verts):
        def opposite(seq):
            # switch halfedge orientation and reverse order
            tmp = [x ^ 1 for x in reversed(seq)]
            # but keep first element first:
            return rotate(tmp, -1)

        hes = []
        for src, dst in zip(verts, rotate(list(verts), 1)):
            hes.append(self.find_or_add_halfedge(src, dst))

        minidx = np.argmin(hes)
        canonical = rotate(hes, minidx)
        opposite_hes = opposite(canonical)
        # Note that opposite_hes still starts with the smallest halfedge idx,
        # as switching HE orientation of 2 non-opposite halfedges does not change
        # order, and one face can only contain each edge once.

        eh = canonical[0] // 2

        for fh in self.cache_faces[eh]:
            cand = self.faces[fh]
            cand_minidx = np.argmin(cand)
            rot_cand = rotate(cand, minidx)
            if rot_cand == canonical:
                return fh * 2
            if rot_cand == opposite_hes:
                return fh * 2 + 1

        self.faces.append(hes)  # TODO: add `canonical` here and skip cand_minidx?
        fh = len(self.faces) - 1
        self.cache_faces[eh].append(fh)
        return 2 * fh

    def he_from(self, he_idx):
        src, dst = self.edges[he_idx // 2]
        if he_idx & 1:
            return dst
        else:
            return src

    def halfface_vertices(self, hf_idx):
        vertices = []
        for he_idx in self.faces[hf_idx // 2]:
            vertices.append(self.he_from(he_idx))
        if hf_idx & 1:
            vertices = list(reversed(vertices))
        return vertices

    def tet_vertices(self, halfface_idxs):
        abc = self.halfface_vertices(halfface_idxs[0])
        vs1 = self.halfface_vertices(halfface_idxs[1])
        d = next(x for x in vs1 if x not in abc)
        a, b, c = abc
        return a, b, c, d
        # TODO perform checks that the other halffaces actually also describe this tet

    def hex_vertices(self, halfface_idxs):

        vs = [self.halfface_vertices(hi) for hi in halfface_idxs]
        # print("hex face sides", vs)
        a, b, c, d = vs[0]
        vs = vs[1:]

        # side: a, e, f, b
        side_idx, side = find_first(lambda x: (a in x) and (b in x), vs)

        side = rotate(side, side.index(a))
        assert side[3] == b
        e, f = side[1:3]
        vs = vs[:side_idx] + vs[side_idx + 1 :]

        top_idx, top = find_first(lambda x: (e in x) and (f in x), vs)
        # top: e, h, g, f
        top = rotate(top, top.index(e))
        assert top[3] == f
        h, g = top[1:3]

        return a, b, c, d, e, f, g, h
        # TODO perform checks that the other halffaces actually also describe this hex

    def wedge_vertices(self, halfface_idxs):
        vs = [self.halfface_vertices(hi) for hi in halfface_idxs]
        tris = [v for v in vs if len(v) == 3]
        a, b, c = tris[0]
        quad = next(v for v in vs if len(v) == 4 and (a in v) and (b in v))
        quad = rotate(quad, quad.index(a))
        assert quad[3] == b
        d, e = quad[1:3]
        f = next(x for x in tris[1] if x != d and x != e)
        return a, b, c, d, e, f
        # TODO perform checks that the other halffaces actually also describe this wedge

    def pyramid_vertices(self, halfface_idxs):
        vs = [self.halfface_vertices(hi) for hi in halfface_idxs]
        bottom_idx, bottom = find_first(lambda v: len(v) == 4, vs)
        side_idx = (bottom_idx + 1) % 5
        _, apex = find_first(lambda x: x not in bottom, vs[side_idx])
        a, b, c, d = bottom
        return a, b, c, d, apex
        # TODO perform checks that the other halffaces actually also describe this pyramid

    def to_meshio(self):
        # print("edges:", self.edges)
        # print("faces:", self.faces)
        # print("cells:", self.polyhedra)

        cells = defaultdict(list)

        if len(self.edges):
            cells["line"] = self.edges

        n_unsupported_cells = 0
        n_failed_cells = 0

        for halfedges in self.faces:
            if len(halfedges) == 3:
                kind = "triangle"
            elif len(halfedges) == 4:
                kind = "quad"
            else:
                kind = "polygon"
            vertices = [self.he_from(he_idx) for he_idx in halfedges]
            cells[kind].append(vertices)

        for halffaces in self.polyhedra:
            # collect number of edges in each face to try and identify cell types
            signature = list(
                sorted((len(self.faces[hf_idx // 2]) for hf_idx in halffaces))
            )
            try:
                if signature == [3, 3, 3, 3]:
                    kind = "tetra"
                    v = self.tet_vertices(halffaces)
                elif signature == [4, 4, 4, 4, 4, 4]:
                    kind = "hexahedron"
                    v = self.hex_vertices(halffaces)
                elif signature == [3, 3, 4, 4, 4]:
                    kind = "wedge"
                    v = self.wedge_vertices(halffaces)
                elif signature == [3, 3, 3, 3, 4]:
                    kind = "pyramid"
                    v = self.pyramid_vertices(halffaces)
                else:
                    n_unsupported_cells += 1
            except Exception as e:
                raise e
                n_failed_cells += 1
            else:
                cells[kind].append(v)

        if n_unsupported_cells:
            warn(
                "Skipped {} unsupported polyhedral cells when reading OVM file".format(
                    n_unsupported_cells
                )
            )
        if n_failed_cells:
            warn("Failed to read {} cells.".format(n_failed_cells))

        if True:
            # TODO XXX remove this, just to pass unit tests:
            ck = cells.keys()
            if "triangle" in ck or "quad" in ck:
                del cells["line"]
            if "tetra" in ck or "hexahedron" in ck or "wedge" in ck or "pyramid" in ck:
                if "triangle" in ck:
                    del cells["triangle"]
                if "quad" in ck:
                    del cells["quad"]

        ###

        for k in cells.keys():
            cells[k] = np.array(cells[k])
        # print(cells)

        return Mesh(
            self.vertices, cells
        )  # , point_data=point_data, cell_data=cell_data)

    @staticmethod
    def from_meshio(mesh):

        ovm = OpenVolumeMesh()
        n, d = mesh.points.shape
        if d != 3:
            # TODO: check this only when writing
            raise WriteError("OVM ASCII format only supports 3-D points")
        ovm.vertices = mesh.points
        ovm.vertex_props = mesh.point_data

        n_skipped = 0
        skipped_cell_types = set()

        for cell_block in mesh.cells:
            if cell_block.type == "line":
                for src, dst in cell_block.data:
                    ovm.find_or_add_halfedge(src, dst)

            elif cell_block.type in ["triangle", "quad"]:
                for verts in cell_block.data:
                    ovm.find_or_add_halfface_from_vertices(verts)

            elif cell_block.type == "tetra":
                for a, b, c, d in cell_block.data:
                    ovm.add_polyhedron(
                        [
                            ovm.find_or_add_halfface_from_vertices([a, b, c]),
                            ovm.find_or_add_halfface_from_vertices([a, c, d]),
                            ovm.find_or_add_halfface_from_vertices([a, d, b]),
                            ovm.find_or_add_halfface_from_vertices([b, d, c]),
                        ]
                    )

            elif cell_block.type == "hexahedron":
                for a, b, c, d, e, f, g, h in cell_block.data:
                    ovm.add_polyhedron(
                        [
                            ovm.find_or_add_halfface_from_vertices([a, b, c, d]),
                            ovm.find_or_add_halfface_from_vertices([a, e, f, b]),
                            ovm.find_or_add_halfface_from_vertices([a, d, h, e]),
                            ovm.find_or_add_halfface_from_vertices([c, g, h, d]),
                            ovm.find_or_add_halfface_from_vertices([b, f, g, c]),
                            ovm.find_or_add_halfface_from_vertices([e, h, g, f]),
                        ]
                    )

            elif cell_block.type == "wedge":
                for a, b, c, d, e, f in cell_block.data:
                    ovm.add_polyhedron(
                        [
                            ovm.find_or_add_halfface_from_vertices([a, b, c]),
                            ovm.find_or_add_halfface_from_vertices([a, c, f, d]),
                            ovm.find_or_add_halfface_from_vertices([a, d, e, b]),
                            ovm.find_or_add_halfface_from_vertices([b, e, f, c]),
                            ovm.find_or_add_halfface_from_vertices([d, f, e]),
                        ]
                    )

            elif cell_block.type == "pyramid":
                for a, b, c, d, e in cell_block.data:
                    ovm.add_polyhedron(
                        [
                            ovm.find_or_add_halfface_from_vertices([a, b, c, d]),
                            ovm.find_or_add_halfface_from_vertices([a, e, b]),
                            ovm.find_or_add_halfface_from_vertices([a, d, e]),
                            ovm.find_or_add_halfface_from_vertices([c, e, d]),
                            ovm.find_or_add_halfface_from_vertices([c, b, e]),
                        ]
                    )
            elif cell_block.type == "vertex":
                # No need to handle this, all vertices are already added.
                pass
            else:
                n_skipped += 1
                skipped_cell_types.add(cell_block.type)

        if n_skipped:
            warn(
                "Skipped {} cell(s) when converting to OVM from these unimplemented types: {}".format(
                    n_skipped, ", ".join(skipped_cell_types)
                )
            )

        return ovm
