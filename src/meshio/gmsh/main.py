import pathlib
import struct
import numpy as np
import copy

from .._mesh import CellBlock, Mesh
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from . import _gmsh22, _gmsh40, _gmsh41
from .common import _fast_forward_to_end_block

# Some mesh files out there have the version specified as version "2" when it really is
# "2.2". Same with "4" vs "4.1".
_readers = {"2": _gmsh22, "2.2": _gmsh22, "4.0": _gmsh40, "4": _gmsh41, "4.1": _gmsh41}
_writers = {"2.2": _gmsh22, "4.0": _gmsh40, "4.1": _gmsh41}


def read(filename):
    """Reads a Gmsh msh file."""
    filename = pathlib.Path(filename)
    with open(filename.as_posix(), "rb") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    # The various versions of the format are specified at
    # <http://gmsh.info/doc/texinfo/gmsh.html#File-formats>.
    line = f.readline().decode().strip()

    # skip any $Comments/$EndComments sections
    while line == "$Comments":
        _fast_forward_to_end_block(f, "Comments")
        line = f.readline().decode().strip()

    if line != "$MeshFormat":
        raise ReadError()
    fmt_version, data_size, is_ascii = _read_header(f)

    try:
        reader = _readers[fmt_version]
    except KeyError:
        try:
            reader = _readers[fmt_version.split(".")[0]]
        except KeyError:
            raise ValueError(
                "Need mesh format in {} (got {})".format(
                    sorted(_readers.keys()), fmt_version
                )
            )
    return reader.read_buffer(f, is_ascii, data_size)


def _read_header(f):
    """Read the mesh format block

    specified as

     version(ASCII double; currently 4.1)
       file-type(ASCII int; 0 for ASCII mode, 1 for binary mode)
       data-size(ASCII int; sizeof(size_t))
     < int with value one; only in binary mode, to detect endianness >

    though here the version is left as str
    """

    # http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format

    line = f.readline().decode()
    # Split the line
    # 4.1 0 8
    # into its components.
    str_list = list(filter(None, line.split()))
    fmt_version = str_list[0]
    if str_list[1] not in ["0", "1"]:
        raise ReadError()
    is_ascii = str_list[1] == "0"
    data_size = int(str_list[2])
    if not is_ascii:
        # The next line is the integer 1 in bytes. Useful for checking endianness.
        # Just assert that we get 1 here.
        one = f.read(struct.calcsize("i"))
        if struct.unpack("i", one)[0] != 1:
            raise ReadError()
    _fast_forward_to_end_block(f, "MeshFormat")
    return fmt_version, data_size, is_ascii


# Gmsh ASCII output uses `%.16g` for floating point values,
# meshio uses same precision but exponential notation `%.16e`.
def write(filename, mesh, fmt_version="4.1", binary=True, float_fmt=".16e"):
    """Writes a Gmsh msh file."""
    mesh = _convert_med_tags_to_gmsh(mesh)
    try:
        writer = _writers[fmt_version]
    except KeyError:
        try:
            writer = _writers[fmt_version]
        except KeyError:
            raise WriteError(
                "Need mesh format in {} (got {})".format(
                    sorted(_writers.keys()), fmt_version
                )
            )

    writer.write(filename, mesh, binary=binary, float_fmt=float_fmt)


register_format(
    "gmsh",
    [".msh"],
    read,
    {
        "gmsh22": lambda f, m, **kwargs: write(f, m, "2.2", **kwargs),
        "gmsh": lambda f, m, **kwargs: write(f, m, "4.1", **kwargs),
    },
)
def _convert_med_tags_to_gmsh(mesh):
    is_med_data = (
        "cell_tags" in mesh.cell_data
        or "point_tags" in mesh.point_data
        or any(k.startswith("med:") for k in mesh.field_data)
        or hasattr(mesh, "cell_tags")
        or hasattr(mesh, "point_tags")
    )
    if not is_med_data:
        return mesh

    mesh = copy.deepcopy(mesh)
    # Mapping from group name to physical tag, and from family id to physical tag.
    family_groups = getattr(mesh, "cell_tags", {})
    group_names = sorted({n for names in family_groups.values() for n in names})
    group_to_phys = {name: i for i, name in enumerate(group_names, start=1)}
    fam_to_phys = {} # Mapping from family id to physical tag, for backward compatibility with older meshio versions that used cell_tags for this purpose.
    for fam_id, names in family_groups.items():
        if names:
            fam_to_phys[int(fam_id)] = group_to_phys[names[0]]
   
    # Check if we have geometrical tags or cell_tags, and if we need to split cells based on them.
    has_geom = "gmsh:geometrical" in mesh.cell_data
    has_tags = "cell_tags" in mesh.cell_data

    if has_geom:
        split_data = mesh.cell_data["gmsh:geometrical"]
    elif has_tags:
        split_data = mesh.cell_data["cell_tags"]
    else:
        return _cleanup_med_fields(mesh, group_to_phys)

    needs_split = any(
        i < len(split_data)
        and split_data[i] is not None
        and len(np.unique(split_data[i])) > 1
        for i in range(len(mesh.cells))
    )
    if not needs_split:
        return _cleanup_med_fields(mesh, group_to_phys)

    new_cells = []
    new_geom = []
    new_phys = []
    entity_counter = 1

    for i, cb in enumerate(mesh.cells):
        if i >= len(split_data) or split_data[i] is None:
            n = len(cb.data)
            new_cells.append(cb)
            new_geom.append(np.full(n, entity_counter, dtype=int))
            new_phys.append(np.full(n, 0, dtype=int))
            entity_counter += 1
            continue

        tags = np.asarray(split_data[i], dtype=int)
        for utag in np.unique(tags):
            mask = tags == utag
            n = int(mask.sum())
            new_cells.append(CellBlock(cb.type, cb.data[mask]))

            if has_geom:
                new_geom.append(np.full(n, int(utag), dtype=int))
            else:
                new_geom.append(np.full(n, entity_counter, dtype=int))

            if has_geom and "gmsh:physical" in mesh.cell_data:
                orig = mesh.cell_data["gmsh:physical"]
                if i < len(orig) and orig[i] is not None:
                    new_phys.append(np.asarray(orig[i], dtype=int)[mask])
                else:
                    new_phys.append(np.full(n, 0, dtype=int))
            else:
                phys = fam_to_phys.get(int(utag), 0)
                new_phys.append(np.full(n, phys, dtype=int))

            entity_counter += 1

    point_data = {k: v for k, v in mesh.point_data.items() if k != "point_tags"}

    if "gmsh:dim_tags" not in point_data:
        n_pts = len(mesh.points)
        dim_tags = np.zeros((n_pts, 2), dtype=int)
        assigned = np.full(n_pts, False)
        for ci, cb in enumerate(new_cells):
            etag = int(new_geom[ci][0])
            nodes = np.unique(cb.data)
            new_nodes = nodes[~assigned[nodes]]
            dim_tags[new_nodes, 0] = cb.dim
            dim_tags[new_nodes, 1] = etag
            assigned[nodes] = True
        if not np.all(assigned):
            rem = np.where(~assigned)[0]
            dim_tags[rem, 0] = new_cells[0].dim if new_cells else 0
            dim_tags[rem, 1] = int(new_geom[0][0]) if new_geom else 1
        point_data["gmsh:dim_tags"] = dim_tags
    else:
        point_data["gmsh:dim_tags"] = np.asarray(
            point_data["gmsh:dim_tags"], dtype=int
        )

    dim_val = new_cells[0].dim if new_cells else 2
    field_data = {name: [tag, dim_val] for name, tag in group_to_phys.items()}

    return Mesh(
        mesh.points,
        new_cells,
        point_data=point_data,
        cell_data={"gmsh:geometrical": new_geom, "gmsh:physical": new_phys},
        field_data=field_data,
    )


def _cleanup_med_fields(mesh, group_to_phys):
    if "cell_tags" in mesh.cell_data:
        if "gmsh:physical" not in mesh.cell_data:
            mesh.cell_data["gmsh:physical"] = mesh.cell_data["cell_tags"]
        del mesh.cell_data["cell_tags"]
    if "point_tags" in mesh.point_data:
        del mesh.point_data["point_tags"]
    mesh.field_data = {
        k: v
        for k, v in mesh.field_data.items()
        if not k.startswith("med:")
        and isinstance(v, (list, tuple))
        and len(v) == 2
        and isinstance(v[0], (int, np.integer))
    }
    return mesh
