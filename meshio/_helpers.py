import pathlib

import numpy

from . import (
    ansys,
    cgns,
    dolfin,
    exodus,
    flac3d,
    gmsh,
    h5m,
    mdpa,
    med,
    medit,
    nastran,
    neuroglancer,
    obj,
    off,
    permas,
    ply,
    stl,
    svg,
    tetgen,
    vtu,
    wkt,
    xdmf,
)
from ._common import num_nodes_per_cell
from ._exceptions import ReadError, WriteError
from ._files import is_buffer
from ._mesh import Mesh


def register(name, extensions, reader, writer_map):
    for ext in extensions:
        _extension_to_filetype[ext] = name

    reader_map[name] = reader
    _writer_map.update(writer_map)


_extension_to_filetype = {
    ".bdf": "nastran",
    ".cgns": "cgns",
    ".e": "exodus",
    ".ex2": "exodus",
    ".exo": "exodus",
    ".f3grid": "flac3d",
    ".fem": "nastran",
    ".med": "med",
    ".mesh": "medit",
    ".msh": "gmsh",
    ".nas": "nastran",
    ".xml": "dolfin-xml",
    ".post": "permas",
    ".post.gz": "permas",
    ".dato": "permas",
    ".dato.gz": "permas",
    ".h5m": "moab",
    ".obj": "obj",
    ".off": "off",
    ".ply": "ply",
    ".stl": "stl",
    ".vtu": "vtu",
    ".wkt": "wkt",
    ".xdmf": "xdmf",
    ".xmf": "xdmf",
    ".mdpa": "mdpa",
    ".svg": "svg",
    ".node": "tetgen",
    ".ele": "tetgen",
}

reader_map = {
    "ansys": ansys.read,
    "cgns": cgns.read,
    "dolfin-xml": dolfin.read,
    "exodus": exodus.read,
    "flac3d": flac3d.read,
    "gmsh": gmsh.read,
    "mdpa": mdpa.read,
    "med": med.read,
    "medit": medit.read,
    "moab": h5m.read,
    "nastran": nastran.read,
    "neuroglancer": neuroglancer.read,
    "obj": obj.read,
    "off": off.read,
    "permas": permas.read,
    "ply": ply.read,
    "stl": stl.read,
    "tetgen": tetgen.read,
    "vtu": vtu.read,
    "wkt": wkt.read,
    "xdmf": xdmf.read,
}

_writer_map = {
    "ansys-ascii": lambda f, m, **kwargs: ansys.write(f, m, **kwargs, binary=False),
    "ansys-binary": lambda f, m, **kwargs: ansys.write(f, m, **kwargs, binary=True),
    "cgns": cgns.write,
    "dolfin-xml": dolfin.write,
    "exodus": exodus.write,
    "flac3d": flac3d.write,
    "gmsh": lambda f, m, **kwargs: gmsh.write(f, m, "4", **kwargs, binary=True),
    "gmsh2-ascii": lambda f, m, **kwargs: gmsh.write(f, m, "2", **kwargs, binary=False),
    "gmsh2-binary": lambda f, m, **kwargs: gmsh.write(f, m, "2", **kwargs, binary=True),
    "gmsh4-ascii": lambda f, m, **kwargs: gmsh.write(f, m, "4", **kwargs, binary=False),
    "gmsh4-binary": lambda f, m, **kwargs: gmsh.write(f, m, "4", **kwargs, binary=True),
    "mdpa": mdpa.write,
    "med": med.write,
    "medit": medit.write,
    "moab": h5m.write,
    "nastran": nastran.write,
    "neuroglancer": neuroglancer.write,
    "obj": obj.write,
    "off": off.write,
    "permas": permas.write,
    "ply-ascii": lambda f, m, **kwargs: ply.write(f, m, **kwargs, binary=False),
    "ply-binary": lambda f, m, **kwargs: ply.write(f, m, **kwargs, binary=True),
    "ply": lambda f, m, **kwargs: ply.write(f, m, **kwargs, binary=True),
    "stl-ascii": lambda f, m, **kwargs: stl.write(f, m, **kwargs, binary=False),
    "stl-binary": lambda f, m, **kwargs: stl.write(f, m, **kwargs, binary=True),
    "stl": lambda f, m, **kwargs: stl.write(f, m, **kwargs, binary=True),
    "svg": svg.write,
    "tetgen": tetgen.write,
    "vtu": lambda f, m, **kwargs: vtu.write(f, m, **kwargs, binary=True),
    "vtu-ascii": lambda f, m, **kwargs: vtu.write(f, m, **kwargs, binary=False),
    "vtu-binary": lambda f, m, **kwargs: vtu.write(f, m, **kwargs, binary=True),
    "wkt": wkt.write,
    "xdmf": xdmf.write,
    "xdmf-binary": lambda f, m, **kwargs: xdmf.write(f, m, data_format="Binary"),
    "xdmf-hdf": lambda f, m, **kwargs: xdmf.write(f, m, data_format="HDF"),
    "xdmf-xml": lambda f, m, **kwargs: xdmf.write(f, m, data_format="XML"),
    "xdmf3": xdmf.write,
    "xdmf3-binary": lambda f, m, **kwargs: xdmf.write(f, m, data_format="Binary"),
    "xdmf3-hdf": lambda f, m, **kwargs: xdmf.write(f, m, data_format="HDF"),
    "xdmf3-xml": lambda f, m, **kwargs: xdmf.write(f, m, data_format="XML"),
}


def _filetype_from_path(path):
    ext = ""
    out = None
    for suffix in reversed(path.suffixes):
        ext = suffix + ext
        if ext in _extension_to_filetype:
            out = _extension_to_filetype[ext]

    if out is None:
        raise ReadError("Could not deduce file format from extension '{}'.".format(ext))
    return out


def read(filename, file_format=None):
    """Reads an unstructured mesh with added data.

    :param filenames: The files/PathLikes to read from.
    :type filenames: str

    :returns mesh{2,3}d: The mesh data.
    """
    if is_buffer(filename, "r"):
        if file_format is None:
            raise ReadError("File format must be given if buffer is used")
        if file_format == "tetgen":
            raise ReadError(
                "tetgen format is spread across multiple files, and so cannot be read from a buffer"
            )
        msg = "Unknown file format '{}'".format(file_format)
    else:
        path = pathlib.Path(filename)
        if not path.exists():
            raise ReadError("File {} not found.".format(filename))

        if not file_format:
            # deduce file format from extension
            file_format = _filetype_from_path(path)

        msg = "Unknown file format '{}' of '{}'.".format(file_format, filename)

    if file_format not in reader_map:
        raise ReadError(msg)

    return reader_map[file_format](filename)


def write_points_cells(
    filename,
    points,
    cells,
    point_data=None,
    cell_data=None,
    field_data=None,
    file_format=None,
    **kwargs
):
    points = numpy.asarray(points)
    cells = {key: numpy.asarray(value) for key, value in cells.items()}
    mesh = Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )
    return write(filename, mesh, file_format=file_format, **kwargs)


def write(filename, mesh, file_format=None, **kwargs):
    """Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    """
    if is_buffer(filename, "r"):
        if file_format is None:
            raise WriteError("File format must be supplied if `filename` is a buffer")
        if file_format == "tetgen":
            raise WriteError(
                "tetgen format is spread across multiple files, and so cannot be written to a buffer"
            )
    else:
        path = pathlib.Path(filename)
        if not file_format:
            # deduce file format from extension
            file_format = _filetype_from_path(path)

    try:
        writer = _writer_map[file_format]
    except KeyError:
        raise KeyError(
            "Unknown format '{}'. Pick one of {}".format(
                file_format, sorted(list(_writer_map.keys()))
            )
        )

    # check cells for sanity
    for key, value in mesh.cells.items():
        if key[:7] == "polygon":
            if value.shape[1] != int(key[7:]):
                raise WriteError()
        elif key in num_nodes_per_cell:
            if value.shape[1] != num_nodes_per_cell[key]:
                raise WriteError()
        else:
            # we allow custom keys <https://github.com/nschloe/meshio/issues/501> and
            # cannot check those
            pass

    # Write
    return writer(filename, mesh, **kwargs)
