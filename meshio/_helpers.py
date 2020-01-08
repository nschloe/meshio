import pathlib

import numpy

from ._common import num_nodes_per_cell
from ._exceptions import ReadError, WriteError
from ._files import is_buffer
from ._mesh import Mesh

_extension_to_filetype = {}
reader_map = {}
_writer_map = {}


def register(name, extensions, reader, writer_map):
    for ext in extensions:
        _extension_to_filetype[ext] = name

    if reader is not None:
        reader_map[name] = reader
    _writer_map.update(writer_map)


def _filetype_from_path(path):
    ext = ""
    out = None
    for suffix in reversed(path.suffixes):
        ext = suffix + ext
        if ext in _extension_to_filetype:
            out = _extension_to_filetype[ext]

    if out is None:
        raise ReadError(f"Could not deduce file format from extension '{ext}'.")
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
                "tetgen format is spread across multiple files "
                "and so cannot be read from a buffer"
            )
        msg = f"Unknown file format '{file_format}'"
    else:
        path = pathlib.Path(filename)
        if not path.exists():
            raise ReadError(f"File {filename} not found.")

        if not file_format:
            # deduce file format from extension
            file_format = _filetype_from_path(path)

        msg = f"Unknown file format '{file_format}' of '{filename}'."

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
    **kwargs,
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
