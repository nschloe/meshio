from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import ArrayLike

from ._common import num_nodes_per_cell
from ._exceptions import ReadError, WriteError
from ._files import is_buffer
from ._mesh import CellBlock, Mesh

extension_to_filetype = {}
reader_map = {}
_writer_map = {}


def register_format(name: str, extensions: list[str], reader, writer_map):
    for ext in extensions:
        extension_to_filetype[ext] = name

    if reader is not None:
        reader_map[name] = reader
    _writer_map.update(writer_map)


def _filetype_from_path(path: pathlib.Path):
    ext = ""
    out = None
    for suffix in reversed(path.suffixes):
        ext = (suffix + ext).lower()
        if ext in extension_to_filetype:
            out = extension_to_filetype[ext]

    if out is None:
        raise ReadError(f"Could not deduce file format from extension '{ext}'.")
    return out


def read(filename, file_format: str | None = None):
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
    points: ArrayLike,
    cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
    point_data: dict[str, ArrayLike] | None = None,
    cell_data: dict[str, list[ArrayLike]] | None = None,
    field_data=None,
    point_sets: dict[str, ArrayLike] | None = None,
    cell_sets: dict[str, list[ArrayLike]] | None = None,
    file_format: str | None = None,
    **kwargs,
):
    points = np.asarray(points)
    mesh = Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )
    mesh.write(filename, file_format=file_format, **kwargs)


def write(filename, mesh: Mesh, file_format: str | None = None, **kwargs):
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
        formats = sorted(list(_writer_map.keys()))
        raise WriteError(f"Unknown format '{file_format}'. Pick one of {formats}")

    # check cells for sanity
    for cell_block in mesh.cells:
        key = cell_block.type
        value = cell_block.data
        if key in num_nodes_per_cell:
            if value.shape[1] != num_nodes_per_cell[key]:
                raise WriteError(
                    f"Unexpected cells array shape {value.shape} for {key} cells."
                )
        else:
            # we allow custom keys <https://github.com/nschloe/meshio/issues/501> and
            # cannot check those
            pass

    # Write
    return writer(filename, mesh, **kwargs)
