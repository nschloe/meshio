import pathlib

import numpy

from ._filetypes import get_writer, get_reader
from ._common import num_nodes_per_cell
from ._exceptions import ReadError, WriteError
from ._files import is_buffer
from ._mesh import Mesh


def read(filename, file_format=None):
    """Reads an unstructured mesh with added data.

    :param filenames: The files/PathLikes to read from.
    :type filenames: str

    :returns mesh{2,3}d: The mesh data.
    """

    if is_buffer(filename, "r"):
        if file_format == "tetgen":
            raise ReadError(
                "tetgen format is spread across multiple files, and so cannot be read from a buffer"
            )
        reader = get_reader(file_format)
    else:
        path = pathlib.Path(filename)
        if not path.exists():
            raise ReadError("File {} not found.".format(filename))

        reader = get_reader(file_format or path)

    return reader(filename)


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
        if file_format == "tetgen":
            raise WriteError(
                "tetgen format is spread across multiple files, and so cannot be written to a buffer"
            )
        writer = get_writer(file_format)
    else:
        writer = get_writer(file_format or filename)

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
