import pathlib
import struct

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
