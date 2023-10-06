import pathlib

from .._exceptions import ReadError
from .._helpers import register_format
from . import _vtk_42, _vtk_51


def read(filename):
    filename = pathlib.Path(filename)
    with open(filename.as_posix(), "rb") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    # The first line specifies the version
    line = f.readline().decode().strip()
    if not line.startswith("# vtk DataFile Version"):
        raise ReadError("Illegal VTK header")

    version = line[23:]
    if version == "5.1":
        return _vtk_51.read(f)

    # this also works for older format versions
    return _vtk_42.read(f)


def write(filename, mesh, fmt_version: str = "5.1", **kwargs):
    if fmt_version == "4.2":
        return _vtk_42.write(filename, mesh, **kwargs)

    assert fmt_version == "5.1"
    _vtk_51.write(filename, mesh, **kwargs)


register_format(
    "vtk",
    [".vtk"],
    read,
    {
        "vtk42": _vtk_42.write,
        "vtk51": _vtk_42.write,
        "vtk": _vtk_51.write,
    },
)
