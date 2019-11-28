from . import _cli
from .__about__ import __author__, __author_email__, __version__, __website__
from ._exceptions import ReadError, WriteError
from ._helpers import read, write, write_points_cells
from ._filetypes import register_reader, register_writer
from ._mesh import Mesh
from .formats import *
from .formats._xdmf import XdmfTimeSeriesReader, XdmfTimeSeriesWriter

__all__ = [
    "_cli",
    "read",
    "write",
    "write_points_cells",
    "Mesh",
    "ReadError",
    "register_reader",
    "register_writer",
    "WriteError",
    "XdmfTimeSeriesReader",
    "XdmfTimeSeriesWriter",
    "__version__",
    "__author__",
    "__author_email__",
    "__website__",
]
