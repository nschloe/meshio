# -*- coding: utf-8 -*-
#
from __future__ import print_function

from .__about__ import __version__, __author__, __author_email__, __website__

from . import cli
from .helpers import read, write, write_points_cells
from .mesh import Mesh
from .xdmf_io import XdmfTimeSeriesReader, XdmfTimeSeriesWriter


__all__ = [
    "cli",
    "read",
    "write",
    "write_points_cells",
    "Mesh",
    "XdmfTimeSeriesReader",
    "XdmfTimeSeriesWriter",
    "__version__",
    "__author__",
    "__author_email__",
    "__website__",
]

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end="")
