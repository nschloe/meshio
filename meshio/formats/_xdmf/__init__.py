"""
I/O for XDMF.
http://www.xdmf.org/index.php/XDMF_Model_and_Format
"""

from .main import read, write, register
from .time_series import XdmfTimeSeriesReader, XdmfTimeSeriesWriter

__all__ = ["read", "write", "register", "XdmfTimeSeriesWriter", "XdmfTimeSeriesReader"]
