"""
I/O for XDMF.
http://www.xdmf.org/index.php/XDMF_Model_and_Format
"""

from .main import read, register, write
from .time_series import XdmfTimeSeriesReader, XdmfTimeSeriesWriter

__all__ = ["read", "write", "register", "XdmfTimeSeriesWriter", "XdmfTimeSeriesReader"]
