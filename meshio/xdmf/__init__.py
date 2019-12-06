"""
I/O for XDMF.
http://www.xdmf.org/index.php/XDMF_Model_and_Format
"""
from .main import read, write
from .time_series import TimeSeriesReader, TimeSeriesWriter

__all__ = ["read", "write", "TimeSeriesWriter", "TimeSeriesReader"]
