import sys
from contextlib import contextmanager

try:
    # Python 3.6+
    from os import PathLike
except ImportError:
    from pathlib import PurePath as PathLike


def is_buffer(obj, mode):
    return ("r" in mode and hasattr(obj, "read")) or (
        "w" in mode and hasattr(obj, "write")
    )


@contextmanager
def open_file(path_or_buf, mode="r"):
    if is_buffer(path_or_buf, mode):
        yield path_or_buf
    elif sys.version_info < (3, 6) and isinstance(path_or_buf, PathLike):
        # TODO remove when python 3.5 is EoL (i.e. 2020-09-13)
        # https://devguide.python.org/#status-of-python-branches
        # https://www.python.org/dev/peps/pep-0478/
        with open(str(path_or_buf), mode) as f:
            yield f
    else:
        with open(path_or_buf, mode) as f:
            yield f
