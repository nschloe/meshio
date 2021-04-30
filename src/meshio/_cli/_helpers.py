from sys import version_info

from ..__about__ import __version__


def _get_version_text():
    return "\n".join(
        [
            f"meshio {__version__} "
            f"[Python {version_info.major}.{version_info.minor}.{version_info.micro}]",
            "Copyright (c) 2015-2021 Nico Schlömer et al.",
        ]
    )
