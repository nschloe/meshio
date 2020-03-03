import sys

from ..__about__ import __version__


def _get_version_text():
    return "\n".join(
        [
            "meshio {} [Python {}.{}.{}]".format(
                __version__,
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ),
            "Copyright (c) 2015-2020 Nico Schlömer et al.",
        ]
    )
