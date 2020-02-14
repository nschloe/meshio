import sys

from ..__about__ import __copyright__, __version__


def _get_version_text():
    return "\n".join(
        [
            "meshio {} [Python {}.{}.{}]".format(
                __version__,
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ),
            __copyright__,
        ]
    )
