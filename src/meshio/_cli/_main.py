import argparse
from sys import version_info

from ..__about__ import __version__
from . import _ascii, _binary, _compress, _convert, _decompress, _info


def main(argv=None):
    parent_parser = argparse.ArgumentParser(
        description="Mesh input/output tools.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parent_parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=_get_version_text(),
        help="display version information",
    )

    subparsers = parent_parser.add_subparsers(
        title="subcommands", dest="command", required=True
    )

    parser = subparsers.add_parser("convert", help="Convert mesh files", aliases=["c"])

    _convert.add_args(parser)
    parser.set_defaults(func=_convert.convert)

    parser = subparsers.add_parser("info", help="Print mesh info", aliases=["i"])
    _info.add_args(parser)
    parser.set_defaults(func=_info.info)

    parser = subparsers.add_parser("compress", help="Compress mesh file")
    _compress.add_args(parser)
    parser.set_defaults(func=_compress.compress)

    parser = subparsers.add_parser("decompress", help="Decompress mesh file")
    _decompress.add_args(parser)
    parser.set_defaults(func=_decompress.decompress)

    parser = subparsers.add_parser("ascii", help="Convert to ASCII", aliases=["a"])
    _ascii.add_args(parser)
    parser.set_defaults(func=_ascii.ascii)

    parser = subparsers.add_parser("binary", help="Convert to binary", aliases=["b"])
    _binary.add_args(parser)
    parser.set_defaults(func=_binary.binary)

    args = parent_parser.parse_args(argv)

    return args.func(args)


def _get_version_text():
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return "\n".join(
        [
            f"meshio {__version__} [Python {python_version}]",
            "Copyright (c) 2015-2021 Nico Schlömer et al.",
        ]
    )
