import argparse
import os
import pathlib

from .. import cgns, h5m, vtu, xdmf
from .._helpers import _filetype_from_path, read, reader_map
from ._helpers import _get_version_text


def decompress(argv=None):
    # Parse command line arguments.
    parser = _get_parser()
    args = parser.parse_args(argv)

    # read mesh data
    fmt = args.input_format or _filetype_from_path(pathlib.Path(args.infile))

    size = os.stat(args.infile).st_size
    print(f"File size before: {size / 1024 ** 2:.2f} MB")
    mesh = read(args.infile, file_format=args.input_format)

    # # Some converters (like VTK) require `points` to be contiguous.
    # mesh.points = np.ascontiguousarray(mesh.points)

    # write it out
    if fmt == "cgns":
        cgns.write(args.infile, mesh, compression=None)
    elif fmt == "h5m":
        h5m.write(args.infile, mesh, compression=None)
    elif fmt == "vtu":
        vtu.write(args.infile, mesh, binary=True, compression=None)
    elif fmt == "xdmf":
        xdmf.write(args.infile, mesh, data_format="HDF", compression=None)
    else:
        print(f"Don't know how to decompress {args.infile}.")
        exit(1)

    size = os.stat(args.infile).st_size
    print(f"File size after:  {size / 1024 ** 2:.2f} MB")


def _get_parser():
    parser = argparse.ArgumentParser(
        description=("Decompress mesh file."),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("infile", type=str, help="mesh file to decompress")

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=_get_version_text(),
        help="display version information",
    )
    return parser
