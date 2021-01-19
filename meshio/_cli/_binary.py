import argparse
import os
import pathlib

from .. import ansys, flac3d, gmsh, mdpa, ply, stl, vtk, vtu, xdmf
from .._helpers import _filetype_from_path, read, reader_map
from ._helpers import _get_version_text


def binary(argv=None):
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
    if fmt == "ansys":
        ansys.write(args.infile, mesh, binary=True)
    elif fmt == "flac3d":
        flac3d.write(args.infile, mesh, binary=True)
    elif fmt == "gmsh":
        gmsh.write(args.infile, mesh, binary=True)
    elif fmt == "mdpa":
        mdpa.write(args.infile, mesh, binary=True)
    elif fmt == "ply":
        ply.write(args.infile, mesh, binary=True)
    elif fmt == "stl":
        stl.write(args.infile, mesh, binary=True)
    elif fmt == "vtk":
        vtk.write(args.infile, mesh, binary=True)
    elif fmt == "vtu":
        vtu.write(args.infile, mesh, binary=True)
    elif fmt == "xdmf":
        xdmf.write(args.infile, mesh, data_format="HDF")
    else:
        print(f"Don't know how to convert {args.infile} to binary format.")
        exit(1)

    size = os.stat(args.infile).st_size
    print(f"File size after: {size / 1024 ** 2:.2f} MB")


def _get_parser():
    parser = argparse.ArgumentParser(
        description=("Covert mesh file to binary format."),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("infile", type=str, help="mesh file to convert")

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
