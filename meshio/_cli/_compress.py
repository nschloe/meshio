import argparse
import os
import pathlib

from .. import ansys, cgns, gmsh, h5m, mdpa, med, ply, stl, vtk, vtu, xdmf
from .._helpers import _filetype_from_path, read, reader_map
from ._helpers import _get_version_text


def compress(argv=None):
    # Parse command line arguments.
    parser = _get_parser()
    args = parser.parse_args(argv)

    # read mesh data
    fmt = args.input_format or _filetype_from_path(pathlib.Path(args.infile))

    size = os.stat(args.infile).st_size
    print("File size before: {:.2f} MB".format(size / 1024 ** 2))
    mesh = read(args.infile, file_format=args.input_format)

    # # Some converters (like VTK) require `points` to be contiguous.
    # mesh.points = numpy.ascontiguousarray(mesh.points)

    # write it out
    if fmt == "ansys":
        ansys.write(args.infile, mesh, binary=True)
    elif fmt == "cgns":
        cgns.write(
            args.infile, mesh, compression="gzip", compression_opts=9 if args.max else 4
        )
    elif fmt == "gmsh":
        gmsh.write(args.infile, mesh, binary=True)
    elif fmt == "h5m":
        h5m.write(
            args.infile, mesh, compression="gzip", compression_opts=9 if args.max else 4
        )
    elif fmt == "mdpa":
        mdpa.write(args.infile, mesh, binary=True)
    elif fmt == "med":
        med.write(
            args.infile, mesh, compression="gzip", compression_opts=9 if args.max else 4
        )
    elif fmt == "ply":
        ply.write(args.infile, mesh, binary=True)
    elif fmt == "stl":
        stl.write(args.infile, mesh, binary=True)
    elif fmt == "vtk":
        vtk.write(args.infile, mesh, binary=True)
    elif fmt == "vtu":
        vtu.write(
            args.infile, mesh, binary=True, compression="lzma" if args.max else "zlib"
        )
    elif fmt == "xdmf":
        xdmf.write(
            args.infile,
            mesh,
            data_format="HDF",
            compression="gzip",
            compression_opts=9 if args.max else 4,
        )
    else:
        print("Don't know how to compress {}.".format(args.infile))
        exit(1)

    size = os.stat(args.infile).st_size
    print("File size after:  {:.2f} MB".format(size / 1024 ** 2))


def _get_parser():
    parser = argparse.ArgumentParser(
        description=("Compress mesh file."),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("infile", type=str, help="mesh file to compress")

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )

    parser.add_argument(
        "--max", "-max", action="store_true", help="maximum compression", default=False,
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=_get_version_text(),
        help="display version information",
    )
    return parser
