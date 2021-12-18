import os
import pathlib

from .. import ansys, flac3d, gmsh, mdpa, ply, stl, vtk, vtu, xdmf
from .._common import error
from .._helpers import _filetype_from_path, read, reader_map


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to convert")

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )


def ascii(args):
    # read mesh data
    fmt = args.input_format or _filetype_from_path(pathlib.Path(args.infile))

    size = os.stat(args.infile).st_size
    print(f"File size before: {size / 1024 ** 2:.2f} MB")
    mesh = read(args.infile, file_format=args.input_format)

    # # Some converters (like VTK) require `points` to be contiguous.
    # mesh.points = np.ascontiguousarray(mesh.points)

    # write it out
    if fmt == "ansys":
        ansys.write(args.infile, mesh, binary=False)
    elif fmt == "flac3d":
        flac3d.write(args.infile, mesh, binary=False)
    elif fmt == "gmsh":
        gmsh.write(args.infile, mesh, binary=False)
    elif fmt == "mdpa":
        mdpa.write(args.infile, mesh, binary=False)
    elif fmt == "ply":
        ply.write(args.infile, mesh, binary=False)
    elif fmt == "stl":
        stl.write(args.infile, mesh, binary=False)
    elif fmt == "vtk":
        vtk.write(args.infile, mesh, binary=False)
    elif fmt == "vtu":
        vtu.write(args.infile, mesh, binary=False)
    elif fmt == "xdmf":
        xdmf.write(args.infile, mesh, data_format="XML")
    else:
        error(f"Don't know how to convert {args.infile} to ASCII format.")
        return 1

    size = os.stat(args.infile).st_size
    print(f"File size after: {size / 1024 ** 2:.2f} MB")
    return 0
