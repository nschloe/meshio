import os
import pathlib

from .. import ansys, flac3d, gmsh, mdpa, ply, stl, vtk, vtu, xdmf
from .._helpers import _filetypes_from_path, read, reader_map


def add_args(parser):

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )

    parser.add_argument("infile", type=str, nargs='*', help="mesh file to convert")


    
def binary(args):
    if not isinstance(args.infile, list):
        args.infile = [args.infile]

    for file in args.infile:
        if args.input_format:
            fmts = [args.input_format]
        else:
            fmts = _filetypes_from_path(pathlib.Path(file))
        # pick the first
        fmt = fmts[0]

        size = os.stat(file).st_size
        print(f"File size before: {size / 1024 ** 2:.2f} MB")
        mesh = read(file, file_format=args.input_format)

        # # Some converters (like VTK) require `points` to be contiguous.
        # mesh.points = np.ascontiguousarray(mesh.points)

        # write it out
        if fmt == "ansys":
            ansys.write(file, mesh, binary=True)
        elif fmt == "flac3d":
            flac3d.write(file, mesh, binary=True)
        elif fmt == "gmsh":
            gmsh.write(file, mesh, binary=True)
        elif fmt == "mdpa":
            mdpa.write(file, mesh, binary=True)
        elif fmt == "ply":
            ply.write(file, mesh, binary=True)
        elif fmt == "stl":
            stl.write(file, mesh, binary=True)
        elif fmt == "vtk":
            vtk.write(file, mesh, binary=True)
        elif fmt == "vtu":
            vtu.write(file, mesh, binary=True)
        elif fmt == "xdmf":
            xdmf.write(file, mesh, data_format="HDF")
        else:
            print(f"Don't know how to convert {file} to binary format.")
            exit(1)

        size = os.stat(file).st_size
        print(f"File size after: {size / 1024 ** 2:.2f} MB")
