import os
import pathlib

from .. import ansys, cgns, gmsh, h5m, mdpa, ply, stl, vtk, vtu, xdmf
from .._common import error
from .._helpers import _filetypes_from_path, read, reader_map


def add_args(parser):
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
        "--max",
        "-max",
        action="store_true",
        help="maximum compression",
        default=False,
    )


def compress(args):
    if args.input_format:
        fmts = [args.input_format]
    else:
        fmts = _filetypes_from_path(pathlib.Path(args.infile))
    # pick the first
    fmt = fmts[0]

    size = os.stat(args.infile).st_size
    print(f"File size before: {size / 1024 ** 2:.2f} MB")
    mesh = read(args.infile, file_format=args.input_format)

    # # Some converters (like VTK) require `points` to be contiguous.
    # mesh.points = np.ascontiguousarray(mesh.points)

    # write it out
    _write_compressed(args.infile, mesh, fmt, args.max)
    
    size = os.stat(args.infile).st_size
    print(f"File size after: {size / 1024 ** 2:.2f} MB")

    def _write_compressed(filename, mesh, fmt, max_compression):
        """Write mesh to file with compression based on format.
        
        Args:
            filename: Output file path
            mesh: Mesh object to write
            fmt: Format string (e.g. "vtu", "xdmf")
            max_compression: Whether to use maximum compression settings
        """
        if fmt == "ansys":
            ansys.write(filename, mesh, binary=True)
        elif fmt == "cgns":
            cgns.write(filename, mesh, compression="gzip", compression_opts=9 if max_compression else 4)
        elif fmt == "gmsh": 
            gmsh.write(filename, mesh, binary=True)
        elif fmt == "h5m":
            h5m.write(filename, mesh, compression="gzip", compression_opts=9 if max_compression else 4)
        elif fmt == "mdpa":
            mdpa.write(filename, mesh, binary=True)
        elif fmt == "ply":
            ply.write(filename, mesh, binary=True) 
        elif fmt == "stl":
            stl.write(filename, mesh, binary=True)
        elif fmt == "vtk":
            vtk.write(filename, mesh, binary=True)
        elif fmt == "vtu":
            vtu.write(filename, mesh, binary=True, compression="lzma" if max_compression else "zlib")
        elif fmt == "xdmf":
            xdmf.write(filename, mesh, data_format="HDF", compression="gzip",
                      compression_opts=9 if max_compression else 4)
        else:
            error(f"Don't know how to compress {filename}.")
            exit(1)