import numpy as np

from .._helpers import _writer_map, read, reader_map, write


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--output-format",
        "-o",
        type=str,
        choices=sorted(list(_writer_map.keys())),
        help="output file format",
        default=None,
    )
    parser.add_argument(
        "--ascii",
        "-a",
        action="store_true",
        help="write in ASCII format variant (where applicable, default: binary)",
    )
    parser.add_argument("outfile", type=str, help="mesh file to be written to")
    parser.add_argument(
        "--float-format",
        "-f",
        type=str,
        help="float format used in output ASCII files (default: .16e)",
    )
    parser.add_argument(
        "--sets-to-int-data",
        "-s",
        action="store_true",
        help="if possible, convert sets to integer data (useful if the output type does not support sets)",
    )
    parser.add_argument(
        "--int-data-to-sets",
        "-d",
        action="store_true",
        help="if possible, convert integer data to sets (useful if the output type does not support integer data)",
    )


def convert(args):
    # read mesh data
    mesh = read(args.infile, file_format=args.input_format)

    # Some converters (like VTK) require `points` to be contiguous.
    mesh.points = np.ascontiguousarray(mesh.points)

    if args.sets_to_int_data:
        mesh.sets_to_int_data()

    if args.int_data_to_sets:
        mesh.int_data_to_sets()

    # write it out
    kwargs = {"file_format": args.output_format}
    if args.float_format is not None:
        kwargs["float_fmt"] = args.float_format
    if args.ascii:
        kwargs["binary"] = False

    write(args.outfile, mesh, **kwargs)
