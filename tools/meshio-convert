#! /usr/bin/env python3
"""
Convert a mesh file to another.
"""
from __future__ import print_function

import numpy

import meshio


def _main():
    # Parse command line arguments.
    args = _parse_options()

    # read mesh data
    mesh = meshio.read(args.infile, file_format=args.input_format)
    print(mesh)

    if args.prune:
        mesh.prune()

    # Some converters (like VTK) require `points` to be contiguous.
    mesh.points = numpy.ascontiguousarray(mesh.points)

    # write it out
    meshio.write(args.outfile, mesh, file_format=args.output_format)
    return


def _parse_options():
    """Parse input options."""
    import argparse

    parser = argparse.ArgumentParser(description=("Convert between mesh formats."))

    parser.add_argument("infile", type=str, help="mesh file to be read from")

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=meshio.helpers.input_filetypes,
        help="input file format",
        default=None,
    )

    parser.add_argument(
        "--output-format",
        "-o",
        type=str,
        choices=meshio.helpers.output_filetypes,
        help="output file format",
        default=None,
    )

    parser.add_argument("outfile", type=str, help="mesh file to be written to")

    parser.add_argument(
        "--prune",
        "-p",
        action="store_true",
        help="remove lower order cells, remove orphaned nodes",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s " + ("(version %s)" % meshio.__version__),
    )

    return parser.parse_args()


if __name__ == "__main__":
    _main()
