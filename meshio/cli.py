# -*- coding: utf-8 -*-
#
"""
Convert a mesh file to another.
"""
from __future__ import print_function

import sys

import numpy

from .__about__ import __version__
from .helpers import read, write, input_filetypes, output_filetypes


def main(argv=None):
    # Parse command line arguments.
    parser = _get_parser()
    args = parser.parse_args(argv)

    # read mesh data
    mesh = read(args.infile, file_format=args.input_format)
    print(mesh)

    if args.prune:
        mesh.prune()

    if (
        args.prune_z_0
        and mesh.points.shape[1] == 3
        and numpy.all(numpy.abs(mesh.points[:, 2]) < 1.0e-13)
    ):
        mesh.points = mesh.points[:, :2]

    # Some converters (like VTK) require `points` to be contiguous.
    mesh.points = numpy.ascontiguousarray(mesh.points)

    # write it out
    write(args.outfile, mesh, file_format=args.output_format)
    return


def _get_parser():
    """Parse input options."""
    import argparse

    parser = argparse.ArgumentParser(description=("Convert between mesh formats."))

    parser.add_argument("infile", type=str, help="mesh file to be read from")

    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=input_filetypes,
        help="input file format",
        default=None,
    )

    parser.add_argument(
        "--output-format",
        "-o",
        type=str,
        choices=output_filetypes,
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
        "--prune-z-0",
        "-z",
        action="store_true",
        help="remove third (z) dimension if all points are 0",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s {}, Python {}".format(__version__, sys.version),
        help="display version information",
    )

    return parser
