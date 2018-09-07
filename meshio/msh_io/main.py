# -*- coding: utf-8 -*-
#
from ctypes import c_int, sizeof
import struct
from . import msh2, msh4


def read(filename):
    """Reads a Gmsh msh file.
    """
    with open(filename, "rb") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    # The various versions of the format are specified at
    # <http://gmsh.info//doc/texinfo/gmsh.html#File-formats>.

    line = f.readline().decode("utf-8")
    assert line.strip() == "$MeshFormat"
    int_size = sizeof(c_int)
    format_version, data_size, is_ascii = _read_header(f, int_size)

    assert format_version in ["2", "4"], "Need mesh format 2 or 4 (got {})".format(
        format_version
    )

    reader = {"2": msh2, "4": msh4}[format_version]

    return reader.read_buffer(f, is_ascii, int_size, data_size)


def _read_header(f, int_size):
    line = f.readline().decode("utf-8")
    # Split the line
    # 2.2 0 8
    # into its components.
    str_list = list(filter(None, line.split()))
    format_version = str_list[0][0]
    assert str_list[1] in ["0", "1"]
    is_ascii = str_list[1] == "0"
    data_size = int(str_list[2])
    if not is_ascii:
        # The next line is the integer 1 in bytes. Useful for checking
        # endianness. Just assert that we get 1 here.
        one = f.read(int_size)
        assert struct.unpack("i", one)[0] == 1
        line = f.readline().decode("utf-8")
        assert line == "\n"
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndMeshFormat"
    return format_version, data_size, is_ascii


def write(filename, mesh, fmt_version, write_binary=True):
    if fmt_version == "2":
        msh2.write(filename, mesh, write_binary=write_binary)
    else:
        assert fmt_version == "4"
        msh4.write(filename, mesh, write_binary=write_binary)
    return
