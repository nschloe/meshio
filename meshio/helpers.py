# -*- coding: utf-8 -*-
#
from .mesh import Mesh
from . import abaqus_io
from . import ansys_io
from . import dolfin_io
from . import exodus_io
from . import h5m_io
from . import mdpa_io
from . import med_io
from . import medit_io
from . import gmsh_io
from . import off_io
from . import permas_io
from . import stl_io
from . import svg_io
from . import vtk_io
from . import vtu_io
from . import xdmf_io

input_filetypes = [
    "abaqus",
    "ansys",
    "dolfin-xml",
    "exodus",
    "gmsh-ascii",
    "gmsh-binary",
    "mdpa",
    "med",
    "medit",
    "moab",
    "permas",
    "off",
    "stl-ascii",
    "stl-binary",
    "vtk-ascii",
    "vtk-binary",
    "vtu-ascii",
    "vtu-binary",
    "xdmf",
]

output_filetypes = [
    "abaqus",
    "ansys-ascii",
    "ansys-binary",
    "dolfin-xml",
    "exodus",
    "gmsh-ascii",
    "gmsh-binary",
    "mdpa",
    "med",
    "medit",
    "moab",
    "off",
    "permas",
    "stl-ascii",
    "stl-binary",
    "svg",
    "vtk-ascii",
    "vtk-binary",
    "vtu-ascii",
    "vtu-binary",
    "xdmf",
]

_extension_to_filetype = {
    ".e": "exodus",
    ".ex2": "exodus",
    ".exo": "exodus",
    ".med": "med",
    ".mesh": "medit",
    ".msh": "gmsh-binary",
    ".xml": "dolfin-xml",
    ".post": "permas",
    ".post.gz": "permas",
    ".dato": "permas",
    ".dato.gz": "permas",
    ".h5m": "moab",
    ".off": "off",
    ".stl": "stl-binary",
    ".vtu": "vtu-binary",
    ".vtk": "vtk-binary",
    ".xdmf": "xdmf",
    ".xmf": "xdmf",
    ".inp": "abaqus",
    ".mdpa": "mdpa",
    ".svg": "svg",
}


def _filetype_from_filename(filename):
    suffixes = [".{}".format(ext) for ext in filename.split(".")[1:]]
    ext = ""

    out = None
    for suffix in reversed(suffixes):
        ext = suffix + ext
        if ext in _extension_to_filetype:
            out = _extension_to_filetype[ext]

    assert out is not None, "Could not deduce file format from extension '{}'.".format(
        ext
    )

    return out


def read(filename, file_format=None):
    """Reads an unstructured mesh with added data.

    :param filenames: The files to read from.
    :type filenames: str

    :returns mesh{2,3}d: The mesh data.
    """
    # https://stackoverflow.com/q/4843173/353337
    assert isinstance(filename, str)

    if not file_format:
        # deduce file format from extension
        file_format = _filetype_from_filename(filename)

    format_to_reader = {
        "ansys": ansys_io,
        "ansys-ascii": ansys_io,
        "ansys-binary": ansys_io,
        #
        "gmsh": gmsh_io,
        "gmsh-ascii": gmsh_io,
        "gmsh-binary": gmsh_io,
        #
        "med": med_io,
        "medit": medit_io,
        "dolfin-xml": dolfin_io,
        "permas": permas_io,
        "moab": h5m_io,
        "off": off_io,
        #
        "stl": stl_io,
        "stl-ascii": stl_io,
        "stl-binary": stl_io,
        #
        "vtu-ascii": vtu_io,
        "vtu-binary": vtu_io,
        #
        "vtk-ascii": vtk_io,
        "vtk-binary": vtk_io,
        #
        "xdmf": xdmf_io,
        "exodus": exodus_io,
        #
        "abaqus": abaqus_io,
        #
        "mdpa": mdpa_io,
    }

    assert file_format in format_to_reader, "Unknown file format '{}' of '{}'.".format(
        file_format, filename
    )

    return format_to_reader[file_format].read(filename)


def write_points_cells(
    filename,
    points,
    cells,
    point_data=None,
    cell_data=None,
    field_data=None,
    file_format=None,
):
    mesh = Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )
    write(filename, mesh, file_format=file_format)
    return


def write(filename, mesh, file_format=None):
    """Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    """
    if not file_format:
        # deduce file format from extension
        file_format = _filetype_from_filename(filename)

    # check cells for sanity
    for key, value in mesh.cells.items():
        assert value.shape[1] == gmsh_io.num_nodes_per_cell[key]

    d = {
        "moab": lambda: h5m_io.write(filename, mesh),
        "ansys-ascii": lambda: ansys_io.write(filename, mesh, write_binary=False),
        "ansys-binary": lambda: ansys_io.write(filename, mesh, write_binary=True),
        "gmsh-ascii": lambda: gmsh_io.write(filename, mesh, write_binary=False),
        "gmsh-binary": lambda: gmsh_io.write(filename, mesh, write_binary=True),
        "med": lambda: med_io.write(filename, mesh),
        "medit": lambda: medit_io.write(filename, mesh),
        "dolfin-xml": lambda: dolfin_io.write(filename, mesh),
        "off": lambda: off_io.write(filename, mesh),
        "permas": lambda: permas_io.write(filename, mesh),
        "stl-ascii": lambda: stl_io.write(filename, mesh, write_binary=False),
        "stl-binary": lambda: stl_io.write(filename, mesh, write_binary=True),
        "vtu-ascii": lambda: vtu_io.write(filename, mesh, write_binary=False),
        "vtu": lambda: vtu_io.write(filename, mesh, write_binary=True),
        "vtu-binary": lambda: vtu_io.write(filename, mesh, write_binary=True),
        "vtk-ascii": lambda: vtk_io.write(filename, mesh, write_binary=False),
        "vtk": lambda: vtk_io.write(filename, mesh, write_binary=True),
        "vtk-binary": lambda: vtk_io.write(filename, mesh, write_binary=True),
        "xdmf": lambda: xdmf_io.write(filename, mesh),
        "xdmf3": lambda: xdmf_io.write(filename, mesh),
        "abaqus": lambda: abaqus_io.write(filename, mesh),
        "exodus": lambda: exodus_io.write(filename, mesh),
        "mdpa": lambda: mdpa_io.write(filename, mesh),
        "svg": lambda: svg_io.write(filename, mesh),
    }

    try:
        writer = d[file_format]
    except KeyError:
        raise KeyError(
            "Unknown format '{}'. Pick one of {}".format(
                file_format, sorted(list(d.keys()))
            )
        )

    # Actually perform the write
    writer()
    return
