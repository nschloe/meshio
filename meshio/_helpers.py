import numpy

from . import (
    _abaqus,
    _ansys,
    _dolfin,
    _exodus,
    _flac3d,
    _gmsh,
    _h5m,
    _mdpa,
    _med,
    _medit,
    _nastran,
    _obj,
    _off,
    _permas,
    _ply,
    _stl,
    _svg,
    _tetgen,
    _vtk,
    _vtu,
    _xdmf,
)
from ._common import num_nodes_per_cell
from ._mesh import Mesh

input_filetypes = [
    "abaqus",
    "ansys",
    "dolfin-xml",
    "exodus",
    "flac3d",
    "gmsh-ascii",
    "gmsh-binary",
    "mdpa",
    "med",
    "medit",
    "moab",
    "nastran",
    "permas",
    "ply-ascii",
    "ply-binary",
    "obj",
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
    "flac3d",
    "gmsh2-ascii",
    "gmsh2-binary",
    "gmsh4-ascii",
    "gmsh4-binary",
    "mdpa",
    "med",
    "medit",
    "moab",
    "nastran",
    "obj",
    "off",
    "permas",
    "ply-ascii",
    "ply-binary",
    "stl-ascii",
    "stl-binary",
    "svg",
    "tetgen",
    "vtk-ascii",
    "vtk-binary",
    "vtu-ascii",
    "vtu-binary",
    "xdmf",
    "xdmf-binary",
    "xdmf-hdf",
    "xdmf-xml",
]

_extension_to_filetype = {
    ".bdf": "nastran",
    ".e": "exodus",
    ".ex2": "exodus",
    ".exo": "exodus",
    ".f3grid": "flac3d",
    ".fem": "nastran",
    ".med": "med",
    ".mesh": "medit",
    ".msh": "gmsh4-binary",
    ".nas": "nastran",
    ".xml": "dolfin-xml",
    ".post": "permas",
    ".post.gz": "permas",
    ".dato": "permas",
    ".dato.gz": "permas",
    ".h5m": "moab",
    ".obj": "obj",
    ".off": "off",
    ".ply": "ply-binary",
    ".stl": "stl-binary",
    ".vtu": "vtu-binary",
    ".vtk": "vtk-binary",
    ".xdmf": "xdmf",
    ".xmf": "xdmf",
    ".inp": "abaqus",
    ".mdpa": "mdpa",
    ".svg": "svg",
    ".node": "tetgen",
    ".ele": "tetgen",
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
        "ansys": _ansys,
        "ansys-ascii": _ansys,
        "ansys-binary": _ansys,
        #
        "gmsh": _gmsh,
        "gmsh-ascii": _gmsh,
        "gmsh-binary": _gmsh,
        "gmsh2": _gmsh,
        "gmsh2-ascii": _gmsh,
        "gmsh2-binary": _gmsh,
        "gmsh4": _gmsh,
        "gmsh4-ascii": _gmsh,
        "gmsh4-binary": _gmsh,
        #
        "flac3d": _flac3d,
        "med": _med,
        "medit": _medit,
        "nastran": _nastran,
        "dolfin-xml": _dolfin,
        "permas": _permas,
        "moab": _h5m,
        "obj": _obj,
        "off": _off,
        #
        "ply": _ply,
        "ply-ascii": _ply,
        "ply-binary": _ply,
        #
        "stl": _stl,
        "stl-ascii": _stl,
        "stl-binary": _stl,
        #
        "tetgen": _tetgen,
        #
        "vtu-ascii": _vtu,
        "vtu-binary": _vtu,
        #
        "vtk-ascii": _vtk,
        "vtk-binary": _vtk,
        #
        "xdmf": _xdmf,
        "exodus": _exodus,
        #
        "abaqus": _abaqus,
        #
        "mdpa": _mdpa,
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
    **kwargs
):
    points = numpy.asarray(points)
    cells = {key: numpy.asarray(value) for key, value in cells.items()}
    mesh = Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )
    return write(filename, mesh, file_format=file_format, **kwargs)


def write(filename, mesh, file_format=None, **kwargs):
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
        if key[:7] == "polygon":
            assert value.shape[1] == int(key[7:])
        elif key in num_nodes_per_cell:
            assert value.shape[1] == num_nodes_per_cell[key]
        else:
            # we allow custom keys <https://github.com/nschloe/meshio/issues/501> and
            # cannot check those
            pass

    try:
        interface, args, default_kwargs = _writer_map[file_format]
    except KeyError:
        raise KeyError(
            "Unknown format '{}'. Pick one of {}".format(
                file_format, sorted(list(_writer_map.keys()))
            )
        )

    # Build kwargs
    _kwargs = default_kwargs.copy()
    _kwargs.update(kwargs)

    # Write
    return interface.write(filename, mesh, *args, **_kwargs)


_writer_map = {
    "moab": (_h5m, (), {}),
    "ansys-ascii": (_ansys, (), {"binary": False}),
    "ansys-binary": (_ansys, (), {"binary": True}),
    "gmsh2-ascii": (_gmsh, ("2",), {"binary": False}),
    "gmsh2-binary": (_gmsh, ("2",), {"binary": True}),
    "gmsh4-ascii": (_gmsh, ("4",), {"binary": False}),
    "gmsh4-binary": (_gmsh, ("4",), {"binary": True}),
    "med": (_med, (), {}),
    "medit": (_medit, (), {}),
    "dolfin-xml": (_dolfin, (), {}),
    "obj": (_obj, (), {}),
    "off": (_off, (), {}),
    "permas": (_permas, (), {}),
    "ply-ascii": (_ply, (), {"binary": False}),
    "ply-binary": (_ply, (), {"binary": True}),
    "stl-ascii": (_stl, (), {"binary": False}),
    "stl-binary": (_stl, (), {"binary": True}),
    "tetgen": (_tetgen, (), {}),
    "vtu-ascii": (_vtu, (), {"binary": False}),
    "vtu": (_vtu, (), {"binary": True}),
    "vtu-binary": (_vtu, (), {"binary": True}),
    "vtk-ascii": (_vtk, (), {"binary": False}),
    "vtk": (_vtk, (), {"binary": True}),
    "vtk-binary": (_vtk, (), {"binary": True}),
    "xdmf": (_xdmf, (), {}),
    "xdmf3": (_xdmf, (), {}),
    "xdmf-binary": (_xdmf, (), {"data_format": "Binary"}),
    "xdmf3-binary": (_xdmf, (), {"data_format": "Binary"}),
    "xdmf-hdf": (_xdmf, (), {"data_format": "HDF"}),
    "xdmf3-hdf": (_xdmf, (), {"data_format": "HDF"}),
    "xdmf-xml": (_xdmf, (), {"data_format": "XML"}),
    "xdmf3-xml": (_xdmf, (), {"data_format": "XML"}),
    "abaqus": (_abaqus, (), {}),
    "exodus": (_exodus, (), {}),
    "mdpa": (_mdpa, (), {}),
    "svg": (_svg, (), {}),
    "nastran": (_nastran, (), {}),
    "flac3d": (_flac3d, (), {}),
}
