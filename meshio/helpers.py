# -*- coding: utf-8 -*-
#
from . import ansys_io
from . import dolfin_io
from . import h5m_io
from . import medit_io
from . import gmsh_io
from . import off_io
from . import permas_io
from . import vtk_io

input_filetypes = [
        'ansys',
        'exodus',
        'gmsh-ascii',
        'gmsh-binary',
        'dolfin-xml',
        'medit',
        'permas',
        'moab',
        'off',
        'vtk-ascii',
        'vtk-binary',
        'vtu-ascii',
        'vtu-binary',
        'xdmf2',
        'xdmf3',
        ]

output_filetypes = [
        'ansys-ascii',
        'ansys-binary',
        'exodus',
        'gmsh-ascii',
        'gmsh-binary',
        'dolfin-xml',
        'medit',
        'permas',
        'moab',
        'off',
        'vtk-ascii',
        'vtk-binary',
        'vtu-ascii',
        'vtu-binary',
        'xdmf2',
        'xdmf3',
        ]

_extension_to_filetype = {
    '.e': 'exodus',
    '.ex2': 'exodus',
    '.exo': 'exodus',
    '.mesh': 'medit',
    '.msh': 'gmsh-binary',
    '.xml': 'dolfin-xml',
    '.post': 'permas',
    '.post.gz': 'permas',
    '.dato': 'permas',
    '.dato.gz': 'permas',
    '.h5m': 'moab',
    '.off': 'off',
    '.vtu': 'vtu-binary',
    '.vtk': 'vtk-binary',
    '.xdmf': 'xdmf3',
    '.xmf': 'xdmf3',
    }


def read(filename, file_format=None):
    '''Reads an unstructured mesh with added data.

    :param filenames: The files to read from.
    :type filenames: str
    :returns mesh{2,3}d: The mesh data.
    :returns point_data: Point data read from file.
    :type point_data: dict
    :returns field_data: Field data read from file.
    :type field_data: dict
    '''
    import os

    # http://stackoverflow.com/questions/4843173/how-to-check-if-a-type-of-variable-is-string-in-python
    assert isinstance(filename, str)

    if not file_format:
        # deduct file format from extension
        _, extension = os.path.splitext(filename)
        file_format = _extension_to_filetype[extension]

    if file_format in ['ansys', 'ansys-ascii', 'ansys-binary']:
        out = ansys_io.read(filename)
    elif file_format in ['gmsh', 'gmsh-ascii', 'gmsh-binary']:
        out = gmsh_io.read(filename)
    elif file_format == 'medit':
        out = medit_io.read(filename)
    elif file_format == 'dolfin-xml':
        out = dolfin_io.read(filename)
    elif file_format == 'permas':
        out = permas_io.read(filename)
    elif file_format == 'moab':
        out = h5m_io.read(filename)
    elif file_format == 'off':
        out = off_io.read(filename)
    elif file_format in ['vtu-ascii', 'vtu-binary']:
        out = vtk_io.read('vtu', filename)
    elif file_format in ['vtk-ascii', 'vtk-binary']:
        out = vtk_io.read('vtk', filename)
    elif file_format in ['xdmf', 'xdmf2']:
        out = vtk_io.read('xdmf2', filename)
    elif file_format == 'xdmf3':
        out = vtk_io.read('xdmf3', filename)
    else:
        assert file_format == 'exodus'
        out = vtk_io.read('exodus', filename)

    return out


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None,
          file_format=None
          ):
    '''Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    '''
    import os

    if not file_format:
        # deduct file format from extension
        _, extension = os.path.splitext(filename)
        file_format = _extension_to_filetype[extension]

    # check cells for sanity
    for key in cells:
        assert cells[key].shape[1] == gmsh_io.num_nodes_per_cell[key]

    if file_format == 'moab':
        h5m_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_format in ['ansys-ascii', 'ansys-binary']:
        ansys_io.write(
            filename, points, cells,
            is_ascii=(file_format == 'ansys-ascii'),
            point_data=point_data,
            cell_data=cell_data
            )
    elif file_format in ['gmsh-ascii', 'gmsh-binary']:
        gmsh_io.write(
            filename, points, cells,
            is_ascii=(file_format == 'gmsh-ascii'),
            point_data=point_data,
            cell_data=cell_data
            )
    elif file_format == 'medit':
        medit_io.write(filename, points, cells)
    elif file_format == 'dolfin-xml':
        dolfin_io.write(filename, points, cells, cell_data=cell_data)
    elif file_format == 'off':
        off_io.write(filename, points, cells)
    elif file_format == 'permas':
        permas_io.write(filename, points, cells)
    elif file_format == 'vtu-ascii':
        vtk_io.write(
            'vtu-ascii', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_format in ['vtu', 'vtu-binary']:
        vtk_io.write(
            'vtu-binary', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_format == 'vtk-ascii':
        vtk_io.write(
            'vtk-ascii', filename,
            points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_format in ['vtk', 'vtk-binary']:
        vtk_io.write(
            'vtk-binary', filename,
            points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_format in ['xdmf', 'xdmf2']:  # XDMF
        vtk_io.write(
            'xdmf', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_format == 'xdmf3':  # XDMF
        vtk_io.write(
            'xdmf3', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    else:
        assert file_format == 'exodus', (
            'Unknown file format \'{}\' of \'{}\'.'
            .format(file_format, filename)
            )
        vtk_io.write(
            'exodus', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    return
