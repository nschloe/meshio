# -*- coding: utf-8 -*-
#
import os

from . import ansys_io
from . import dolfin_io
from . import exodus_io
from . import h5m_io
from . import med_io
from . import medit_io
from . import gmsh_io
from . import off_io
from . import permas_io
from . import stl_io
from . import vtk_io
from . import vtu_io
from . import xdmf_io

input_filetypes = [
    'ansys',
    'exodus',
    'gmsh-ascii',
    'gmsh-binary',
    'dolfin-xml',
    'med',
    'medit',
    'permas',
    'moab',
    'off',
    'stl-ascii',
    'stl-binary',
    'vtk-ascii',
    'vtk-binary',
    'vtu-ascii',
    'vtu-binary',
    'xdmf',
    ]

output_filetypes = [
    'ansys-ascii',
    'ansys-binary',
    'exodus',
    'gmsh-ascii',
    'gmsh-binary',
    'dolfin-xml',
    'med',
    'medit',
    'permas',
    'moab',
    'off',
    'stl-ascii',
    'stl-binary',
    'vtk-ascii',
    'vtk-binary',
    'vtu-ascii',
    'vtu-binary',
    'xdmf',
    ]

_extension_to_filetype = {
    '.e': 'exodus',
    '.ex2': 'exodus',
    '.exo': 'exodus',
    '.med': 'med',
    '.mesh': 'medit',
    '.msh': 'gmsh-binary',
    '.xml': 'dolfin-xml',
    '.post': 'permas',
    '.post.gz': 'permas',
    '.dato': 'permas',
    '.dato.gz': 'permas',
    '.h5m': 'moab',
    '.off': 'off',
    '.stl': 'stl-binary',
    '.vtu': 'vtu-binary',
    '.vtk': 'vtk-binary',
    '.xdmf': 'xdmf',
    '.xmf': 'xdmf',
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
    # https://stackoverflow.com/q/4843173/353337
    assert isinstance(filename, str)

    if not file_format:
        # deduce file format from extension
        extension = '.' + os.path.basename(filename).split(os.extsep, 1)[-1]
        file_format = _extension_to_filetype[extension]

    format_to_reader = {
        'ansys': ansys_io,
        'ansys-ascii': ansys_io,
        'ansys-binary': ansys_io,
        #
        'gmsh': gmsh_io,
        'gmsh-ascii': gmsh_io,
        'gmsh-binary': gmsh_io,
        #
        'med': med_io,
        'medit': medit_io,
        'dolfin-xml': dolfin_io,
        'permas': permas_io,
        'moab': h5m_io,
        'off': off_io,
        #
        'stl': stl_io,
        'stl-ascii': stl_io,
        'stl-binary': stl_io,
        #
        'vtu-ascii': vtu_io,
        'vtu-binary': vtu_io,
        #
        'vtk-ascii': vtk_io,
        'vtk-binary': vtk_io,
        #
        'xdmf': xdmf_io,
        'exodus': exodus_io,
        }

    return format_to_reader[file_format].read(filename)


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None,
          file_format=None):
    '''Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    '''
    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    if not file_format:
        # deduce file format from extension
        extension = '.' + os.path.basename(filename).split(os.extsep, 1)[-1]
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
            point_data=point_data,
            cell_data=cell_data,
            write_binary=(file_format == 'ansys-binary')
            )
    elif file_format in ['gmsh-ascii', 'gmsh-binary']:
        gmsh_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            write_binary=(file_format == 'gmsh-binary')
            )
    elif file_format == 'med':
        med_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data)
    elif file_format == 'medit':
        medit_io.write(filename, points, cells)
    elif file_format == 'dolfin-xml':
        dolfin_io.write(filename, points, cells, cell_data=cell_data)
    elif file_format == 'off':
        off_io.write(filename, points, cells)
    elif file_format == 'permas':
        permas_io.write(filename, points, cells)
    elif file_format in ['stl-ascii', 'stl-binary']:
        stl_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            write_binary=(file_format != 'stl-ascii')
            )
    elif file_format == 'vtu-ascii':
        vtu_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            write_binary=False
            )
    elif file_format in ['vtu', 'vtu-binary']:
        vtu_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            write_binary=True
            )
    elif file_format == 'vtk-ascii':
        vtk_io.write(
            filename,
            points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            write_binary=False
            )
    elif file_format in ['vtk', 'vtk-binary']:
        vtk_io.write(
            filename,
            points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            write_binary=True
            )
    elif file_format in ['xdmf', 'xdmf3']:  # XDMF
        xdmf_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    else:
        assert file_format == 'exodus', (
            'Unknown file format \'{}\' of \'{}\'.'
            .format(file_format, filename)
            )
        exodus_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    return
