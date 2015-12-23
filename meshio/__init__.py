# -*- coding: utf-8 -*-
#
from . import dolfin_io
from . import h5m_io
from . import msh_io
from . import permas_io
from . import vtk_io

from meta import __version__, __author__, __author_email__, __website__


_extension_to_filetype = {
    'e': 'exodus',
    'ex2': 'exodus',
    'exo': 'exodus',
    'msh': 'gmsh',
    'xml': 'dolfin-xml',
    'post': 'permas',
    'post.gz': 'permas',
    'dato': 'permas',
    'dato.gz': 'permas',
    'h5m': 'moab',
    'vtu': 'vtu',
    'vtk': 'vtk',
    'xdmf': 'xdmf',
    'xmf': 'xdmf',
    }


def read(filename, file_type=None, timestep=None):
    '''Reads an unstructured mesh with added data.

    :param filenames: The files to read from.
    :type filenames: str
    :param timestep: Time step to read from, in case of an Exodus input mesh.
    :type timestep: int, optional
    :returns mesh{2,3}d: The mesh data.
    :returns point_data: Point data read from file.
    :type point_data: dict
    :returns field_data: Field data read from file.
    :type field_data: dict
    '''
    import os

    # http://stackoverflow.com/questions/4843173/how-to-check-if-a-type-of-variable-is-string-in-python
    assert isinstance(filename, str)

    if not file_type:
        # deduct file type from extension
        extension = filename.split(os.extsep, 1)[1]
        file_type = _extension_to_filetype[extension]

    if file_type == 'gmsh':
        return msh_io.read(filename)
    elif file_type == 'dolfin-xml':
        return dolfin_io.read(filename)
    elif file_type == 'permas':
        return permas_io.read(filename)
    elif file_type == 'moab':
        return h5m_io.read(filename)
    elif file_type == 'vtu':
        return vtk_io.read('vtu', filename)
    elif file_type == '.vtk':
        return vtk_io.read('vtk', filename)
    elif file_type == 'xdmf':
        return vtk_io.read('xdmf', filename)
    elif file_type == 'exodus':
        return vtk_io.read('exodus', filename)
    else:
        raise RuntimeError('Unknown file type \'%s\'.' % filename)


def write(filename,
          points,
          cells,
          file_type=None,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    '''Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    '''
    import os

    if not file_type:
        # deduct file type from extension
        extension = filename.split(os.extsep, 1)[1]
        file_type = _extension_to_filetype[extension]

    if file_type == 'moab':
        h5m_io.write(
            filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_type == 'gmsh':
        msh_io.write(filename, points, cells)
    elif file_type == 'dolfin-xml':
        dolfin_io.write(filename, points, cells)
    elif file_type == 'permas':
        permas_io.write(filename, points, cells)
    elif file_type == 'vtu':  # vtk xml format
        vtk_io.write(
            'vtu', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_type == 'vtk':  # classical vtk format
        vtk_io.write(
            'vtk', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_type == 'xdmf':  # XDMF
        vtk_io.write(
            'xdmf', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    elif file_type == 'exodus':  # exodus ii format
        vtk_io.write(
            'exodus', filename, points, cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
    else:
        raise RuntimeError('unknown file type \'%s\'.' % filename)
    return
