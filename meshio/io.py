# -*- coding: utf-8 -*-
#
'''
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from itertools import islice
import os
import numpy
import re
import vtk

from meshio import __name__, __version__

from datetime import datetime
import h5py
import numpy
import os
import re
import vtk
from vtk.util import numpy_support


def read(filenames, timestep=None):
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
    if isinstance(filenames, (list, tuple)) and len(filenames) == 1:
        filenames = filenames[0]

    # http://stackoverflow.com/questions/4843173/how-to-check-if-a-type-of-variable-is-string-in-python
    # if isinstance(filenames, basestring):
    if isinstance(filenames, str):
        filename = filenames
        # serial files
        extension = os.path.splitext(filename)[1]

        # setup the reader
        if extension == '.msh':
            # Gmsh file
            points, cells_nodes = _read_gmsh(filename)
            return points, cells_nodes, None, None, None
        # setup the reader
        elif extension == '.h5m':
            # H5M file
            return _read_h5m(filename)
        else:
            if extension == '.vtu':
                reader = vtk.vtkXMLUnstructuredGridReader()
                vtk_mesh = _read_vtk_mesh(reader, filename)
            elif extension == '.vtk':
                reader = vtk.vtkUnstructuredGridReader()
                vtk_mesh = _read_vtk_mesh(reader, filename)
            elif extension in ['.ex2', '.exo', '.e']:
                reader = vtk.vtkExodusIIReader()
                reader.SetFileName(filename)
                vtk_mesh = _read_exodusii_mesh(reader, timestep=timestep)
            elif re.match('[^\.]*\.e\.\d+\.\d+', filename):
                # Parallel Exodus files.
                # TODO handle with vtkPExodusIIReader
                reader = vtk.vtkExodusIIReader()
                reader.SetFileName(filenames[0])
                vtk_mesh = _read_exodusii_mesh(reader, timestep=timestep)
            else:
                raise RuntimeError('Unknown file type \'%s\'.' % filename)

        # # Parallel files.
        # # Assume Exodus format as we don't know anything else yet.
        # # TODO Guess the file pattern or whatever.
        # reader = vtk.vtkPExodusIIReader()
        # reader.SetFileNames(filenames)
        # vtk_mesh = _read_exodusii_mesh(reader, filename, timestep=timestep)

        # Explicitly extract points, cells, point data, field data
        points = vtk.util.numpy_support.vtk_to_numpy(
                vtk_mesh.GetPoints().GetData()
                )
        cells_nodes = _read_cells_nodes(vtk_mesh)
        point_data = _read_data(vtk_mesh.GetPointData())
        cell_data = _read_data(vtk_mesh.GetCellData())
        field_data = _read_data(vtk_mesh.GetFieldData())

        return points, cells_nodes, point_data, cell_data, field_data


def write(filename,
          points,
          cells,
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
    extension = os.path.splitext(filename)[1]

    if extension == '.h5m':
        _write_h5m(
            filename,
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data
            )
        return
    elif extension == '.msh':
        _write_gmsh(filename, points, cells)
        return
    else:
        vtk_mesh = _generate_vtk_mesh(points, cells)
        # add point data
        if point_data:
            for name, X in point_data.iteritems():
                # There is a naming inconsistency in VTK when it comes to
                # multivectors in Exodus files:
                # If a vector 'v' has two components, they are called 'v_r',
                # 'v_z' (note the underscore), if it has three, then they are
                # called 'vx', 'vy', 'vz'.  Make this consistent by appending
                # an underscore if needed.  Note that for VTK files, this
                # problem does not occur since the label of a vector is always
                # stored as a string.
                is_exodus_format = extension in ['.ex2', '.exo', '.e']
                if is_exodus_format and len(X.shape) == 2 \
                   and X.shape[1] == 3 and name[-1] != '_':
                    name += '_'
                pd = vtk_mesh.GetPointData()
                pd.AddArray(_create_vtkarray(X, name))

        # add cell data
        if cell_data:
            for key, value in cell_data.iteritems():
                cd = vtk_mesh.GetCellData()
                cd.AddArray(_create_vtkarray(value, key))

        # add field data
        if field_data:
            for key, value in field_data.iteritems():
                vtk_mesh.GetFieldData() \
                        .AddArray(_create_vtkarray(value, key))

        if extension == '.vtu':  # VTK XML format
            writer = vtk.vtkXMLUnstructuredGridWriter()
        elif extension == '.pvtu':  # parallel VTK XML format
            writer = vtk.vtkXMLPUnstructuredGridWriter()
        elif extension == '.vtk':  # classical VTK format
            writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileTypeToASCII()
        elif extension in ['.ex2', '.exo', '.e']:  # Exodus II format
            writer = vtk.vtkExodusIIWriter()
            # If the mesh contains vtkModelData information, make use of it
            # and write out all time steps.
            writer.WriteAllTimeStepsOn()
        elif re.match('[^\.]*\.e\.\d+\.\d+', filename):
            # TODO handle parallel I/O with vtkPExodusIIWriter
            writer = vtk.vtkExodusIIWriter()
            # If the mesh contains vtkModelData information, make use of it
            # and write out all time steps.
            writer.WriteAllTimeStepsOn()
        else:
            raise IOError('Unknown file type \'%s\'.' % filename)
        writer.SetFileName(filename)
        writer.SetInput(vtk_mesh)
        writer.Write()
        return
