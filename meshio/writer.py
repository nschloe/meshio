# -*- coding: utf-8 -*-
#
from meshio import __name__, __version__

from datetime import datetime
import h5py
import numpy
import os
import re
import vtk
from vtk.util import numpy_support


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


def _write_gmsh(filename, points, cells):
    raise NotImplementedError()


def _write_h5m(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):
    '''Writes H5M files, cf.
    https://trac.mcs.anl.gov/projects/ITAPS/wiki/MOAB/h5m.
    '''

    f = h5py.File(filename, 'w')

    tstt = f.create_group('tstt')

    # The base index for h5m is 1.
    global_id = 1

    # add nodes
    nodes = tstt.create_group('nodes')
    coords = nodes.create_dataset('coordinates', data=points)
    coords.attrs.create('start_id', global_id)
    global_id += len(points)

    # Global tags
    tstt_tags = tstt.create_group('tags')

    # add point data
    if point_data:
        tags = nodes.create_group('tags')
        for key, value in point_data.items():
            tags.create_dataset(key, data=value)
            # Create entry in global tags
            g = tstt_tags.create_group(key)
            g['type'] = value.dtype
            # Add a class tag:
            # From
            # <http://lists.mcs.anl.gov/pipermail/moab-dev/2015/007104.html>:
            # ```
            # /* Was dense tag data in mesh database */
            #  define mhdf_DENSE_TYPE   2
            # /** \brief Was sparse tag data in mesh database */
            # #define mhdf_SPARSE_TYPE  1
            # /** \brief Was bit-field tag data in mesh database */
            # #define mhdf_BIT_TYPE     0
            # /** \brief Unused */
            # #define mhdf_MESH_TYPE    3
            #
            g.attrs['class'] = 2

    # add elements
    elements = tstt.create_group('elements')

    elem_dt = h5py.special_dtype(
        enum=('i', {
            'Edge': 1,
            'Tri': 2,
            'Quad': 3,
            'Polygon': 4,
            'Tet': 5,
            'Pyramid': 6,
            'Prism': 7,
            'Knife': 8,
            'Hex': 9,
            'Polyhedron': 10
            })
        )

    tstt['elemtypes'] = elem_dt

    tstt.create_dataset(
        'history',
        data=[
         __name__,
         __version__,
         str(datetime.now())
        ]
        )

    # number of nodes to h5m name, element type
    h5m_type = {
            2: {'name': 'Edge2', 'type': 1},
            3: {'name': 'Tri3', 'type': 2},
            4: {'name': 'Tet4', 'type': 5}
            }
    this_type = h5m_type[cells.shape[1]]
    elem_group = elements.create_group(this_type['name'])
    elem_group.attrs.create('element_type', this_type['type'], dtype=elem_dt)
    # h5m node indices are 1-based
    conn = elem_group.create_dataset('connectivity', data=(cells + 1))
    conn.attrs.create('start_id', global_id)
    global_id += len(cells)

    # add cell data
    if cell_data:
        tags = elem_group.create_group('tags')
        for key, value in cell_data.items():
            tags.create_dataset(key, data=value)

    # add empty set -- MOAB wants this
    sets = tstt.create_group('sets')
    sets.create_group('tags')

    # set max_id
    tstt.attrs.create('max_id', global_id, dtype='u8')

    return


def _generate_vtk_mesh(points, cellsNodes):
    mesh = vtk.vtkUnstructuredGrid()

    # set points
    vtk_points = vtk.vtkPoints()
    # Not using a deep copy here results in a segfault.
    vtk_array = numpy_support.numpy_to_vtk(points, deep=True)
    vtk_points.SetData(vtk_array)
    mesh.SetPoints(vtk_points)

    # set cells
    # TODO use numpy_support here, avoid the copying
    #
    # # create cell_array. It's a one-dimensional vector with
    # # (num_points2, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
    # numcells, num_local_nodes = cellsNodes.shape
    # cc = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(
    #     numpy.c_[
    #         num_local_nodes * numpy.ones(numcells, dtype=cellsNodes.dtype),
    #         cellsNodes
    #         ],
    #     deep=1
    #     )
    # # wrap the data into a vtkCellArray
    # cell_array = vtk.vtkCellArray()
    # cell_array.SetCells(numcells, cc)

    cell_array = vtk.vtkCellArray()
    for cellNodes in cellsNodes:
        pts = vtk.vtkIdList()
        num_local_nodes = len(cellNodes)
        pts.SetNumberOfIds(num_local_nodes)
        # Get the connectivity for this element.
        for k, node_index in enumerate(cellNodes):
            pts.InsertId(k, node_index)
        cell_array.InsertNextCell(pts)

    numnodes_to_type = {
        2: vtk.VTK_LINE,
        3: vtk.VTK_TRIANGLE,
        4: vtk.VTK_TETRA
        }
    mesh.SetCells(
        numnodes_to_type[num_local_nodes],
        cell_array
        )

    return mesh


def _create_vtkarray(X, name):
    array = vtk.util.numpy_support.numpy_to_vtk(X, deep=1)
    array.SetName(name)
    return array
