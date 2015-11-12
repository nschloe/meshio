# -*- coding: utf-8 -*-
#
'''
Module for reading unstructured grids (and related data) from various file
formats.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import h5py
from itertools import islice
import os
import numpy
import re
import vtk


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
        points = \
            vtk.util.numpy_support.vtk_to_numpy(
                vtk_mesh.GetPoints().GetData()
                )
        cells_nodes = _read_cells_nodes(vtk_mesh)
        point_data = _read_point_data(vtk_mesh)
        cell_data = _read_cell_data(vtk_mesh)
        field_data = _read_field_data(vtk_mesh)

        return points, cells_nodes, point_data, cell_data, field_data


def _read_gmsh(filename):
    '''Reads a Gmsh msh file.
    '''
    # The format is specified at
    # <http://geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format>.
    with open(filename) as f:
        while True:
            try:
                line = islice(f, 1).next()
            except StopIteration:
                break
            assert(line[0] == '$')
            environ = line[1:].strip()
            if environ == 'MeshFormat':
                line = islice(f, 1).next()
                # 2.2 0 8
                line = islice(f, 1).next()
                assert(line.strip() == '$EndMeshFormat')
            elif environ == 'Nodes':
                # The first line is the number of nodes
                line = islice(f, 1).next()
                num_nodes = int(line)
                points = numpy.empty((num_nodes, 3))
                for k, line in enumerate(islice(f, num_nodes)):
                    # Throw away the index immediately
                    points[k, :] = numpy.array(line.split(), dtype=float)[1:]
                line = islice(f, 1).next()
                assert(line.strip() == '$EndNodes')
            elif environ == 'Elements':
                # The first line is the number of elements
                line = islice(f, 1).next()
                num_elems = int(line)
                elems = {
                    'points': [],
                    'lines': [],
                    'triangles': [],
                    'tetrahedra': []
                    }
                for k, line in enumerate(islice(f, num_elems)):
                    # Throw away the index immediately
                    data = numpy.array(line.split(), dtype=int)
                    if data[1] == 15:
                        elems['points'].append(data[-1:])
                    elif data[1] == 1:
                        elems['lines'].append(data[-2:])
                    elif data[1] == 2:
                        elems['triangles'].append(data[-3:])
                    elif data[1] == 4:
                        elems['tetrahedra'].append(data[-4:])
                    else:
                        raise RuntimeError('Unknown element type')
                for key in elems:
                    # Subtract one to account for the fact that python indices
                    # are 0-based.
                    elems[key] = numpy.array(elems[key], dtype=int) - 1
                line = islice(f, 1).next()
                assert(line.strip() == '$EndElements')
            else:
                raise RuntimeError('Unknown environment \'%s\'.' % environ)

    if len(elems['tetrahedra']) > 0:
        cells = elems['tetrahedra']
    elif len(elems['triangles']) > 0:
        cells = elems['triangles']
    else:
        raise RuntimeError('Expected at least triangles.')

    return points, cells


def int_to_bool_list(num):
    # From <http://stackoverflow.com/a/33608387/353337>.
    bin_string = format(num, '04b')
    return [x == '1' for x in bin_string[::-1]]


def _read_h5m(filename):
    '''Reads H5M files, cf.
    https://trac.mcs.anl.gov/projects/ITAPS/wiki/MOAB/h5m.
    '''
    f = h5py.File(filename, 'r')
    dset = f['tstt']

    points = dset['nodes']['coordinates'][()]
    # read point data
    point_data = {}
    if 'tags' in dset['nodes']:
        for name, dataset in dset['nodes']['tags'].items():
            point_data[name] = dataset[()]

    # # Assert that the GLOBAL_IDs are contiguous.
    # point_gids = dset['nodes']['tags']['GLOBAL_ID'][()]
    # point_start_gid = dset['nodes']['coordinates'].attrs['start_id']
    # point_end_gid = point_start_gid + len(point_gids) - 1
    # assert all(point_gids == range(point_start_gid, point_end_gid + 1))

    # Note that the indices are off by 1 in h5m.
    if 'Tri3' in dset['elements']:
        elems = dset['elements']['Tri3']
    elif 'Tet4' in dset['elements']:
        elems = dset['elements']['Tet4']
    else:
        raise RuntimeError('Need Tri3s or Tet4s.')

    conn = elems['connectivity']
    cells = conn[()] - 1

    cell_data = {}
    if 'tags' in elems:
        for name, dataset in elems['tags'].items():
            cell_data[name] = dataset[()]

    # The `sets` in H5M are special in that they represent a segration of data
    # in the current file, particularly by a load balancer (Metis, Zoltan,
    # etc.). This segregation has no equivalent in other data types, but is
    # certainly worthwhile visualizing.
    # Hence, we will translate the sets into cell data with the prefix "set::"
    # here.
    # TODO deal with point data
    field_data = {}
    if 'sets' in dset and 'contents' in dset['sets']:
        # read sets
        sets_contents = dset['sets']['contents'][()]
        sets_list = dset['sets']['list'][()]
        sets_tags = dset['sets']['tags']

        cell_start_gid = conn.attrs['start_id']
        cell_gids = cell_start_gid + elems['tags']['GLOBAL_ID'][()]
        cell_end_gid = cell_start_gid + len(cell_gids) - 1
        assert all(cell_gids == range(cell_start_gid, cell_end_gid + 1))

        # create the sets
        for key, value in sets_tags.items():
            mod_key = 'set::' + key
            cell_data[mod_key] = numpy.empty(len(cells), dtype=int)
            end = 0
            for k, row in enumerate(sets_list):
                bits = int_to_bool_list(row[3])
                # is_owner = bits[0]
                # is_unique = bits[1]
                # is_ordered = bits[2]
                is_range_compressed = bits[3]
                if is_range_compressed:
                    start_gids = sets_contents[end:row[0]+1:2]
                    lengths = sets_contents[end+1:row[0]+1:2]
                    for start_gid, length in zip(start_gids, lengths):
                        end_gid = start_gid + length - 1
                        if start_gid >= cell_start_gid and \
                                end_gid <= cell_end_gid:
                            i0 = start_gid - cell_start_gid
                            i1 = end_gid - cell_start_gid + 1
                            cell_data[mod_key][i0:i1] = value[k]
                        else:
                            # TODO deal with point data
                            raise RuntimeError('')
                else:
                    gids = sets_contents[end:row[0]+1]
                    cell_data[mod_key][gids - cell_start_gid] = value[k]

                end = row[0] + 1

    return points, cells, point_data, cell_data, field_data


def _read_vtk_mesh(reader, file_name):
    '''Uses a vtkReader to return a vtkUnstructuredGrid.
    '''
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


# def _read_exodus_mesh(reader, file_name):
#     '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
#     '''
#     reader.SetFileName(file_name)
#
#     # Create Exodus metadata that can be used later when writing the file.
#     reader.ExodusModelMetadataOn()
#
#     # Fetch metadata.
#     reader.UpdateInformation()
#
#     # Make sure the point fields are read during Update().
#     for k in range(reader.GetNumberOfPointArrays()):
#         arr_name = reader.GetPointArrayName(k)
#         reader.SetPointArrayStatus(arr_name, 1)
#
#     # Read the file.
#     reader.Update()
#
#     return reader.GetOutput()


def _read_exodusii_mesh(reader, timestep=None):
    '''Uses a vtkExodusIIReader to return a vtkUnstructuredGrid.
    '''
    # Fetch metadata.
    reader.UpdateInformation()

    # Set time step to read.
    if timestep:
        reader.SetTimeStep(timestep)

    # Make sure the point fields are read during Update().
    for k in range(reader.GetNumberOfPointResultArrays()):
        arr_name = reader.GetPointResultArrayName(k)
        reader.SetPointResultArrayStatus(arr_name, 1)

    # Make sure the point fields are read during Update().
    for k in range(reader.GetNumberOfElementResultArrays()):
        arr_name = reader.GetElementResultArrayName(k)
        reader.SetElementResultArrayStatus(arr_name, 1)

    # Make sure all field data is read.
    for k in range(reader.GetNumberOfGlobalResultArrays()):
        arr_name = reader.GetGlobalResultArrayName(k)
        reader.SetGlobalResultArrayStatus(arr_name, 1)

    # Read the file.
    reader.Update()
    out = reader.GetOutput()

    # Loop through the blocks and search for a vtkUnstructuredGrid.
    vtk_mesh = []
    for i in range(out.GetNumberOfBlocks()):
        blk = out.GetBlock(i)
        for j in range(blk.GetNumberOfBlocks()):
            sub_block = blk.GetBlock(j)
            if sub_block.IsA('vtkUnstructuredGrid'):
                vtk_mesh.append(sub_block)

    if len(vtk_mesh) == 0:
        raise IOError('No \'vtkUnstructuredGrid\' found!')
    elif len(vtk_mesh) > 1:
        raise IOError('More than one \'vtkUnstructuredGrid\' found!')

    # Cut off trailing '_' from array names.
    for k in range(vtk_mesh[0].GetPointData().GetNumberOfArrays()):
        array = vtk_mesh[0].GetPointData().GetArray(k)
        array_name = array.GetName()
        if array_name[-1] == '_':
            array.SetName(array_name[0:-1])

    # time_values = reader.GetOutputInformation(0).Get(
    #     vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    #     )

    return vtk_mesh[0]  # , time_values


def _read_cells_nodes(vtk_mesh):

    num_cells = vtk_mesh.GetNumberOfCells()
    # Assume that all cells have the same number of local nodes.
    max_num_local_nodes = vtk_mesh.GetCell(0).GetNumberOfPoints()
    cells_nodes = numpy.empty(num_cells,
                              dtype=numpy.dtype((int, max_num_local_nodes))
                              )

    for k in range(num_cells):
        cell = vtk_mesh.GetCell(k)
        num_local_nodes = cell.GetNumberOfPoints()
        assert num_local_nodes == max_num_local_nodes, 'Cells not uniform.'
        if num_local_nodes == max_num_local_nodes:
            # Gather up the points.
            for l in range(num_local_nodes):
                cells_nodes[k][l] = cell.GetPointId(l)

    return cells_nodes


def _read_point_data(vtk_data):
    '''Extract point data from a VTK data set.
    '''
    arrays = []
    for k in range(vtk_data.GetPointData().GetNumberOfArrays()):
        arrays.append(vtk_data.GetPointData().GetArray(k))

    # Go through all arrays, fetch psi and A.
    out = {}
    for array in arrays:
        array_name = array.GetName()
        out[array_name] = vtk.util.numpy_support.vtk_to_numpy(array)

    return out


def _read_cell_data(vtk_data):
    '''Extract cell data from a VTK data set.
    '''
    arrays = []
    for k in range(vtk_data.GetCellData().GetNumberOfArrays()):
        arrays.append(vtk_data.GetCellData().GetArray(k))

    out = {}
    for array in arrays:
        array_name = array.GetName()
        out[array_name] = vtk.util.numpy_support.vtk_to_numpy(array)

    return out


def _read_field_data(vtk_data):
    '''Gather field data.
    '''
    vtk_field_data = vtk_data.GetFieldData()
    num_arrays = vtk_field_data.GetNumberOfArrays()

    field_data = {}
    for k in range(num_arrays):
        array = vtk_field_data.GetArray(k)
        name = array.GetName()
        num_values = array.GetDataSize()
        # Data type as specified in vtkSetGet.h.
        data_type = array.GetDataType()
        if data_type == 1:
            dtype = numpy.bool
        elif data_type in [2, 3]:
            dtype = numpy.str
        elif data_type in [4, 5, 6, 7, 8, 9]:
            dtype = numpy.int
        elif data_type in [10, 11]:
            dtype = numpy.float
        else:
            raise TypeError('Unknown VTK data type %d.' % data_type)
        values = numpy.empty(num_values, dtype=dtype)
        for i in range(num_values):
            values[i] = array.GetValue(i)
        field_data[name] = values

    return field_data
