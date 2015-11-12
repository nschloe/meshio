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
        points = vtk.util.numpy_support.vtk_to_numpy(
                vtk_mesh.GetPoints().GetData()
                )
        cells_nodes = _read_cells_nodes(vtk_mesh)
        point_data = _read_data(vtk_mesh.GetPointData())
        cell_data = _read_data(vtk_mesh.GetCellData())
        field_data = _read_data(vtk_mesh.GetFieldData())

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
    array = vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetCells().GetData())
    # array is a one-dimensional vector with
    # (num_points0, p0, p1, ... ,pk, numpoints1, p10, p11, ..., p1k, ...
    num_nodes_per_cell = array[0]
    assert all(array[::num_nodes_per_cell+1] == num_nodes_per_cell)
    cells = array.reshape(num_cells, num_nodes_per_cell+1)

    # remove first column; it only lists the number of points
    return numpy.delete(cells, 0, 1)


def _read_data(data):
    '''Extract numpy arrays from a VTK data set.
    '''
    # Go through all arrays, fetch data.
    out = {}
    for k in range(data.GetNumberOfArrays()):
        array = data.GetArray(k)
        array_name = array.GetName()
        out[array_name] = vtk.util.numpy_support.vtk_to_numpy(array)

    return out
