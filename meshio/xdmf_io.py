# -*- coding: utf-8 -*-
#
'''
I/O for VTU <https://www.vtk.org/Wiki/VTK_XML_Formats>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy

from .vtk_io import translate_cells

# Make explicit copies of the data; some (all?) of it is quite volatile and
# contains garbage once the vtk_mesh goes out of scopy.


def read(filetype, filename):
    # pylint: disable=import-error
    import vtk
    from vtk.util import numpy_support

    def _read_data(data):
        '''Extract numpy arrays from a VTK data set.
        '''
        # Go through all arrays, fetch data.
        out = {}
        for k in range(data.GetNumberOfArrays()):
            array = data.GetArray(k)
            if array:
                array_name = array.GetName()
                out[array_name] = numpy.copy(
                    vtk.util.numpy_support.vtk_to_numpy(array)
                    )
        return out

    if filetype in ['xdmf', 'xdmf2']:
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(filename)
        reader.SetReadAllColorScalars(1)
        reader.SetReadAllFields(1)
        reader.SetReadAllNormals(1)
        reader.SetReadAllScalars(1)
        reader.SetReadAllTCoords(1)
        reader.SetReadAllTensors(1)
        reader.SetReadAllVectors(1)
        reader.Update()
        vtk_mesh = reader.GetOutputDataObject(0)
    else:
        assert filetype == 'xmdf3', \
            'Unknown file type \'{}\'.'.format(filename)
        reader = vtk.vtkXdmf3Reader()
        reader.SetFileName(filename)
        reader.SetReadAllColorScalars(1)
        reader.SetReadAllFields(1)
        reader.SetReadAllNormals(1)
        reader.SetReadAllScalars(1)
        reader.SetReadAllTCoords(1)
        reader.SetReadAllTensors(1)
        reader.SetReadAllVectors(1)
        reader.Update()
        vtk_mesh = reader.GetOutputDataObject(0)

    # Explicitly extract points, cells, point data, field data
    points = numpy.copy(numpy_support.vtk_to_numpy(
            vtk_mesh.GetPoints().GetData()
            ))

    data = numpy.copy(vtk.util.numpy_support.vtk_to_numpy(
            vtk_mesh.GetCells().GetData()
            ))
    offsets = numpy.copy(vtk.util.numpy_support.vtk_to_numpy(
            vtk_mesh.GetCellLocationsArray()
            ))
    types = numpy.copy(vtk.util.numpy_support.vtk_to_numpy(
            vtk_mesh.GetCellTypesArray()
            ))
    cells = translate_cells(data, offsets, types)

    point_data = _read_data(vtk_mesh.GetPointData())
    field_data = _read_data(vtk_mesh.GetFieldData())

    cell_data = _read_data(vtk_mesh.GetCellData())
    # split cell_data by the cell type
    cd = {}
    index = 0
    for cell_type in cells:
        num_cells = len(cells[cell_type])
        cd[cell_type] = {}
        for name, array in cell_data.items():
            cd[cell_type][name] = array[index:index+num_cells]
        index += num_cells
    cell_data = cd

    return points, cells, point_data, cell_data, field_data


def write(filetype,
          filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None
          ):
    # pylint: disable=import-error
    from .vtk_io import write as vtk_write
    return vtk_write(
        filetype, filename, points, cells,
        point_data=point_data, cell_data=cell_data, field_data=field_data
        )
