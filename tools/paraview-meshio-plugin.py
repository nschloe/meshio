import numpy as np
from paraview.util.vtkAlgorithm import (
    VTKPythonAlgorithmBase,
    smdomain,
    smhint,
    smproperty,
    smproxy,
)
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

import meshio

paraview_plugin_version = meshio.__version__
vtk_to_meshio_type = meshio._vtk_common.vtk_to_meshio_type
meshio_to_vtk_type = meshio._vtk_common.meshio_to_vtk_type
meshio_input_filetypes = list(meshio._helpers.reader_map.keys())
meshio_extensions = [ext[1:] for ext in meshio.extension_to_filetypes.keys()]
meshio_input_filetypes = ["automatic"] + meshio_input_filetypes


@smproxy.reader(
    name="meshio reader",
    extensions=meshio_extensions,
    file_description="meshio-supported files",
    support_reload=False,
)
class MeshioReader(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkUnstructuredGrid"
        )
        self._filename = None
        self._file_format = None

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(
        extensions=meshio_extensions, file_description="meshio-supported files"
    )
    def SetFileName(self, filename):
        if self._filename != filename:
            self._filename = filename
            self.Modified()

    @smproperty.stringvector(name="StringInfo", information_only="1")
    def GetStrings(self):
        return meshio_input_filetypes

    @smproperty.stringvector(name="FileFormat", number_of_elements="1")
    @smdomain.xml(
        """
        <StringListDomain name="list">
            <RequiredProperties>
                <Property name="StringInfo" function="StringInfo"/>
            </RequiredProperties>
        </StringListDomain>
        """
    )
    def SetFileFormat(self, file_format):
        # Automatically deduce input format
        if file_format == "automatic":
            file_format = None

        if self._file_format != file_format:
            self._file_format = file_format
            self.Modified()

    def RequestData(self, request, inInfoVec, outInfoVec):
        output = dsa.WrapDataObject(vtkUnstructuredGrid.GetData(outInfoVec))

        # Use meshio to read the mesh
        mesh = meshio.read(self._filename, self._file_format)
        points, cells = mesh.points, mesh.cells

        # Points
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((len(points), 1))])
        output.SetPoints(points)

        # CellBlock, adapted from test/legacy_writer.py
        cell_types = np.array([], dtype=np.ubyte)
        cell_offsets = np.array([], dtype=int)
        cell_conn = np.array([], dtype=int)
        for meshio_type, data in cells:
            vtk_type = meshio_to_vtk_type[meshio_type]
            ncells, npoints = data.shape
            cell_types = np.hstack(
                [cell_types, np.full(ncells, vtk_type, dtype=np.ubyte)]
            )
            offsets = len(cell_conn) + (1 + npoints) * np.arange(ncells, dtype=int)
            cell_offsets = np.hstack([cell_offsets, offsets])
            conn = np.hstack(
                [npoints * np.ones((ncells, 1), dtype=int), data]
            ).flatten()
            cell_conn = np.hstack([cell_conn, conn])
        output.SetCells(cell_types, cell_offsets, cell_conn)

        # Point data
        for name, array in mesh.point_data.items():
            output.PointData.append(array, name)

        # Cell data
        for name, data in mesh.cell_data.items():
            array = np.concatenate(data)
            output.CellData.append(array, name)

        # Field data
        for name, array in mesh.field_data.items():
            output.FieldData.append(array, name)

        return 1


@smproxy.writer(
    name="meshio Writer",
    extensions=meshio_extensions,
    file_description="meshio-supported files",
    support_reload=False,
)
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(dataTypes=["vtkUnstructuredGrid"], composite_data_supported=False)
class MeshioWriter(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=1, nOutputPorts=0, inputType="vtkUnstructuredGrid"
        )
        self._filename = None

    @smproperty.stringvector(name="FileName", panel_visibility="never")
    @smdomain.filelist()
    def SetFileName(self, filename):
        if self._filename != filename:
            self._filename = filename
            self.Modified()

    def RequestData(self, request, inInfoVec, outInfoVec):
        mesh = dsa.WrapDataObject(vtkUnstructuredGrid.GetData(inInfoVec[0]))

        # Read points
        points = np.asarray(mesh.GetPoints())

        # Read cells
        # Adapted from test/legacy_reader.py
        cell_conn = mesh.GetCells()
        cell_offsets = mesh.GetCellLocations()
        cell_types = mesh.GetCellTypes()
        cells_dict = {}
        for vtk_cell_type in np.unique(cell_types):
            offsets = cell_offsets[cell_types == vtk_cell_type]
            ncells = len(offsets)
            npoints = cell_conn[offsets[0]]
            array = np.empty((ncells, npoints), dtype=int)
            for i in range(npoints):
                array[:, i] = cell_conn[offsets + i + 1]
            cells_dict[vtk_to_meshio_type[vtk_cell_type]] = array
        cells = [meshio.CellBlock(key, cells_dict[key]) for key in cells_dict]

        # Read point and field data
        # Adapted from test/legacy_reader.py
        def _read_data(data):
            out = {}
            for i in range(data.VTKObject.GetNumberOfArrays()):
                name = data.VTKObject.GetArrayName(i)
                array = np.asarray(data.GetArray(i))
                out[name] = array
            return out

        point_data = _read_data(mesh.GetPointData())
        field_data = _read_data(mesh.GetFieldData())

        # Read cell data
        cell_data_flattened = _read_data(mesh.GetCellData())
        cell_data = {}
        for name, array in cell_data_flattened.items():
            cell_data[name] = []
            for cell_type in cells_dict:
                vtk_cell_type = meshio_to_vtk_type[cell_type]
                mask_cell_type = cell_types == vtk_cell_type
                cell_data[name].append(array[mask_cell_type])

        # Use meshio to write mesh
        meshio.write_points_cells(
            self._filename,
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )
        return 1

    def Write(self):
        self.Modified()
        self.Update()
