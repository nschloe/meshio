from __future__ import annotations

import os
import pathlib
from io import BytesIO
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import ArrayLike

from .._common import cell_data_from_raw, raw_from_cell_data, write_xml
from .._exceptions import ReadError, WriteError
from .._mesh import CellBlock
from .common import (
    attribute_type,
    dtype_to_format_string,
    meshio_to_xdmf_type,
    meshio_type_to_xdmf_index,
    numpy_to_xdmf_dtype,
    translate_mixed_cells,
    xdmf_to_meshio_type,
    xdmf_to_numpy_type,
)


class TimeSeriesReader:
    def __init__(self, filename):  # noqa: C901
        self.filename = pathlib.Path(filename)

        parser = ET.XMLParser()
        tree = ET.parse(self.filename, parser)
        root = tree.getroot()

        if root.tag != "Xdmf":
            raise ReadError()

        version = root.attrib["Version"]
        if version.split(".")[0] != "3":
            raise ReadError(f"Unknown XDMF version {version}.")

        domains = list(root)
        if len(domains) != 1:
            raise ReadError()
        self.domain = domains[0]
        if self.domain.tag != "Domain":
            raise ReadError()

        grids = list(self.domain)

        # find the collection grid
        collection_grid = None
        for g in grids:
            if g.get("GridType") == "Collection":
                collection_grid = g
        if collection_grid is None:
            raise ReadError("Couldn't find the mesh grid")
        if collection_grid.tag != "Grid":
            raise ReadError()
        if collection_grid.get("CollectionType") != "Temporal":
            raise ReadError()

        # get the collection at once
        self.collection = list(collection_grid)
        self.num_steps = len(self.collection)
        self.cells = None
        self.hdf5_files = {}

        # find the uniform grid
        self.mesh_grid = None
        for g in grids:
            if g.get("GridType") == "Uniform":
                self.mesh_grid = g
        # if not found, take the first uniform grid in the collection grid
        if self.mesh_grid is None:
            for g in self.collection:
                if g.get("GridType") == "Uniform":
                    self.mesh_grid = g
                    break
        if self.mesh_grid is None:
            raise ReadError("Couldn't find the mesh grid")
        if self.mesh_grid.tag != "Grid":
            raise ReadError()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        # Those files are opened in _read_data_item()
        for f in self.hdf5_files.values():
            f.close()

    def read_points_cells(self):
        points = None
        cells = []

        assert self.mesh_grid is not None
        for c in self.mesh_grid:
            if c.tag == "Topology":
                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]

                data = self._read_data_item(data_item)

                # The XDMF2 key is `TopologyType`, just `Type` for XDMF3.
                # Allow both.
                if "Type" in c.attrib:
                    if "TopologyType" in c.attrib:
                        raise ReadError("Both 'Type' and 'TopologyType' keys present")
                    cell_type = c.attrib["Type"]
                else:
                    cell_type = c.attrib["TopologyType"]

                if cell_type == "Mixed":
                    cells = translate_mixed_cells(data)
                else:
                    cells.append(CellBlock(xdmf_to_meshio_type[cell_type], data))

            elif c.tag == "Geometry":
                if "GeometryType" in c.attrib:
                    geo_type = c.attrib["GeometryType"]
                    if geo_type not in ["XY", "XYZ"]:
                        raise ReadError(f"Unknown geometry_type '{geo_type}'")

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]
                points = self._read_data_item(data_item)

        self.cells = cells
        return points, cells

    def read_data(self, k: int):
        point_data = {}
        cell_data_raw = {}

        t = None

        for c in list(self.collection[k]):
            if c.tag == "Time":
                t = float(c.attrib["Value"])
            elif c.tag == "Attribute":
                name = c.get("Name")

                if len(list(c)) != 1:
                    raise ReadError()
                data_item = list(c)[0]
                data = self._read_data_item(data_item)

                if c.get("Center") == "Node":
                    point_data[name] = data
                else:
                    if c.get("Center") != "Cell":
                        raise ReadError()
                    cell_data_raw[name] = data
            else:
                # skip the xi:included mesh
                continue

        if self.cells is None:
            raise ReadError()
        cell_data = cell_data_from_raw(self.cells, cell_data_raw)
        if t is None:
            raise ReadError()

        return t, point_data, cell_data

    def _read_data_item(self, data_item):
        dims = [int(d) for d in data_item.get("Dimensions").split()]

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files out there
        # use both keys interchangeably.
        if data_item.get("DataType"):
            if data_item.get("NumberType"):
                raise ReadError()
            data_type = data_item.get("DataType")
        elif data_item.get("NumberType"):
            if data_item.get("DataType"):
                raise ReadError()
            data_type = data_item.get("NumberType")
        else:
            # Default, see
            # <https://xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
            data_type = "Float"

        try:
            precision = data_item.attrib["Precision"]
        except KeyError:
            precision = "4"

        data_format = data_item.attrib["Format"]

        if data_format == "XML":
            return np.fromstring(
                data_item.text,
                dtype=xdmf_to_numpy_type[(data_type, precision)],
                sep=" ",
            ).reshape(dims)
        elif data_format == "Binary":
            return np.fromfile(
                data_item.text.strip(), dtype=xdmf_to_numpy_type[(data_type, precision)]
            ).reshape(dims)

        if data_format != "HDF":
            raise ReadError(f"Unknown XDMF Format '{data_format}'.")

        info = data_item.text.strip()
        filename, h5path = info.split(":")

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        dirpath = self.filename.resolve().parent
        full_hdf5_path = dirpath / filename

        if full_hdf5_path in self.hdf5_files:
            f = self.hdf5_files[full_hdf5_path]
        else:
            import h5py

            f = h5py.File(full_hdf5_path, "r")
            self.hdf5_files[full_hdf5_path] = f

        if h5path[0] != "/":
            raise ReadError()

        for key in h5path[1:].split("/"):
            f = f[key]
        # `[()]` gives a np.ndarray
        return f[()]


class TimeSeriesWriter:
    def __init__(self, filename, data_format: str = "HDF") -> None:
        if data_format not in ["XML", "Binary", "HDF"]:
            raise WriteError(
                "Unknown XDMF data format "
                f"'{data_format}' (use 'XML', 'Binary', or 'HDF'.)"
            )

        self.filename = pathlib.Path(filename)
        self.data_format = data_format
        self.data_counter = 0

        self.xdmf_file = ET.Element("Xdmf", Version="3.0")

        self.domain = ET.SubElement(self.xdmf_file, "Domain")
        self.collection = ET.SubElement(
            self.domain,
            "Grid",
            Name="TimeSeries_meshio",
            GridType="Collection",
            CollectionType="Temporal",
        )

        ET.register_namespace("xi", "https://www.w3.org/2001/XInclude/")

        self.has_mesh = False
        self.mesh_name = "mesh"

    def __enter__(self):
        if self.data_format == "HDF":
            import h5py

            self.h5_filename = self.filename.with_suffix(".h5")
            self.h5_file = h5py.File(self.h5_filename, "w")
        return self

    def __exit__(self, *_):
        write_xml(self.filename, self.xdmf_file)
        if self.data_format == "HDF":
            self.h5_file.close()

    def write_points_cells(
        self,
        points: ArrayLike,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
    ) -> None:
        # <Grid Name="mesh" GridType="Uniform">
        #   <Topology NumberOfElements="16757" TopologyType="Triangle" NodesPerElement="3">
        #     <DataItem Dimensions="16757 3" NumberType="UInt" Format="HDF">maxwell.h5:/Mesh/0/mesh/topology</DataItem>
        #   </Topology>
        #   <Geometry GeometryType="XY">
        #     <DataItem Dimensions="8457 2" Format="HDF">maxwell.h5:/Mesh/0/mesh/geometry</DataItem>
        #   </Geometry>
        # </Grid>
        grid = ET.SubElement(
            self.domain, "Grid", Name=self.mesh_name, GridType="Uniform"
        )
        self.points(grid, np.asarray(points))
        self.cells(cells, grid)
        self.has_mesh = True

    def write_data(self, t, point_data=None, cell_data=None):
        cell_data = {} if cell_data is None else cell_data
        # <Grid>
        #   <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_phi&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
        #   <Time Value="3.3333333333333335e-05" />
        #   <Attribute Name="phi" AttributeType="Scalar" Center="Node">
        #     <DataItem Dimensions="8457 1" Format="HDF">maxwell.h5:/VisualisationVector/1</DataItem>
        #   </Attribute>
        # </Grid>
        grid = ET.SubElement(self.collection, "Grid")
        if not self.has_mesh:
            raise WriteError()
        ptr = f'xpointer(//Grid[@Name="{self.mesh_name}"]/*[self::Topology or self::Geometry])'
        ET.SubElement(grid, "{http://www.w3.org/2003/XInclude}include", xpointer=ptr)
        ET.SubElement(grid, "Time", Value=str(t))

        if point_data:
            self.point_data(point_data, grid)

        # permit old dict structure, convert it to list of tuples
        for name, entry in cell_data.items():
            if isinstance(entry, dict):
                cell_data[name] = np.array(list(entry.values()))
        if cell_data:
            self.cell_data(cell_data, grid)

    def numpy_to_xml_string(self, data):
        if self.data_format == "XML":
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype.name]
            np.savetxt(s, data.flatten(), fmt)
            return s.getvalue().decode()
        elif self.data_format == "Binary":
            bin_filename = f"{self.filename.stem}{self.data_counter}.bin"
            self.data_counter += 1
            # write binary data to file
            with open(bin_filename, "wb") as f:
                data.tofile(f)
            return bin_filename

        if self.data_format != "HDF":
            raise WriteError()
        name = f"data{self.data_counter}"
        self.data_counter += 1
        self.h5_file.create_dataset(name, data=data)
        return os.path.basename(self.h5_filename) + ":/" + name

    def points(self, grid, points):
        if points.shape[1] == 2:
            geometry_type = "XY"
        else:
            if points.shape[1] != 3:
                raise WriteError("Need 3D points.")
            geometry_type = "XYZ"

        geo = ET.SubElement(grid, "Geometry", GeometryType=geometry_type)
        dt, prec = numpy_to_xdmf_dtype[points.dtype.name]
        dim = "{} {}".format(*points.shape)
        data_item = ET.SubElement(
            geo,
            "DataItem",
            DataType=dt,
            Dimensions=dim,
            Format=self.data_format,
            Precision=prec,
        )
        data_item.text = self.numpy_to_xml_string(points)

    def cells(
        self,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
        grid: ET.Element,
    ) -> None:
        if isinstance(cells, dict):
            # convert dict to list of tuples
            cells = list(cells.items())

        # conver to cell_blocks
        cell_blocks = []
        for cell_block in cells:
            if isinstance(cell_block, tuple):
                cell_type, data = cell_block
                cell_block = CellBlock(cell_type, np.asarray(data))
            cell_blocks.append(cell_block)

        if len(cell_blocks) == 1:
            meshio_type = cell_blocks[0].type
            num_cells = len(cell_blocks[0].data)
            xdmf_type = meshio_to_xdmf_type[meshio_type][0]
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType=xdmf_type,
                NumberOfElements=str(num_cells),
            )
            dt, prec = numpy_to_xdmf_dtype[cell_blocks[0].data.dtype.name]
            data_item = ET.SubElement(
                topo,
                "DataItem",
                DataType=dt,
                Dimensions="{} {}".format(*cell_blocks[0].data.shape),
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(cell_blocks[0].data)
        elif len(cell_blocks) > 1:
            total_num_cells = sum(len(c.data) for c in cell_blocks)
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType="Mixed",
                NumberOfElements=str(total_num_cells),
            )
            total_num_cell_items = sum(np.prod(c.data.shape) for c in cell_blocks)
            dim = total_num_cell_items + total_num_cells
            # Lines translate to Polylines, and one needs to specify the exact
            # number of nodes. Hence, prepend 2.
            for c in cell_blocks:
                if c.type == "line":
                    c.data[:] = np.insert(c.data, 0, 2, axis=1)
                    dim += len(c.data)
            dim = str(dim)
            cd = np.concatenate(
                [
                    # prepend column with xdmf type index
                    np.insert(
                        c.data, 0, meshio_type_to_xdmf_index[c.type], axis=1
                    ).flatten()
                    for c in cell_blocks
                ]
            )
            dt, prec = numpy_to_xdmf_dtype[cd.dtype.name]
            data_item = ET.SubElement(
                topo,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(cd)

    def point_data(self, point_data: dict[str, np.ndarray], grid: ET.Element):
        for name, data in point_data.items():
            att = ET.SubElement(
                grid,
                "Attribute",
                Name=name,
                AttributeType=attribute_type(data),
                Center="Node",
            )
            dt, prec = numpy_to_xdmf_dtype[data.dtype.name]
            dim = " ".join([str(s) for s in data.shape])
            data_item = ET.SubElement(
                att,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(data)

    def cell_data(
        self, cell_data: dict[str, list[np.ndarray]], grid: ET.Element
    ) -> None:
        raw = raw_from_cell_data(cell_data)
        for name, data in raw.items():
            att = ET.SubElement(
                grid,
                "Attribute",
                Name=name,
                AttributeType=attribute_type(data),
                Center="Cell",
            )
            dt, prec = numpy_to_xdmf_dtype[data.dtype.name]
            dim = " ".join([str(s) for s in data.shape])
            data_item = ET.SubElement(
                att,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(data)
