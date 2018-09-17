# -*- coding: utf-8 -*-
#
import os

try:
    from StringIO import cStringIO as BytesIO
except ImportError:
    from io import BytesIO

import numpy

from .common import (
    numpy_to_xdmf_dtype,
    dtype_to_format_string,
    meshio_to_xdmf_type,
    meshio_type_to_xdmf_index,
    xdmf_to_numpy_type,
    xdmf_to_meshio_type,
    translate_mixed_cells,
)
from ..common import write_xml, cell_data_from_raw
from ..vtk_io import raw_from_cell_data


class XdmfTimeSeriesReader(object):
    def __init__(self, filename):
        from lxml import etree as ET

        self.filename = filename

        parser = ET.XMLParser(remove_comments=True, huge_tree=True)
        tree = ET.parse(self.filename, parser)
        root = tree.getroot()

        assert root.tag == "Xdmf"

        version = root.attrib["Version"]
        assert version.split(".")[0] == "3", "Unknown XDMF version {}.".format(version)

        domains = list(root)
        assert len(domains) == 1
        self.domain = domains[0]
        assert self.domain.tag == "Domain"

        grids = list(self.domain)

        # find the uniform grid
        self.mesh_grid = None
        for g in grids:
            if g.attrib["GridType"] == "Uniform":
                self.mesh_grid = g
        assert self.mesh_grid is not None, "Couldn't find the mesh grid"
        assert self.mesh_grid.tag == "Grid"

        # find the collection grid
        collection_grid = None
        for g in grids:
            if g.attrib["GridType"] == "Collection":
                collection_grid = g
        assert collection_grid is not None, "Couldn't find the mesh grid"
        assert collection_grid.tag == "Grid"
        assert collection_grid.attrib["CollectionType"] == "Temporal"
        # get the collection at once
        self.collection = list(collection_grid)
        self.num_steps = len(self.collection)
        self.cells = None
        return

    def read_points_cells(self):
        grid = self.mesh_grid

        points = None
        cells = {}

        for c in grid:
            if c.tag == "Topology":
                data_items = list(c)
                assert len(data_items) == 1
                data_item = data_items[0]

                data = self.read_data_item(data_item)

                # The XDMF2 key is `TopologyType`, just `Type` for XDMF3.
                # Allow both.
                if "Type" in c.attrib:
                    assert "TopologyType" not in c.attrib
                    cell_type = c.attrib["Type"]
                else:
                    cell_type = c.attrib["TopologyType"]

                if cell_type == "Mixed":
                    cells = translate_mixed_cells(data)
                else:
                    meshio_type = xdmf_to_meshio_type[cell_type]
                    cells[meshio_type] = data

            elif c.tag == "Geometry":
                try:
                    geometry_type = c.attrib["GeometryType"]
                except KeyError:
                    pass
                else:
                    assert geometry_type in ["XY", "XYZ"]

                data_items = list(c)
                assert len(data_items) == 1
                data_item = data_items[0]
                points = self.read_data_item(data_item)

        self.cells = cells
        return points, cells

    def read_data(self, k):
        point_data = {}
        cell_data_raw = {}

        t = None

        for c in list(self.collection[k]):
            if c.tag == "Time":
                t = float(c.attrib["Value"])
            elif c.tag == "Attribute":
                name = c.attrib["Name"]

                assert len(list(c)) == 1
                data_item = list(c)[0]
                data = self.read_data_item(data_item)

                if c.attrib["Center"] == "Node":
                    point_data[name] = data
                else:
                    assert c.attrib["Center"] == "Cell"
                    cell_data_raw[name] = data
            else:
                # skip the xi:included mesh
                continue

        assert self.cells is not None
        cell_data = cell_data_from_raw(self.cells, cell_data_raw)
        assert t is not None

        return t, point_data, cell_data

    def read_data_item(self, data_item):
        import h5py

        dims = [int(d) for d in data_item.attrib["Dimensions"].split()]

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files
        # out there use both keys interchangeably.
        if "DataType" in data_item.attrib:
            assert "NumberType" not in data_item.attrib
            data_type = data_item.attrib["DataType"]
        elif "NumberType" in data_item.attrib:
            assert "DataType" not in data_item.attrib
            data_type = data_item.attrib["NumberType"]
        else:
            # Default, see
            # <http://www.xdmf.org/index.php/XDMF_Model_and_Format#XML_Element_.28Xdmf_ClassName.29_and_Default_XML_Attributes>
            data_type = "Float"

        try:
            precision = data_item.attrib["Precision"]
        except KeyError:
            precision = "4"

        if data_item.attrib["Format"] == "XML":
            return numpy.array(
                data_item.text.split(), dtype=xdmf_to_numpy_type[(data_type, precision)]
            ).reshape(dims)
        elif data_item.attrib["Format"] == "Binary":
            return numpy.fromfile(
                data_item.text.strip(), dtype=xdmf_to_numpy_type[(data_type, precision)]
            ).reshape(dims)

        assert data_item.attrib["Format"] == "HDF", "Unknown XDMF Format '{}'.".format(
            data_item.attrib["Format"]
        )

        info = data_item.text.strip()
        filename, h5path = info.split(":")

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        full_hdf5_path = os.path.join(os.path.dirname(self.filename), filename)

        f = h5py.File(full_hdf5_path, "r")
        assert h5path[0] == "/"

        for key in h5path[1:].split("/"):
            f = f[key]
        # `[()]` gives a numpy.ndarray
        return f[()]


class XdmfTimeSeriesWriter(object):
    def __init__(self, filename, pretty_xml=True, data_format="HDF"):
        from lxml import etree as ET

        assert data_format in ["XML", "Binary", "HDF"], (
            "Unknown XDMF data format "
            "'{}' (use 'XML', 'Binary', or 'HDF'.)".format(data_format)
        )

        self.filename = filename
        self.data_format = data_format
        self.data_counter = 0
        self.pretty_xml = pretty_xml

        if data_format == "HDF":
            import h5py

            self.h5_filename = os.path.splitext(self.filename)[0] + ".h5"
            self.h5_file = h5py.File(self.h5_filename, "w")

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
        self.mesh_name = None
        return

    def write_points_cells(self, points, cells):
        # <Grid Name="mesh" GridType="Uniform">
        #   <Topology NumberOfElements="16757" TopologyType="Triangle" NodesPerElement="3">
        #     <DataItem Dimensions="16757 3" NumberType="UInt" Format="HDF">maxwell.h5:/Mesh/0/mesh/topology</DataItem>
        #   </Topology>
        #   <Geometry GeometryType="XY">
        #     <DataItem Dimensions="8457 2" Format="HDF">maxwell.h5:/Mesh/0/mesh/geometry</DataItem>
        #   </Geometry>
        # </Grid>
        from lxml import etree as ET

        self.mesh_name = "mesh"
        grid = ET.SubElement(
            self.domain, "Grid", Name=self.mesh_name, GridType="Uniform"
        )
        self.points(grid, points)
        self.cells(cells, grid)
        self.has_mesh = True

        write_xml(self.filename, self.xdmf_file, self.pretty_xml)
        return

    def write_data(self, t, point_data=None, cell_data=None):
        # <Grid>
        #   <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_phi&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
        #   <Time Value="3.3333333333333335e-05" />
        #   <Attribute Name="phi" AttributeType="Scalar" Center="Node">
        #     <DataItem Dimensions="8457 1" Format="HDF">maxwell.h5:/VisualisationVector/1</DataItem>
        #   </Attribute>
        # </Grid>
        from lxml import etree as ET

        grid = ET.SubElement(self.collection, "Grid")
        assert self.has_mesh
        ptr = 'xpointer(//Grid[@Name="{}"]/*[self::Topology or self::Geometry])'.format(
            self.mesh_name
        )
        ET.SubElement(grid, "{http://www.w3.org/2003/XInclude}include", xpointer=ptr)
        ET.SubElement(grid, "Time", Value="{}".format(t))

        if point_data:
            self.point_data(point_data, grid)
        if cell_data:
            self.cell_data(cell_data, grid)

        write_xml(self.filename, self.xdmf_file, self.pretty_xml)
        return

    def numpy_to_xml_string(self, data):
        if self.data_format == "XML":
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype.name]
            numpy.savetxt(s, data.flatten(), fmt)
            return s.getvalue().decode()
        elif self.data_format == "Binary":
            bin_filename = "{}{}.bin".format(
                os.path.splitext(self.filename)[0], self.data_counter
            )
            self.data_counter += 1
            # write binary data to file
            with open(bin_filename, "wb") as f:
                data.tofile(f)
            return bin_filename

        assert self.data_format == "HDF"
        name = "data{}".format(self.data_counter)
        self.data_counter += 1
        self.h5_file.create_dataset(name, data=data)
        return os.path.basename(self.h5_filename) + ":/" + name

    def points(self, grid, points):
        from lxml import etree as ET

        if points.shape[1] == 2:
            geometry_type = "XY"
        else:
            assert points.shape[1] == 3
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
        return

    def cells(self, cells, grid):
        from lxml import etree as ET

        if len(cells) == 1:
            meshio_type = list(cells.keys())[0]
            num_cells = len(cells[meshio_type])
            xdmf_type = meshio_to_xdmf_type[meshio_type][0]
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType=xdmf_type,
                NumberOfElements=str(num_cells),
            )
            dt, prec = numpy_to_xdmf_dtype[cells[meshio_type].dtype.name]
            dim = "{} {}".format(*cells[meshio_type].shape)
            data_item = ET.SubElement(
                topo,
                "DataItem",
                DataType=dt,
                Dimensions=dim,
                Format=self.data_format,
                Precision=prec,
            )
            data_item.text = self.numpy_to_xml_string(cells[meshio_type])
        elif len(cells) > 1:
            total_num_cells = sum(c.shape[0] for c in cells.values())
            topo = ET.SubElement(
                grid,
                "Topology",
                TopologyType="Mixed",
                NumberOfElements=str(total_num_cells),
            )
            total_num_cell_items = sum(numpy.prod(c.shape) for c in cells.values())
            dim = total_num_cell_items + total_num_cells
            # Lines translate to Polylines, and one needs to specify the exact
            # number of nodes. Hence, prepend 2.
            if "line" in cells:
                cells["line"] = numpy.insert(cells["line"], 0, 2, axis=1)
                dim += len(cells["line"])
            dim = str(dim)
            cd = numpy.concatenate(
                [
                    # prepend column with xdmf type index
                    numpy.insert(
                        value, 0, meshio_type_to_xdmf_index[key], axis=1
                    ).flatten()
                    for key, value in cells.items()
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
        return

    def point_data(self, point_data, grid):
        from lxml import etree as ET

        for name, data in point_data.items():
            att = ET.SubElement(grid, "Attribute", Name=name, Center="Node")
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
        return

    def cell_data(self, cell_data, grid):
        from lxml import etree as ET

        raw = raw_from_cell_data(cell_data)
        for name, data in raw.items():
            att = ET.SubElement(
                grid, "Attribute", Name=name, Type="None", Center="Cell"
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
        return
