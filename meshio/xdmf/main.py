"""
I/O for XDMF.
http://www.xdmf.org/index.php/XDMF_Model_and_Format
"""
import os
from io import BytesIO

import numpy

from .._common import cell_data_from_raw, raw_from_cell_data, write_xml
from .._exceptions import ReadError, WriteError
from .._helpers import register
from .._mesh import Mesh
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


def read(filename):
    return XdmfReader(filename).read()


class XdmfReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        from lxml import etree as ET

        parser = ET.XMLParser(remove_comments=True, huge_tree=True)
        tree = ET.parse(self.filename, parser)
        root = tree.getroot()

        if root.tag != "Xdmf":
            raise ReadError()

        version = root.attrib["Version"]

        if version.split(".")[0] == "2":
            return self.read_xdmf2(root)

        if version.split(".")[0] != "3":
            raise ReadError("Unknown XDMF version {}.".format(version))

        return self.read_xdmf3(root)

    def _read_data_item(self, data_item):
        import h5py

        dims = [int(d) for d in data_item.attrib["Dimensions"].split()]

        # Actually, `NumberType` is XDMF2 and `DataType` XDMF3, but many files
        # out there use both keys interchangeably.
        if "DataType" in data_item.attrib:
            if "NumberType" in data_item.attrib:
                raise ReadError()
            data_type = data_item.attrib["DataType"]
        elif "NumberType" in data_item.attrib:
            if "DataType" in data_item.attrib:
                raise ReadError()
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

        if data_item.attrib["Format"] != "HDF":
            raise ReadError(
                "Unknown XDMF Format '{}'.".format(data_item.attrib["Format"])
            )

        info = data_item.text.strip()
        filename, h5path = info.split(":")

        # The HDF5 file path is given with respect to the XDMF (XML) file.
        full_hdf5_path = os.path.join(os.path.dirname(self.filename), filename)

        f = h5py.File(full_hdf5_path, "r")
        if h5path[0] != "/":
            raise ReadError()

        for key in h5path[1:].split("/"):
            f = f[key]
        # `[()]` gives a numpy.ndarray
        return f[()]

    def read_information(self, c_data):
        from lxml import etree as ET

        field_data = {}
        root = ET.fromstring(c_data)
        for child in root:
            str_tag = child.attrib["key"]
            dim = int(child.attrib["dim"])
            num_tag = int(child.text)
            field_data[str_tag] = numpy.array([num_tag, dim])
        return field_data

    def read_xdmf2(self, root):  # noqa: C901
        domains = list(root)
        if len(domains) != 1:
            raise ReadError()
        domain = domains[0]
        if domain.tag != "Domain":
            raise ReadError()

        grids = list(domain)
        if len(grids) != 1:
            raise ReadError("XDMF reader: Only supports one grid right now.")
        grid = grids[0]
        if grid.tag != "Grid":
            raise ReadError()

        if "GridType" in grid.attrib and grid.attrib["GridType"] != "Uniform":
            raise ReadError()

        points = None
        cells = {}
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
            if c.tag == "Topology":
                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                topology_type = c.attrib["TopologyType"]
                if topology_type == "Mixed":
                    cells = translate_mixed_cells(
                        numpy.fromstring(
                            data_items[0].text,
                            int,
                            int(data_items[0].attrib["Dimensions"]),
                            " ",
                        )
                    )
                else:
                    meshio_type = xdmf_to_meshio_type[topology_type]
                    cells[meshio_type] = self._read_data_item(data_items[0])

            elif c.tag == "Geometry":
                if "GeometryType" in c.attrib and c.attrib["GeometryType"] != "XYZ":
                    raise ReadError()
                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                points = self._read_data_item(data_items[0])

            elif c.tag == "Information":
                c_data = c.text
                if not c_data:
                    raise ReadError()
                field_data = self.read_information(c_data)

            elif c.tag == "Attribute":
                # assert c.attrib['Active'] == '1'
                # assert c.attrib['AttributeType'] == 'None'

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()

                data = self._read_data_item(data_items[0])

                name = c.attrib["Name"]
                if c.attrib["Center"] == "Node":
                    point_data[name] = data
                elif c.attrib["Center"] == "Cell":
                    cell_data_raw[name] = data
                else:
                    # TODO field data?
                    if c.attrib["Center"] != "Grid":
                        raise ReadError()
            else:
                raise ReadError("Unknown section '{}'.".format(c.tag))

        cell_data = cell_data_from_raw(cells, cell_data_raw)

        return Mesh(
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

    def read_xdmf3(self, root):  # noqa: C901
        domains = list(root)
        if len(domains) != 1:
            raise ReadError()
        domain = domains[0]
        if domain.tag != "Domain":
            raise ReadError()

        grids = list(domain)
        if len(grids) != 1:
            raise ReadError("XDMF reader: Only supports one grid right now.")
        grid = grids[0]
        if grid.tag != "Grid":
            raise ReadError()

        points = None
        cells = {}
        point_data = {}
        cell_data_raw = {}
        field_data = {}

        for c in grid:
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
                        raise ReadError()
                    cell_type = c.attrib["Type"]
                else:
                    cell_type = c.attrib["TopologyType"]

                if cell_type == "Mixed":
                    cells = translate_mixed_cells(data)
                else:
                    cells[xdmf_to_meshio_type[cell_type]] = data

            elif c.tag == "Geometry":
                try:
                    geometry_type = c.attrib["GeometryType"]
                except KeyError:
                    pass
                else:
                    if geometry_type not in ["XY", "XYZ"]:
                        raise ReadError()

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]
                points = self._read_data_item(data_item)

            elif c.tag == "Information":
                c_data = c.text
                if not c_data:
                    raise ReadError()
                field_data = self.read_information(c_data)

            elif c.tag == "Attribute":
                # Don't be too strict here: FEniCS, for example, calls this
                # 'AttributeType'.
                # assert c.attrib['Type'] == 'None'

                data_items = list(c)
                if len(data_items) != 1:
                    raise ReadError()
                data_item = data_items[0]

                data = self._read_data_item(data_item)

                name = c.attrib["Name"]
                if c.attrib["Center"] == "Node":
                    point_data[name] = data
                else:
                    if c.attrib["Center"] != "Cell":
                        raise ReadError()
                    cell_data_raw[name] = data
            else:
                raise ReadError("Unknown section '{}'.".format(c.tag))

        cell_data = cell_data_from_raw(cells, cell_data_raw)

        return Mesh(
            points,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )


class XdmfWriter:
    def __init__(
        self,
        filename,
        mesh,
        pretty_xml=True,
        data_format="HDF",
        compression=None,
        compression_opts=None,
    ):
        from lxml import etree as ET

        if data_format not in ["XML", "Binary", "HDF"]:
            raise WriteError(
                "Unknown XDMF data format "
                "'{}' (use 'XML', 'Binary', or 'HDF'.)".format(data_format)
            )

        self.filename = filename
        self.data_format = data_format
        self.data_counter = 0
        self.compression = compression
        self.compression_opts = compression_opts

        if data_format == "HDF":
            import h5py

            self.h5_filename = os.path.splitext(self.filename)[0] + ".h5"
            self.h5_file = h5py.File(self.h5_filename, "w")

        xdmf_file = ET.Element("Xdmf", Version="3.0")

        domain = ET.SubElement(xdmf_file, "Domain")
        grid = ET.SubElement(domain, "Grid", Name="Grid")
        information = ET.SubElement(
            grid, "Information", Name="Information", Value=str(len(mesh.field_data))
        )

        self.points(grid, mesh.points)
        self.field_data(mesh.field_data, information)
        self.cells(mesh.cells, grid)
        self.point_data(mesh.point_data, grid)
        self.cell_data(mesh.cell_data, grid)

        ET.register_namespace("xi", "https://www.w3.org/2001/XInclude/")

        write_xml(filename, xdmf_file, pretty_xml)

    def numpy_to_xml_string(self, data):
        if self.data_format == "XML":
            s = BytesIO()
            fmt = dtype_to_format_string[data.dtype.name]
            numpy.savetxt(s, data, fmt)
            return "\n" + s.getvalue().decode()
        elif self.data_format == "Binary":
            bin_filename = "{}{}.bin".format(
                os.path.splitext(self.filename)[0], self.data_counter
            )
            self.data_counter += 1
            # write binary data to file
            with open(bin_filename, "wb") as f:
                data.tofile(f)
            return bin_filename

        if self.data_format != "HDF":
            raise WriteError()
        name = "data{}".format(self.data_counter)
        self.data_counter += 1
        self.h5_file.create_dataset(
            name,
            data=data,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        return os.path.basename(self.h5_filename) + ":/" + name

    def points(self, grid, points):
        from lxml import etree as ET

        if points.shape[1] == 1:
            geometry_type = "X"
        elif points.shape[1] == 2:
            geometry_type = "XY"
        else:
            if points.shape[1] != 3:
                raise WriteError()
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
            dim = str(
                total_num_cell_items
                + total_num_cells
                + (cells["line"].shape[0] if "line" in cells else 0)
            )
            cd = numpy.concatenate(
                [
                    numpy.hstack(
                        [
                            numpy.full(
                                (value.shape[0], 2 if key == "line" else 1),
                                meshio_type_to_xdmf_index[key],
                            ),
                            value,
                        ]
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

    def point_data(self, point_data, grid):
        from lxml import etree as ET

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

    def cell_data(self, cell_data, grid):
        from lxml import etree as ET

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

    def field_data(self, field_data, information):
        from lxml import etree as ET

        info = ET.Element("main")
        for name, data in field_data.items():
            data_item = ET.SubElement(info, "map", key=name, dim=str(data[1]))
            data_item.text = str(data[0])
        information.text = ET.CDATA(ET.tostring(info))


def write(*args, **kwargs):
    XdmfWriter(*args, **kwargs)


register(
    "xdmf",
    [".xdmf", ".xmf"],
    read,
    {
        "xdmf": write,
        "xdmf-binary": lambda f, m, **kwargs: write(f, m, data_format="Binary"),
        "xdmf-hdf": lambda f, m, **kwargs: write(f, m, data_format="HDF"),
        "xdmf-xml": lambda f, m, **kwargs: write(f, m, data_format="XML"),
        "xdmf3": write,
        "xdmf3-binary": lambda f, m, **kwargs: write(f, m, data_format="Binary"),
        "xdmf3-hdf": lambda f, m, **kwargs: write(f, m, data_format="HDF"),
        "xdmf3-xml": lambda f, m, **kwargs: write(f, m, data_format="XML"),
    },
)
