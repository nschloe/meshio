# -*- coding: utf-8 -*-
#
"""
I/O for VTU.
"""
import base64
import logging

try:
    from StringIO import cStringIO as BytesIO
except ImportError:
    from io import BytesIO
import sys
import zlib

import numpy

from .__about__ import __version__
from .mesh import Mesh
from .vtk_io import vtk_to_meshio_type, meshio_to_vtk_type, raw_from_cell_data
from .common import num_nodes_per_cell, write_xml


def num_bytes_to_num_base64_chars(num_bytes):
    # Rounding up in integer division works by double negation since Python
    # always rounds down.
    return -(-num_bytes // 3) * 4


def _cells_from_data(connectivity, offsets, types, cell_data_raw):
    # Translate it into the cells dictionary.
    # `connectivity` is a one-dimensional vector with
    # (p0, p1, ... ,pk, p10, p11, ..., p1k, ...

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    uniques = numpy.unique(types)
    bins = {u: numpy.where(types == u)[0] for u in uniques}

    assert len(offsets) == len(types)

    cells = {}
    cell_data = {}
    for tpe, b in bins.items():
        meshio_type = vtk_to_meshio_type[tpe]
        n = num_nodes_per_cell[meshio_type]
        # The offsets point to the _end_ of the indices
        indices = numpy.add.outer(offsets[b], numpy.arange(-n, 0))
        cells[meshio_type] = connectivity[indices]
        cell_data[meshio_type] = {key: value[b] for key, value in cell_data_raw.items()}

    return cells, cell_data


def _organize_cells(point_offsets, cells, cell_data_raw):
    assert len(point_offsets) == len(cells)

    out_cells = {}
    out_cell_data = {}
    for offset, cls, cdr in zip(point_offsets, cells, cell_data_raw):
        cls, cell_data = _cells_from_data(
            cls["connectivity"], cls["offsets"], cls["types"], cdr
        )
        for key in cls:
            if key not in out_cells:
                out_cells[key] = []
            out_cells[key].append(cls[key] + offset)

            if key not in out_cell_data:
                out_cell_data[key] = {}

            for name in cell_data[key]:
                if name not in out_cell_data[key]:
                    out_cell_data[key][name] = []
                out_cell_data[key][name].append(cell_data[key][name])

    for key in out_cells:
        out_cells[key] = numpy.concatenate(out_cells[key])
        for name in out_cell_data[key]:
            out_cell_data[key][name] = numpy.concatenate(out_cell_data[key][name])

    return out_cells, out_cell_data


def get_grid(root):
    grid = None
    appended_data = None
    for c in root:
        if c.tag == "UnstructuredGrid":
            assert grid is None, "More than one UnstructuredGrid found."
            grid = c
        else:
            assert c.tag == "AppendedData", "Unknown main tag '{}'.".format(c.tag)
            assert appended_data is None, "More than one AppendedData found."
            assert c.attrib["encoding"] == "base64"
            appended_data = c.text.strip()
            # The appended data always begins with a (meaningless)
            # underscore.
            assert appended_data[0] == "_"
            appended_data = appended_data[1:]

    assert grid is not None, "No UnstructuredGrid found."
    return grid, appended_data


vtu_to_numpy_type = {
    "Float32": numpy.dtype(numpy.float32),
    "Float64": numpy.dtype(numpy.float64),
    "Int8": numpy.dtype(numpy.int8),
    "Int16": numpy.dtype(numpy.int16),
    "Int32": numpy.dtype(numpy.int32),
    "Int64": numpy.dtype(numpy.int64),
    "UInt8": numpy.dtype(numpy.uint8),
    "UInt16": numpy.dtype(numpy.uint16),
    "UInt32": numpy.dtype(numpy.uint32),
    "UInt64": numpy.dtype(numpy.uint64),
}
numpy_to_vtu_type = {v: k for k, v in vtu_to_numpy_type.items()}


class VtuReader(object):
    """Helper class for reading VTU files. Some properties are global to the
    file (e.g., byte_order), and instead of passing around these parameters,
    make them properties of this class.
    """

    def __init__(self, filename):
        from lxml import etree as ET

        # libxml2 and with it lxml have a safety net for memory overflows; see,
        # e.g., <https://stackoverflow.com/q/33828728/353337>.
        # This causes the error
        # ```
        # cannot parse large files and instead throws the exception
        #
        # lxml.etree.XMLSyntaxError: xmlSAX2Characters: huge text node, [...]
        # ```
        # Setting huge_tree=True removes the limit. Another alternative would
        # be to use Python's native xml parser to avoid this error,
        # import xml.etree.cElementTree as ET
        parser = ET.XMLParser(remove_comments=True, huge_tree=True)
        tree = ET.parse(filename, parser)
        root = tree.getroot()

        assert root.tag == "VTKFile"
        assert root.attrib["type"] == "UnstructuredGrid"
        assert root.attrib["version"] in [
            "0.1",
            "1.0",
        ], "Unknown VTU file version '{}'.".format(root.attrib["version"])

        try:
            assert root.attrib["compressor"] == "vtkZLibDataCompressor"
        except KeyError:
            pass

        self.header_type = (
            root.attrib["header_type"] if "header_type" in root.attrib else "UInt32"
        )

        try:
            self.byte_order = root.attrib["byte_order"]
            assert self.byte_order in [
                "LittleEndian",
                "BigEndian",
            ], "Unknown byte order '{}'.".format(self.byte_order)
        except KeyError:
            self.byte_order = None

        grid, appended_data = get_grid(root)

        pieces = []
        field_data = {}
        for c in grid:
            if c.tag == "Piece":
                pieces.append(c)
            else:
                assert c.tag == "FieldData", "Unknown grid subtag '{}'.".format(c.tag)
                # TODO test field data
                for data_array in c:
                    field_data[data_array.attrib["Name"]] = self.read_data(data_array)

        assert pieces, "No Piece found."

        points = []
        cells = []
        point_data = []
        cell_data_raw = []

        for piece in pieces:
            piece_cells = {}
            piece_point_data = {}
            piece_cell_data_raw = {}

            num_points = int(piece.attrib["NumberOfPoints"])
            num_cells = int(piece.attrib["NumberOfCells"])

            for child in piece:
                if child.tag == "Points":
                    data_arrays = list(child)
                    assert len(data_arrays) == 1
                    data_array = data_arrays[0]

                    assert data_array.tag == "DataArray"

                    pts = self.read_data(data_array)

                    num_components = int(data_array.attrib["NumberOfComponents"])
                    points.append(pts.reshape(num_points, num_components))

                elif child.tag == "Cells":
                    for data_array in child:
                        assert data_array.tag == "DataArray"
                        piece_cells[data_array.attrib["Name"]] = self.read_data(
                            data_array
                        )

                    assert len(piece_cells["offsets"]) == num_cells
                    assert len(piece_cells["types"]) == num_cells

                    cells.append(piece_cells)

                elif child.tag == "PointData":
                    for c in child:
                        assert c.tag == "DataArray"
                        piece_point_data[c.attrib["Name"]] = self.read_data(c)

                    point_data.append(piece_point_data)

                else:
                    assert child.tag == "CellData", "Unknown tag '{}'.".format(
                        child.tag
                    )

                    for c in child:
                        assert c.tag == "DataArray"
                        piece_cell_data_raw[c.attrib["Name"]] = self.read_data(c)

                    cell_data_raw.append(piece_cell_data_raw)

        if not cell_data_raw:
            cell_data_raw = [{}] * len(cells)

        assert len(cell_data_raw) == len(cells)

        point_offsets = (
            numpy.cumsum([pts.shape[0] for pts in points]) - points[0].shape[0]
        )

        # Now merge across pieces
        assert points
        self.points = numpy.concatenate(points)

        if point_data:
            self.point_data = {
                key: numpy.concatenate([pd[key] for pd in point_data])
                for key in point_data[0]
            }
        else:
            self.point_data = None

        self.cells, self.cell_data = _organize_cells(
            point_offsets, cells, cell_data_raw
        )

        self.field_data = field_data
        return

    def read_binary(self, data, data_type):
        # first read the the block size; it determines the size of the header
        dtype = vtu_to_numpy_type[self.header_type]
        num_bytes_per_item = numpy.dtype(dtype).itemsize
        num_chars = num_bytes_to_num_base64_chars(num_bytes_per_item)
        byte_string = base64.b64decode(data[:num_chars])[:num_bytes_per_item]
        num_blocks = numpy.frombuffer(byte_string, dtype)[0]

        # read the entire header
        num_header_items = 3 + num_blocks
        num_header_bytes = num_bytes_per_item * num_header_items
        num_header_chars = num_bytes_to_num_base64_chars(num_header_bytes)
        byte_string = base64.b64decode(data[:num_header_chars])
        header = numpy.frombuffer(byte_string, dtype)

        # num_blocks = header[0]
        # max_uncompressed_block_size = header[1]
        # last_compressed_block_size = header[2]
        block_sizes = header[3:]

        # Read the block data
        byte_array = base64.b64decode(data[num_header_chars:])
        dtype = vtu_to_numpy_type[data_type]
        num_bytes_per_item = numpy.dtype(dtype).itemsize

        byte_offsets = numpy.empty(block_sizes.shape[0] + 1, dtype=block_sizes.dtype)
        byte_offsets[0] = 0
        numpy.cumsum(block_sizes, out=byte_offsets[1:])

        # process the compressed data
        block_data = numpy.concatenate(
            [
                numpy.frombuffer(
                    zlib.decompress(byte_array[byte_offsets[k] : byte_offsets[k + 1]]),
                    dtype=dtype,
                )
                for k in range(num_blocks)
            ]
        )

        return block_data

    def read_data(self, c):
        fmt = c.attrib["format"] if "format" in c.attrib else "ascii"

        if fmt == "ascii":
            # ascii
            data = numpy.array(
                c.text.split(), dtype=vtu_to_numpy_type[c.attrib["type"]]
            )
        elif fmt == "binary":
            data = self.read_binary(c.text.strip(), c.attrib["type"])
        else:
            # appended data
            assert fmt == "appended", "Unknown data format '{}'.".format(fmt)
            offset = int(c.attrib["offset"])
            data = self.read_binary(self.appended_data[offset:], c.attrib["type"])

        if "NumberOfComponents" in c.attrib:
            data = data.reshape(-1, int(c.attrib["NumberOfComponents"]))
        return data


def read(filename):
    reader = VtuReader(filename)
    return Mesh(
        reader.points,
        reader.cells,
        point_data=reader.point_data,
        cell_data=reader.cell_data,
        field_data=reader.field_data,
    )


def _chunk_it(array, n):
    out = []
    k = 0
    while k * n < len(array):
        out.append(array[k * n : (k + 1) * n])
        k += 1
    return out


def write(filename, mesh, write_binary=True, pretty_xml=True):
    from lxml import etree as ET

    if not write_binary:
        logging.warning("VTU ASCII files are only meant for debugging.")

    if mesh.points.shape[1] == 2:
        logging.warning(
            "VTU requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        mesh.points = numpy.column_stack(
            [mesh.points[:, 0], mesh.points[:, 1], numpy.zeros(mesh.points.shape[0])]
        )

    header_type = "UInt32"

    vtk_file = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="0.1",
        # Use the native endianness. Not strictly necessary, but this
        # simplifies things a bit.
        byte_order=("LittleEndian" if sys.byteorder == "little" else "BigEndian"),
        header_type=header_type,
        compressor="vtkZLibDataCompressor",
    )

    # swap the data to match the system byteorder
    # Don't use byteswap to make sure that the dtype is changed; see
    # <https://github.com/numpy/numpy/issues/10372>.
    points = mesh.points.astype(mesh.points.dtype.newbyteorder("="))
    for data in mesh.point_data.values():
        data = data.astype(data.dtype.newbyteorder("="))
    for data in mesh.cell_data.values():
        for dat in data.values():
            dat = dat.astype(dat.dtype.newbyteorder("="))
    for data in mesh.field_data.values():
        data = data.astype(data.dtype.newbyteorder("="))

    def numpy_to_xml_array(parent, name, fmt, data):
        da = ET.SubElement(
            parent, "DataArray", type=numpy_to_vtu_type[data.dtype], Name=name
        )
        if len(data.shape) == 2:
            da.set("NumberOfComponents", "{}".format(data.shape[1]))
        if write_binary:
            da.set("format", "binary")
            max_block_size = 32768
            data_bytes = data.tostring()
            blocks = _chunk_it(data_bytes, max_block_size)
            num_blocks = len(blocks)
            last_block_size = len(blocks[-1])

            compressed_blocks = [zlib.compress(block) for block in blocks]
            # collect header
            header = numpy.array(
                [num_blocks, max_block_size, last_block_size]
                + [len(b) for b in compressed_blocks],
                dtype=vtu_to_numpy_type[header_type],
            )
            da.text = (
                base64.b64encode(header.tostring())
                + base64.b64encode(b"".join(compressed_blocks))
            ).decode()
        else:
            da.set("format", "ascii")
            s = BytesIO()
            numpy.savetxt(s, data.flatten(), fmt)
            da.text = s.getvalue().decode()
        return

    comment = ET.Comment("This file was created by meshio v{}".format(__version__))
    vtk_file.insert(1, comment)

    grid = ET.SubElement(vtk_file, "UnstructuredGrid")

    total_num_cells = sum([len(c) for c in mesh.cells.values()])
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints="{}".format(len(points)),
        NumberOfCells="{}".format(total_num_cells),
    )

    # points
    if points is not None:
        pts = ET.SubElement(piece, "Points")
        numpy_to_xml_array(pts, "Points", "%.11e", points)

    if mesh.cells is not None:
        cls = ET.SubElement(piece, "Cells")

        # create connectivity, offset, type arrays
        connectivity = numpy.concatenate(
            [numpy.concatenate(v) for v in mesh.cells.values()]
        )
        # offset (points to the first element of the next cell)
        offsets = [
            v.shape[1] * numpy.arange(1, v.shape[0] + 1) for v in mesh.cells.values()
        ]
        for k in range(1, len(offsets)):
            offsets[k] += offsets[k - 1][-1]
        offsets = numpy.concatenate(offsets)
        # types
        types = numpy.concatenate(
            [numpy.full(len(v), meshio_to_vtk_type[k]) for k, v in mesh.cells.items()]
        )

        numpy_to_xml_array(cls, "connectivity", "%d", connectivity)
        numpy_to_xml_array(cls, "offsets", "%d", offsets)
        numpy_to_xml_array(cls, "types", "%d", types)

    if mesh.point_data:
        pd = ET.SubElement(piece, "PointData")
        for name, data in mesh.point_data.items():
            numpy_to_xml_array(pd, name, "%.11e", data)

    if mesh.cell_data:
        cd = ET.SubElement(piece, "CellData")
        for name, data in raw_from_cell_data(mesh.cell_data).items():
            numpy_to_xml_array(cd, name, "%.11e", data)

    write_xml(filename, vtk_file, pretty_xml)
    return
