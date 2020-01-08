"""
I/O for VTU.
"""
import base64
import logging
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy

from ..__about__ import __version__
from .._common import (
    meshio_to_vtk_type,
    num_nodes_per_cell,
    raw_from_cell_data,
    vtk_to_meshio_type,
)
from .._exceptions import ReadError
from .._helpers import register
from .._mesh import Mesh


def num_bytes_to_num_base64_chars(num_bytes):
    # Rounding up in integer division works by double negation since Python
    # always rounds down.
    return -(-num_bytes // 3) * 4


def _cells_from_data(connectivity, offsets, types, cell_data_raw):
    # Translate it into the cells dictionary.
    # `connectivity` is a one-dimensional vector with
    # (p0, p1, ... ,pk, p10, p11, ..., p1k, ...
    if len(offsets) != len(types):
        raise ReadError()

    cells = {}
    cell_data = {}
    for vtk_type in numpy.unique(types):
        meshio_type = vtk_to_meshio_type[vtk_type]
        n = num_nodes_per_cell[meshio_type]
        # The offsets point to the _end_ of the indices
        b = types == vtk_type
        indices = numpy.add.outer(offsets[b], numpy.arange(-n, 0, dtype=offsets.dtype))
        cells[meshio_type] = connectivity[indices]
        for name, d in cell_data_raw.items():
            if name not in cell_data:
                cell_data[name] = {}
            cell_data[name][meshio_type] = d[b]

    return cells, cell_data


def _organize_cells(point_offsets, cells, cell_data_raw):
    if len(point_offsets) != len(cells):
        raise ReadError()

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

            for name in cell_data:
                if name not in out_cell_data:
                    out_cell_data[name] = {}
                if key not in out_cell_data[name]:
                    out_cell_data[name][key] = []
                out_cell_data[name][key].append(cell_data[name][key])

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
            if grid is not None:
                raise ReadError("More than one UnstructuredGrid found.")
            grid = c
        else:
            if c.tag != "AppendedData":
                raise ReadError(f"Unknown main tag '{c.tag}'.")
            if appended_data is not None:
                raise ReadError("More than one AppendedData section found.")
            if c.attrib["encoding"] != "base64":
                raise ReadError("")
            appended_data = c.text.strip()
            # The appended data always begins with a (meaningless) underscore.
            if appended_data[0] != "_":
                raise ReadError()
            appended_data = appended_data[1:]

    if grid is None:
        raise ReadError("No UnstructuredGrid found.")
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


class VtuReader:
    """Helper class for reading VTU files. Some properties are global to the file (e.g.,
    byte_order), and instead of passing around these parameters, make them properties of
    this class.
    """

    def __init__(self, filename):  # noqa: C901
        parser = ET.XMLParser()
        tree = ET.parse(filename, parser)
        root = tree.getroot()

        if root.tag != "VTKFile":
            raise ReadError()
        if root.attrib["type"] != "UnstructuredGrid":
            raise ReadError()
        if root.attrib["version"] not in ["0.1", "1.0"]:
            raise ReadError(
                "Unknown VTU file version '{}'.".format(root.attrib["version"])
            )

        if (
            "compressor" in root.attrib
            and root.attrib["compressor"] != "vtkZLibDataCompressor"
        ):
            raise ReadError()

        self.header_type = (
            root.attrib["header_type"] if "header_type" in root.attrib else "UInt32"
        )

        try:
            self.byte_order = root.attrib["byte_order"]
            if self.byte_order not in ["LittleEndian", "BigEndian"]:
                raise ReadError(f"Unknown byte order '{self.byte_order}'.")
        except KeyError:
            self.byte_order = None

        grid, self.appended_data = get_grid(root)

        pieces = []
        field_data = {}
        for c in grid:
            if c.tag == "Piece":
                pieces.append(c)
            elif c.tag == "FieldData":
                # TODO test field data
                for data_array in c:
                    field_data[data_array.attrib["Name"]] = self.read_data(data_array)
            else:
                raise ReadError(f"Unknown grid subtag '{c.tag}'.")

        if not pieces:
            raise ReadError("No Piece found.")

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
                    if len(data_arrays) != 1:
                        raise ReadError()
                    data_array = data_arrays[0]

                    if data_array.tag != "DataArray":
                        raise ReadError()

                    pts = self.read_data(data_array)

                    num_components = int(data_array.attrib["NumberOfComponents"])
                    points.append(pts.reshape(num_points, num_components))

                elif child.tag == "Cells":
                    for data_array in child:
                        if data_array.tag != "DataArray":
                            raise ReadError()
                        piece_cells[data_array.attrib["Name"]] = self.read_data(
                            data_array
                        )

                    if len(piece_cells["offsets"]) != num_cells:
                        raise ReadError()
                    if len(piece_cells["types"]) != num_cells:
                        raise ReadError()

                    cells.append(piece_cells)

                elif child.tag == "PointData":
                    for c in child:
                        if c.tag != "DataArray":
                            raise ReadError()
                        piece_point_data[c.attrib["Name"]] = self.read_data(c)

                    point_data.append(piece_point_data)

                elif child.tag == "CellData":
                    for c in child:
                        if c.tag != "DataArray":
                            raise ReadError()
                        piece_cell_data_raw[c.attrib["Name"]] = self.read_data(c)

                    cell_data_raw.append(piece_cell_data_raw)
                else:
                    raise ReadError(f"Unknown tag '{child.tag}'.")

        if not cell_data_raw:
            cell_data_raw = [{}] * len(cells)

        if len(cell_data_raw) != len(cells):
            raise ReadError()

        point_offsets = (
            numpy.cumsum([pts.shape[0] for pts in points]) - points[0].shape[0]
        )

        # Now merge across pieces
        if not points:
            raise ReadError()
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

    def read_binary(self, data, data_type):
        # first read the the block size; it determines the size of the header
        dtype = vtu_to_numpy_type[self.header_type]
        if self.byte_order is not None:
            dtype = dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
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
        if self.byte_order is not None:
            dtype = dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
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
            data = numpy.fromstring(
                c.text, dtype=vtu_to_numpy_type[c.attrib["type"]], sep=" "
            )
        elif fmt == "binary":
            data = self.read_binary(c.text.strip(), c.attrib["type"])
        elif fmt == "appended":
            offset = int(c.attrib["offset"])
            data = self.read_binary(self.appended_data[offset:], c.attrib["type"])
        else:
            raise ReadError(f"Unknown data format '{fmt}'.")

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
    k = 0
    while k * n < len(array):
        yield array[k * n : (k + 1) * n]
        k += 1


def write(filename, mesh, binary=True):
    # Writing XML with an etree required first transforming the (potentially large)
    # arrays into string, which are much larger in memory still. This makes this writer
    # very memory hungry. See <https://stackoverflow.com/q/59272477/353337>.
    from .._cxml import etree as ET

    if not binary:
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
        # Use the native endianness. Not strictly necessary, but this simplifies things
        # a bit.
        byte_order=("LittleEndian" if sys.byteorder == "little" else "BigEndian"),
        header_type=header_type,
        compressor="vtkZLibDataCompressor",
    )

    # swap the data to match the system byteorder
    # Don't use byteswap to make sure that the dtype is changed; see
    # <https://github.com/numpy/numpy/issues/10372>.
    points = mesh.points.astype(mesh.points.dtype.newbyteorder("="))
    for key, data in mesh.point_data.items():
        mesh.point_data[key] = data.astype(data.dtype.newbyteorder("="))
    for data in mesh.cell_data.values():
        for key, dat in data.items():
            data[key] = dat.astype(dat.dtype.newbyteorder("="))
    for key, data in mesh.field_data.items():
        mesh.field_data[key] = data.astype(data.dtype.newbyteorder("="))

    def numpy_to_xml_array(parent, name, fmt, data):
        da = ET.SubElement(
            parent, "DataArray", type=numpy_to_vtu_type[data.dtype], Name=name
        )
        if len(data.shape) == 2:
            da.set("NumberOfComponents", "{}".format(data.shape[1]))
        if binary:
            da.set("format", "binary")

            def text_writer(f):
                max_block_size = 32768
                data_bytes = data.tostring()

                # round up
                num_blocks = -int(-len(data_bytes) // max_block_size)
                last_block_size = len(data_bytes) - (num_blocks - 1) * max_block_size

                # It's too bad that we have to keep all blocks in memory. This is
                # necessary because the header, written first, needs to know the lengths
                # of all blocks. Also, the blocks are encoded _after_ having been
                # concatenated.
                compressed_blocks = [
                    # This zlib.compress is the slowest part of the writer
                    zlib.compress(block)
                    for block in _chunk_it(data_bytes, max_block_size)
                ]

                # collect header
                header = numpy.array(
                    [num_blocks, max_block_size, last_block_size]
                    + [len(b) for b in compressed_blocks],
                    dtype=vtu_to_numpy_type[header_type],
                )
                f.write(base64.b64encode(header.tostring()).decode())
                f.write(base64.b64encode(b"".join(compressed_blocks)).decode())

        else:
            da.set("format", "ascii")

            def text_writer(f):
                # This write() loop is the bottleneck for the write. Alternatives:
                # savetxt is super slow:
                #   numpy.savetxt(f, data.reshape(-1), fmt=fmt)
                # joining and writing is a bit faster, but consumes huge amounts of
                # memory:
                #   f.write("\n".join(map(fmt.format, data.reshape(-1))))
                for item in data.reshape(-1):
                    f.write((fmt + "\n").format(item))

        da.text_writer = text_writer
        return

    comment = ET.Comment(f"This file was created by meshio v{__version__}")
    vtk_file.insert(1, comment)

    grid = ET.SubElement(vtk_file, "UnstructuredGrid")

    total_num_cells = sum([len(c) for c in mesh.cells.values()])
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints="{}".format(len(points)),
        NumberOfCells=f"{total_num_cells}",
    )

    # points
    if points is not None:
        pts = ET.SubElement(piece, "Points")
        numpy_to_xml_array(pts, "Points", "{:.11e}", points)

    if mesh.cells is not None:
        cls = ET.SubElement(piece, "Cells")

        # create connectivity, offset, type arrays
        connectivity = numpy.concatenate([v.reshape(-1) for v in mesh.cells.values()])

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

        numpy_to_xml_array(cls, "connectivity", "{:d}", connectivity)
        numpy_to_xml_array(cls, "offsets", "{:d}", offsets)
        numpy_to_xml_array(cls, "types", "{:d}", types)

    if mesh.point_data:
        pd = ET.SubElement(piece, "PointData")
        for name, data in mesh.point_data.items():
            numpy_to_xml_array(pd, name, "{:.11e}", data)

    if mesh.cell_data:
        cd = ET.SubElement(piece, "CellData")
        for name, data in raw_from_cell_data(mesh.cell_data).items():
            numpy_to_xml_array(cd, name, "{:.11e}", data)

    # write_xml(filename, vtk_file, pretty_xml)

    tree = ET.ElementTree(vtk_file)
    tree.write(filename)


register(
    "vtu",
    [".vtu"],
    read,
    {
        "vtu": lambda f, m, **kwargs: write(f, m, **kwargs, binary=True),
        "vtu-ascii": lambda f, m, **kwargs: write(f, m, **kwargs, binary=False),
        "vtu-binary": lambda f, m, **kwargs: write(f, m, **kwargs, binary=True),
    },
)
