"""
I/O for VTU.
<https://vtk.org/Wiki/VTK_XML_Formats>
<https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>
"""
import base64
import logging
import lzma
import re
import sys
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
from .._mesh import CellBlock, Mesh


def num_bytes_to_num_base64_chars(num_bytes):
    # Rounding up in integer division works by double negation since Python
    # always rounds down.
    return -(-num_bytes // 3) * 4


def _cells_from_data(connectivity, offsets, types, cell_data_raw):
    # Translate it into the cells array.
    # `connectivity` is a one-dimensional vector with
    # (p0, p1, ... ,pk, p10, p11, ..., p1k, ...
    if len(offsets) != len(types):
        raise ReadError()

    b = numpy.concatenate(
        [[0], numpy.where(types[:-1] != types[1:])[0] + 1, [len(types)]]
    )

    cells = []
    cell_data = {}
    for start, end in zip(b[:-1], b[1:]):
        try:
            meshio_type = vtk_to_meshio_type[types[start]]
        except KeyError:
            raise ReadError("File contains cells that meshio cannot handle.")
        if meshio_type == "polygon" or meshio_type == "polyhedron":
            # Polygons and polyhedra have unknown and varying number of nodes per cell.

            # Index where the previous block of cells stopped. Needed to know the number
            # of nodes for the first cell in the block.
            if start == 0:
                # This is the very start of the offset array
                first_node = 0
            else:
                # First node is the end of the offset for the previous block
                first_node = offsets[start - 1]

            # Start of the cell-node relation for each cell in this block
            start_cn = numpy.hstack((first_node, offsets[start:end]))
            # Find the size of each cell
            size = numpy.diff(start_cn)

            # Loop over all cell sizes, find all cells with this size, and assign
            # connectivity
            for sz in numpy.unique(size):
                items = numpy.where(size == sz)[0]
                indices = numpy.add.outer(
                    start_cn[items + 1], numpy.arange(-sz, 0, dtype=offsets.dtype)
                )
                cells.append(CellBlock(meshio_type + str(sz), connectivity[indices]))
                # Store cell data for this set of cells
                for name, d in cell_data_raw.items():
                    if name not in cell_data:
                        cell_data[name] = []
                    cell_data[name].append(d[start + items])
        else:
            # Same number of nodes per cell
            n = num_nodes_per_cell[meshio_type]
            indices = numpy.add.outer(
                offsets[start:end], numpy.arange(-n, 0, dtype=offsets.dtype)
            )
            cells.append(CellBlock(meshio_type, connectivity[indices]))
            for name, d in cell_data_raw.items():
                if name not in cell_data:
                    cell_data[name] = []
                cell_data[name].append(d[start:end])

    return cells, cell_data


def _organize_cells(point_offsets, cells, cell_data_raw):
    if len(point_offsets) != len(cells):
        raise ReadError()

    out_cells = []
    for offset, cls, cdr in zip(point_offsets, cells, cell_data_raw):
        cls, cell_data = _cells_from_data(
            cls["connectivity"].ravel(),
            cls["offsets"].ravel(),
            cls["types"].ravel(),
            cdr,
        )
        for c in cls:
            out_cells.append(CellBlock(c.type, c.data + offset))

    return out_cells, cell_data


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
                raise ReadError("Unknown main tag '{}'.".format(c.tag))
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


def _parse_raw_binary(filename):
    import xml.etree.ElementTree as ET

    with open(filename, "rb") as f:
        raw = f.read()

    try:
        i_start = re.search(re.compile(b'<AppendedData[^>]+(?:">)'), raw).end()
        i_stop = raw.find(b"</AppendedData>")
    except Exception:
        raise ReadError()

    header = raw[:i_start].decode()
    footer = raw[i_stop:].decode()
    data = raw[i_start:i_stop].split(b"_", 1)[1].rsplit(b"\n", 1)[0]

    root = ET.fromstring(header + footer)

    dtype = vtu_to_numpy_type[root.get("header_type", "UInt32")]
    if "byte_order" in root.attrib:
        dtype = dtype.newbyteorder(
            "<" if root.get("byte_order") == "LittleEndian" else ">"
        )

    appended_data_tag = root.find("AppendedData")
    appended_data_tag.set("encoding", "base64")

    if "compressor" in root.attrib:
        c = {"vtkLZMADataCompressor": lzma, "vtkZLibDataCompressor": zlib}[
            root.get("compressor")
        ]
        root.attrib.pop("compressor")

        # raise ReadError("Compressed raw binary VTU files not supported.")
        arrays = ""
        i = 0
        while i < len(data):
            da_tag = root.find(".//DataArray[@offset='%d']" % i)
            da_tag.set("offset", "%d" % len(arrays))

            num_blocks = int(numpy.frombuffer(data[i : i + dtype.itemsize], dtype)[0])
            num_header_items = 3 + num_blocks
            num_header_bytes = num_header_items * dtype.itemsize
            header = numpy.frombuffer(data[i : i + num_header_bytes], dtype)

            block_data = b""
            j = 0
            for k in range(num_blocks):
                block_size = int(header[k + 3])
                block_data += c.decompress(
                    data[
                        i + j + num_header_bytes : i + j + block_size + num_header_bytes
                    ]
                )
                j += block_size

            block_size = numpy.array([len(block_data)]).astype(dtype).tobytes()
            arrays += base64.b64encode(block_size + block_data).decode()

            i += j + num_header_bytes

    else:
        arrays = ""
        i = 0
        while i < len(data):
            da_tag = root.find(".//DataArray[@offset='%d']" % i)
            da_tag.set("offset", "%d" % len(arrays))

            block_size = int(numpy.frombuffer(data[i : i + dtype.itemsize], dtype)[0])
            arrays += base64.b64encode(
                data[i : i + block_size + dtype.itemsize]
            ).decode()
            i += block_size + dtype.itemsize

    appended_data_tag.text = "_" + arrays
    return root


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
        import xml.etree.ElementTree as ET

        parser = ET.XMLParser()
        try:
            tree = ET.parse(str(filename), parser)
            root = tree.getroot()
        except ET.ParseError:
            root = _parse_raw_binary(str(filename))

        if root.tag != "VTKFile":
            raise ReadError()
        if root.attrib["type"] != "UnstructuredGrid":
            raise ReadError()
        if root.attrib["version"] not in ["0.1", "1.0"]:
            raise ReadError(
                "Unknown VTU file version '{}'.".format(root.attrib["version"])
            )

        # fix empty NumberOfComponents attributes as produced by Firedrake
        for da_tag in root.findall(".//DataArray[@NumberOfComponents='']"):
            da_tag.attrib.pop("NumberOfComponents")

        if "compressor" in root.attrib:
            assert root.attrib["compressor"] in [
                "vtkLZMADataCompressor",
                "vtkZLibDataCompressor",
            ]
            self.compression = root.attrib["compressor"]
        else:
            self.compression = None

        self.header_type = (
            root.attrib["header_type"] if "header_type" in root.attrib else "UInt32"
        )

        try:
            self.byte_order = root.attrib["byte_order"]
            if self.byte_order not in ["LittleEndian", "BigEndian"]:
                raise ReadError("Unknown byte order '{}'.".format(self.byte_order))
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
                raise ReadError("Unknown grid subtag '{}'.".format(c.tag))

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
                    raise ReadError("Unknown tag '{}'.".format(child.tag))

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

    def read_uncompressed_binary(self, data, data_type):
        byte_string = base64.b64decode(data)

        dtype = vtu_to_numpy_type[self.header_type]
        if self.byte_order is not None:
            dtype = dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
        num_bytes_per_item = numpy.dtype(dtype).itemsize
        total_num_bytes = int(
            numpy.frombuffer(byte_string[:num_bytes_per_item], dtype)[0]
        )

        # Check if block size was decoded separately
        # (so decoding stopped after block size due to padding)
        if len(byte_string) == num_bytes_per_item:
            header_len = len(base64.b64encode(byte_string))
            byte_string = base64.b64decode(data[header_len:])
        else:
            byte_string = byte_string[num_bytes_per_item:]

        # Read the block data; multiple blocks possible here?
        dtype = vtu_to_numpy_type[data_type]
        if self.byte_order is not None:
            dtype = dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
        return numpy.frombuffer(byte_string[:total_num_bytes], dtype=dtype)

    def read_compressed_binary(self, data, data_type):
        # first read the block size; it determines the size of the header
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
        num_header_items = 3 + int(num_blocks)
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

        c = {"vtkLZMADataCompressor": lzma, "vtkZLibDataCompressor": zlib}[
            self.compression
        ]

        # process the compressed data
        block_data = numpy.concatenate(
            [
                numpy.frombuffer(
                    c.decompress(byte_array[byte_offsets[k] : byte_offsets[k + 1]]),
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
            reader = (
                self.read_uncompressed_binary
                if self.compression is None
                else self.read_compressed_binary
            )
            data = reader(c.text.strip(), c.attrib["type"])
        elif fmt == "appended":
            offset = int(c.attrib["offset"])
            reader = (
                self.read_uncompressed_binary
                if self.compression is None
                else self.read_compressed_binary
            )
            data = reader(self.appended_data[offset:], c.attrib["type"])
        else:
            raise ReadError("Unknown data format '{}'.".format(fmt))

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


def write(filename, mesh, binary=True, compression="zlib", header_type=None):
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

    vtk_file = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="0.1",
        # Use the native endianness. Not strictly necessary, but this simplifies things
        # a bit.
        byte_order=("LittleEndian" if sys.byteorder == "little" else "BigEndian"),
    )
    header_type = (
        "UInt32" if header_type is None else vtk_file.set("header_type", header_type)
    )

    if binary and compression:
        # TODO lz4, lzma <https://vtk.org/doc/nightly/html/classvtkDataCompressor.html>
        compressions = {
            "lzma": "vtkLZMADataCompressor",
            "zlib": "vtkZLibDataCompressor",
        }
        assert compression in compressions
        vtk_file.set("compressor", compressions[compression])

    # swap the data to match the system byteorder
    # Don't use byteswap to make sure that the dtype is changed; see
    # <https://github.com/numpy/numpy/issues/10372>.
    points = mesh.points.astype(mesh.points.dtype.newbyteorder("="), copy=False)
    for k, (cell_type, data) in enumerate(mesh.cells):
        mesh.cells[k] = CellBlock(
            cell_type, data.astype(data.dtype.newbyteorder("="), copy=False)
        )
    for key, data in mesh.point_data.items():
        mesh.point_data[key] = data.astype(data.dtype.newbyteorder("="), copy=False)
    for data in mesh.cell_data.values():
        for k, dat in enumerate(data):
            data[k] = dat.astype(dat.dtype.newbyteorder("="), copy=False)
    for key, data in mesh.field_data.items():
        mesh.field_data[key] = data.astype(data.dtype.newbyteorder("="), copy=False)

    def numpy_to_xml_array(parent, name, data):
        vtu_type = numpy_to_vtu_type[data.dtype]
        fmt = "{:.11e}" if vtu_type.startswith("Float") else "{:d}"
        da = ET.SubElement(parent, "DataArray", type=vtu_type, Name=name)
        if len(data.shape) == 2:
            da.set("NumberOfComponents", "{}".format(data.shape[1]))
        if binary:
            da.set("format", "binary")
            if compression:
                # compressed write
                def text_writer(f):
                    max_block_size = 32768
                    data_bytes = data.tobytes()

                    # round up
                    num_blocks = -int(-len(data_bytes) // max_block_size)
                    last_block_size = (
                        len(data_bytes) - (num_blocks - 1) * max_block_size
                    )

                    # It's too bad that we have to keep all blocks in memory. This is
                    # necessary because the header, written first, needs to know the
                    # lengths of all blocks. Also, the blocks are encoded _after_ having
                    # been concatenated.
                    c = {"lzma": lzma, "zlib": zlib}[compression]
                    compressed_blocks = [
                        # This compress is the slowest part of the writer
                        c.compress(block)
                        for block in _chunk_it(data_bytes, max_block_size)
                    ]

                    # collect header
                    header = numpy.array(
                        [num_blocks, max_block_size, last_block_size]
                        + [len(b) for b in compressed_blocks],
                        dtype=vtu_to_numpy_type[header_type],
                    )
                    f.write(base64.b64encode(header.tobytes()).decode())
                    f.write(base64.b64encode(b"".join(compressed_blocks)).decode())

            else:
                # uncompressed write
                def text_writer(f):
                    data_bytes = data.tobytes()
                    # collect header
                    header = numpy.array(
                        len(data_bytes), dtype=vtu_to_numpy_type[header_type]
                    )
                    f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

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

    comment = ET.Comment("This file was created by meshio v{}".format(__version__))
    vtk_file.insert(1, comment)

    grid = ET.SubElement(vtk_file, "UnstructuredGrid")

    total_num_cells = sum([len(c.data) for c in mesh.cells])
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints="{}".format(len(points)),
        NumberOfCells="{}".format(total_num_cells),
    )

    # points
    if points is not None:
        pts = ET.SubElement(piece, "Points")
        numpy_to_xml_array(pts, "Points", points)

    if mesh.cells is not None:
        cls = ET.SubElement(piece, "Cells")

        # create connectivity, offset, type arrays
        connectivity = numpy.concatenate([v.data.reshape(-1) for v in mesh.cells])

        # offset (points to the first element of the next cell)
        offsets = [
            v.data.shape[1]
            * numpy.arange(1, v.data.shape[0] + 1, dtype=connectivity.dtype)
            for v in mesh.cells
        ]
        for k in range(1, len(offsets)):
            offsets[k] += offsets[k - 1][-1]
        offsets = numpy.concatenate(offsets)

        # types
        types_array = []
        for k, v in mesh.cells:
            # For polygon and polyhedron grids, the number of nodes is part of the cell
            # type key. This part must be stripped away.
            if k[:7] == "polygon":
                key_ = k[:7]
            elif k[:10] == "polyhedron":
                key_ = k[:10]
            else:
                # No special treatment
                key_ = k
            types_array.append(numpy.full(len(v), meshio_to_vtk_type[key_]))

        types = numpy.concatenate(
            types_array
            # [numpy.full(len(v), meshio_to_vtk_type[k]) for k, v in mesh.cells]
        )

        numpy_to_xml_array(cls, "connectivity", connectivity)
        numpy_to_xml_array(cls, "offsets", offsets)
        numpy_to_xml_array(cls, "types", types)

    if mesh.point_data:
        pd = ET.SubElement(piece, "PointData")
        for name, data in mesh.point_data.items():
            numpy_to_xml_array(pd, name, data)

    if mesh.cell_data:
        cd = ET.SubElement(piece, "CellData")
        for name, data in raw_from_cell_data(mesh.cell_data).items():
            numpy_to_xml_array(cd, name, data)

    # write_xml(filename, vtk_file, pretty_xml)
    tree = ET.ElementTree(vtk_file)
    tree.write(filename)


register("vtu", [".vtu"], read, {"vtu": write})
