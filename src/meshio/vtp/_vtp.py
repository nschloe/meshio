"""
I/O for VTP.
<https://vtk.org/Wiki/VTK_XML_Formats>
<https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>
"""
import base64
import re
import sys
import zlib

import numpy as np

from ..__about__ import __version__
from .._common import raw_from_cell_data, warn
from .._exceptions import CorruptionError, ReadError, WriteError
from .._helpers import register_format
from .._mesh import CellBlock, Mesh
from .._vtk_common import meshio_to_vtk_order, meshio_to_vtk_type, vtk_cells_from_data

# Paraview 5.8.1's built-in Python doesn't have lzma.
try:
    import lzma
except ModuleNotFoundError:
    lzma = None


def num_bytes_to_num_base64_chars(num_bytes):
    # Rounding up in integer division works by double negation since Python
    # always rounds down.
    return -(-num_bytes // 3) * 4


def _organize_cells(point_offsets, cells, cell_data_raw):
    if len(point_offsets) != len(cells):
        raise ReadError("Inconsistent data!")

    out_cells = []

    for offset, cls, cdr in zip(point_offsets, cells, cell_data_raw):
        cls, cell_data = vtk_cells_from_data(
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
        if c.tag == "PolyData":
            if grid is not None:
                raise ReadError("More than one PolyData found.")
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
        raise ReadError("No PolyData found.")
    return grid, appended_data


def _parse_raw_binary(filename):
    from xml.etree import ElementTree as ET

    with open(filename, "rb") as f:
        raw = f.read()

    try:
        res = re.search(re.compile(b'<AppendedData[^>]+(?:">)'), raw)
        assert res is not None
        i_start = res.end()
        i_stop = raw.find(b"</AppendedData>")
    except Exception:
        raise ReadError()

    header = raw[:i_start].decode()
    footer = raw[i_stop:].decode()
    data = raw[i_start:i_stop].split(b"_", 1)[1].rsplit(b"\n", 1)[0]

    root = ET.fromstring(header + footer)

    dtype = vtp_to_numpy_type[root.get("header_type", "UInt32")]
    if "byte_order" in root.attrib:
        dtype = dtype.newbyteorder(
            "<" if root.get("byte_order") == "LittleEndian" else ">"
        )

    appended_data_tag = root.find("AppendedData")
    assert appended_data_tag is not None
    appended_data_tag.set("encoding", "base64")

    compressor = root.get("compressor")
    if compressor is None:
        arrays = ""
        i = 0
        while i < len(data):
            # The following find() runs into issues if offset is padded with spaces, see
            # <https://github.com/nschloe/meshio/issues/1135>. It works in ParaView.
            # Unfortunately, Python's built-in XML tree can't handle regexes, see
            # <https://stackoverflow.com/a/38810731/353337>.
            da_tag = root.find(f".//DataArray[@offset='{i}']")
            if da_tag is None:
                raise RuntimeError(f"Could not find .//DataArray[@offset='{i}']")
            da_tag.set("offset", str(len(arrays)))

            block_size = int(np.frombuffer(data[i : i + dtype.itemsize], dtype)[0])
            arrays += base64.b64encode(
                data[i : i + block_size + dtype.itemsize]
            ).decode()
            i += block_size + dtype.itemsize

    else:
        c = {"vtkLZMADataCompressor": lzma, "vtkZLibDataCompressor": zlib}[compressor]
        root.attrib.pop("compressor")

        # raise ReadError("Compressed raw binary VTP files not supported.")
        arrays = ""
        i = 0
        while i < len(data):
            da_tag = root.find(f".//DataArray[@offset='{i}']")
            assert da_tag is not None
            da_tag.set("offset", str(len(arrays)))

            num_blocks = int(np.frombuffer(data[i : i + dtype.itemsize], dtype)[0])
            num_header_items = 3 + num_blocks
            num_header_bytes = num_header_items * dtype.itemsize
            header = np.frombuffer(data[i : i + num_header_bytes], dtype)

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

            block_size = np.array([len(block_data)]).astype(dtype).tobytes()
            arrays += base64.b64encode(block_size + block_data).decode()

            i += j + num_header_bytes

    appended_data_tag.text = "_" + arrays
    return root


vtp_to_numpy_type = {
    "Float32": np.dtype(np.float32),
    "Float64": np.dtype(np.float64),
    "Int8": np.dtype(np.int8),
    "Int16": np.dtype(np.int16),
    "Int32": np.dtype(np.int32),
    "Int64": np.dtype(np.int64),
    "UInt8": np.dtype(np.uint8),
    "UInt16": np.dtype(np.uint16),
    "UInt32": np.dtype(np.uint32),
    "UInt64": np.dtype(np.uint64),
    "String": np.dtype(bytes),
}
numpy_to_vtp_type = {v: k for k, v in vtp_to_numpy_type.items()}


def _decompose_strips(piece_strips):
    """
    Decompose triangle strips into triangles
    """
    offset_start = 0
    cells = {"connectivity": [], "offsets": []}
    offset_count = 0
    num_triangles = 0
    tris_per_strip = []
    type_dtype = vtp_to_numpy_type["Int64"]
    for offset in np.nditer(piece_strips["offsets"]):
        npts = offset - offset_start
        p1 = piece_strips["connectivity"][offset_start]
        p2 = piece_strips["connectivity"][offset_start + 1]
        tris_per_strip.append(npts - 2)
        num_triangles += tris_per_strip[-1]
        for i in range(npts - 2):
            p3 = piece_strips["connectivity"][offset_start + i + 2]
            if i % 2:
                cells["connectivity"].extend([p2, p1, p3])
            else:
                cells["connectivity"].extend([p1, p2, p3])

            p1 = p2
            p2 = p3
            offset_count += 3
            cells["offsets"].append(offset_count)
        offset_start = offset
    cells["types"] = np.full(
        num_triangles, meshio_to_vtk_type["triangle"], dtype=type_dtype
    )
    cells["connectivity"] = np.array(
        cells["connectivity"], dtype=piece_strips["connectivity"].dtype
    )
    cells["offsets"] = np.array(cells["offsets"], dtype=piece_strips["offsets"].dtype)

    cells["tris_per_strip"] = np.array(tris_per_strip, dtype=type_dtype)

    return cells


def _copy_cell_data_piece(cells, cell_data_raw, num_strips):
    tris_per_strip = cells.pop("tris_per_strip")
    append_sz = tris_per_strip.sum()
    # for each cell_data array
    for name, values in cell_data_raw.items():
        append_ar = np.empty(append_sz, dtype=values.dtype)
        _start = 0
        # Order of vtkPolyData requires that strips
        # are inserted last in mixed cell type file
        # so expect them at end of cellData
        for i in range(num_strips):
            _stop = _start + tris_per_strip[i]
            append_ar[_start:_stop] = values[-num_strips + i]
            _start = _stop

        cell_data_raw[name] = np.append(values[0:-num_strips], append_ar)
        if cells["offsets"].shape[0] != cell_data_raw[name].shape[0]:
            # should be a conversion error
            raise ReadError()
    return


def _set_polygon_cell_types(piece_polys, num_polys):
    """
    set polygons cell types to triangles, quads, or polygons
    """
    offset_start = 0
    type_dtype = vtp_to_numpy_type["Int64"]
    cell_types = np.zeros(num_polys, dtype=type_dtype)
    for i in range(num_polys):
        d_offset = piece_polys["offsets"][i] - offset_start
        if d_offset == 3:
            cell_types[i] = meshio_to_vtk_type["triangle"]
        elif d_offset == 4:
            cell_types[i] = meshio_to_vtk_type["quad"]
        elif d_offset > 4:
            cell_types[i] = meshio_to_vtk_type["polygon"]
        else:
            warn("Encountered an offset that does not make sense")
        offset_start = piece_polys["offsets"][i]
    piece_polys["types"] = cell_types

    return


class VtpReader:
    """Helper class for reading VTP files. Some properties are global to the file (e.g.,
    byte_order), and instead of passing around these parameters, make them properties of
    this class.
    """

    def __init__(self, filename):  # noqa: C901
        from xml.etree import ElementTree as ET

        parser = ET.XMLParser()
        try:
            tree = ET.parse(str(filename), parser)
            root = tree.getroot()
        except ET.ParseError:
            root = _parse_raw_binary(str(filename))

        if root.tag != "VTKFile":
            raise ReadError(f"Expected tag 'VTKFile', found {root.tag}")
        if root.attrib["type"] != "PolyData":
            tpe = root.attrib["type"]
            raise ReadError(f"Expected type PolyData, found {tpe}")

        if "version" in root.attrib:
            version = root.attrib["version"]
            if version not in ["0.1", "1.0"]:
                raise ReadError(f"Unknown VTP file version '{version}'.")

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
            piece_verts = {}
            piece_lines = {}
            piece_strips = {}
            piece_polys = {}
            piece_point_data = {}
            piece_cell_data_raw = {}

            num_points = int(piece.attrib["NumberOfPoints"])
            num_verts = int(piece.attrib["NumberOfVerts"])
            num_lines = int(piece.attrib["NumberOfLines"])
            num_strips = int(piece.attrib["NumberOfStrips"])
            num_polys = int(piece.attrib["NumberOfPolys"])

            # maybe need something better than hard coding the type here
            type_dtype = vtp_to_numpy_type["Int64"]
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

                elif child.tag == "Verts":
                    for data_array in child:
                        if data_array.tag != "DataArray":
                            raise ReadError()
                        piece_verts[data_array.attrib["Name"]] = self.read_data(
                            data_array
                        )

                    if len(piece_verts["offsets"]) != num_verts:
                        raise ReadError()

                    if num_verts > 0:
                        piece_verts["types"] = np.full(
                            num_verts, meshio_to_vtk_type["vertex"], dtype=type_dtype
                        )

                        for key, values in piece_verts.items():
                            if key in piece_cells:
                                warn(
                                    "Vtp file may be out of order, proceed with caution"
                                )
                                if key == "offsets":
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values + piece_cells[key][-1]
                                    )
                                else:
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values
                                    )
                            else:
                                piece_cells[key] = values

                elif child.tag == "Lines":
                    for data_array in child:
                        if data_array.tag != "DataArray":
                            raise ReadError()
                        piece_lines[data_array.attrib["Name"]] = self.read_data(
                            data_array
                        )

                    if len(piece_lines["offsets"]) != num_lines:
                        raise ReadError()

                    if num_lines > 0:
                        piece_lines["types"] = np.full(
                            num_lines, meshio_to_vtk_type["line"], dtype=type_dtype
                        )

                        for key, values in piece_lines.items():
                            if key in piece_cells:
                                if key == "offsets":
                                    print(piece_cells[key])
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values + piece_cells[key][-1]
                                    )
                                else:
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values
                                    )
                            else:
                                piece_cells[key] = values

                elif child.tag == "Strips":
                    warn(
                        f"Meshio cannot handle (type {child.tag}) but we explicitly convert them to triangles here."
                    )

                    for data_array in child:
                        if data_array.tag != "DataArray":
                            raise ReadError()
                        piece_strips[data_array.attrib["Name"]] = self.read_data(
                            data_array
                        )

                    if len(piece_strips["offsets"]) != num_strips:
                        raise ReadError()

                    if num_strips > 0:
                        # Convert strips to triangles as strips aren't supported by meshio
                        piece_tris = _decompose_strips(piece_strips)

                        for key, values in piece_tris.items():
                            if key in piece_cells:
                                if key == "offsets":
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values + piece_cells[key][-1]
                                    )
                                else:
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values
                                    )
                            else:
                                piece_cells[key] = values

                elif child.tag == "Polys":
                    for data_array in child:
                        if data_array.tag != "DataArray":
                            raise ReadError()
                        piece_polys[data_array.attrib["Name"]] = self.read_data(
                            data_array
                        )

                    if len(piece_polys["offsets"]) != num_polys:
                        raise ReadError()

                    if num_polys > 0:
                        _set_polygon_cell_types(piece_polys, num_polys)

                        for key, values in piece_polys.items():
                            if key in piece_cells:
                                if key == "offsets":
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values + piece_cells[key][-1]
                                    )
                                else:
                                    piece_cells[key] = np.append(
                                        piece_cells[key], values
                                    )
                            else:
                                piece_cells[key] = values

                elif child.tag == "PointData":
                    for c in child:
                        if c.tag != "DataArray":
                            raise ReadError()
                        try:
                            piece_point_data[c.attrib["Name"]] = self.read_data(c)
                        except CorruptionError as e:
                            warn(e.args[0] + " Skipping.")

                    point_data.append(piece_point_data)

                elif child.tag == "CellData":
                    for c in child:
                        if c.tag != "DataArray":
                            raise ReadError()
                        piece_cell_data_raw[c.attrib["Name"]] = self.read_data(c)

                    cell_data_raw.append(piece_cell_data_raw)
                else:
                    raise ReadError(f"Unknown tag '{child.tag}'.")

            if cell_data_raw:
                # have to copy cell data to converted triangles
                # will invalidate CellId numbering
                if num_strips > 0:
                    _copy_cell_data_piece(piece_cells, piece_cell_data_raw, num_strips)

            # only append if not empty
            if piece_cells:
                cells.append(piece_cells)
        if not cell_data_raw:
            cell_data_raw = [{}] * len(cells)

        if len(cell_data_raw) != len(cells):
            raise ReadError()

        point_offsets = np.cumsum([0] + [pts.shape[0] for pts in points][:-1])

        # Now merge across pieces
        if not points:
            raise ReadError()
        self.points = np.concatenate(points)

        if point_data:
            self.point_data = {
                key: np.concatenate([pd[key] for pd in point_data])
                for key in point_data[0]
            }
        else:
            self.point_data = None

        self.cells, self.cell_data = _organize_cells(
            point_offsets, cells, cell_data_raw
        )
        self.field_data = field_data

    def read_uncompressed_binary(self, data, dtype):
        byte_string = base64.b64decode(data)

        # the first item is the total_num_bytes, given in header_dtype
        header_dtype = vtp_to_numpy_type[self.header_type]
        if self.byte_order is not None:
            header_dtype = header_dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
        num_header_bytes = np.dtype(header_dtype).itemsize
        total_num_bytes = np.frombuffer(byte_string[:num_header_bytes], header_dtype)[0]

        # Check if block size was decoded separately
        # (so decoding stopped after block size due to padding)
        if len(byte_string) == num_header_bytes:
            header_len = len(base64.b64encode(byte_string))
            byte_string = base64.b64decode(data[header_len:])
        else:
            byte_string = byte_string[num_header_bytes:]

        # Read the block data; multiple blocks possible here?
        if self.byte_order is not None:
            dtype = dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
        return np.frombuffer(byte_string[:total_num_bytes], dtype=dtype)

    def read_compressed_binary(self, data, dtype):
        # first read the block size; it determines the size of the header
        header_dtype = vtp_to_numpy_type[self.header_type]
        if self.byte_order is not None:
            header_dtype = header_dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )
        num_bytes_per_item = np.dtype(header_dtype).itemsize
        num_chars = num_bytes_to_num_base64_chars(num_bytes_per_item)
        byte_string = base64.b64decode(data[:num_chars])[:num_bytes_per_item]
        num_blocks = np.frombuffer(byte_string, header_dtype)[0]

        # read the entire header
        num_header_items = 3 + int(num_blocks)
        num_header_bytes = num_bytes_per_item * num_header_items
        num_header_chars = num_bytes_to_num_base64_chars(num_header_bytes)
        byte_string = base64.b64decode(data[:num_header_chars])
        header = np.frombuffer(byte_string, header_dtype)

        # num_blocks = header[0]
        # max_uncompressed_block_size = header[1]
        # last_compressed_block_size = header[2]
        block_sizes = header[3:]

        # Read the block data
        byte_array = base64.b64decode(data[num_header_chars:])
        if self.byte_order is not None:
            dtype = dtype.newbyteorder(
                "<" if self.byte_order == "LittleEndian" else ">"
            )

        byte_offsets = np.empty(block_sizes.shape[0] + 1, dtype=block_sizes.dtype)
        byte_offsets[0] = 0
        np.cumsum(block_sizes, out=byte_offsets[1:])

        assert self.compression is not None
        c = {"vtkLZMADataCompressor": lzma, "vtkZLibDataCompressor": zlib}[
            self.compression
        ]

        # process the compressed data
        block_data = np.concatenate(
            [
                np.frombuffer(
                    c.decompress(byte_array[byte_offsets[k] : byte_offsets[k + 1]]),
                    dtype=dtype,
                )
                for k in range(num_blocks)
            ]
        )

        return block_data

    def read_data(self, c):
        fmt = c.attrib["format"] if "format" in c.attrib else "ascii"

        data_type = c.attrib["type"]
        try:
            dtype = vtp_to_numpy_type[data_type]
        except KeyError:
            raise ReadError(f"Illegal data type '{data_type}'.")

        if fmt == "ascii":
            # ascii
            if c.text.strip() == "":
                # https://github.com/numpy/numpy/issues/18435
                data = np.empty((0,), dtype=dtype)
            else:
                data = np.fromstring(c.text, dtype=dtype, sep=" ")
        elif fmt == "binary":
            reader = (
                self.read_uncompressed_binary
                if self.compression is None
                else self.read_compressed_binary
            )
            data = reader(c.text.strip(), dtype)
        elif fmt == "appended":
            offset = int(c.attrib["offset"])
            reader = (
                self.read_uncompressed_binary
                if self.compression is None
                else self.read_compressed_binary
            )
            assert self.appended_data is not None
            data = reader(self.appended_data[offset:], dtype)
        else:
            raise ReadError(f"Unknown data format '{fmt}'.")

        if "NumberOfComponents" in c.attrib:
            nc = int(c.attrib["NumberOfComponents"])
            try:
                data = data.reshape(-1, nc)
            except ValueError:
                name = c.attrib["Name"]
                raise CorruptionError(
                    "VTP file corrupt. "
                    + f"The size of the data array '{name}' is {data.size} "
                    + f"which doesn't fit the number of components {nc}."
                )
        return data


def read(filename):
    reader = VtpReader(filename)
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


def _format_polydata(mesh, type_list):

    # create connectivity, offset
    connectivity = []
    for cell_arr in mesh.cells:
        if cell_arr.type not in type_list:
            continue
        data_arr = cell_arr.data
        new_order = meshio_to_vtk_order(cell_arr.type)
        if new_order is not None:
            data_arr = data_arr[:, new_order]
        connectivity.append(data_arr.flatten())
    connectivity = np.concatenate(connectivity)

    # offset (points to the first element of the next cell)
    offsets = [
        cell_arr.data.shape[1]
        * np.arange(1, cell_arr.data.shape[0] + 1, dtype=connectivity.dtype)
        for cell_arr in mesh.cells
        if cell_arr.type in type_list
    ]
    for k in range(1, len(offsets)):
        offsets[k] += offsets[k - 1][-1]
    offsets = np.concatenate(offsets)

    return connectivity, offsets


def write(filename, mesh, binary=True, compression="zlib", header_type=None):
    # Writing XML with an etree required first transforming the (potentially large)
    # arrays into string, which are much larger in memory still. This makes this writer
    # very memory hungry. See <https://stackoverflow.com/q/59272477/353337>.
    from .._cxml import etree as ET

    for c in mesh.cells:
        if c.type not in ["vertex", "line", "triangle", "quad", "polygon"]:
            raise WriteError(
                "PolyData .vtp files can only contain vertex, line, triangle, quad, and polygon cells."
            )

    if not binary:
        warn("VTP ASCII files are only meant for debugging.")

    if mesh.points.shape[1] == 2:
        warn(
            "VTP requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    vtk_file = ET.Element(
        "VTKFile",
        type="PolyData",
        version="0.1",
        # Use the native endianness. Not strictly necessary, but this simplifies things
        # a bit.
        byte_order=("LittleEndian" if sys.byteorder == "little" else "BigEndian"),
    )

    if header_type is None:
        header_type = "UInt32"
    else:
        vtk_file.set("header_type", header_type)
    assert header_type is not None

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
    points = points.astype(points.dtype.newbyteorder("="), copy=False)
    for k, cell_block in enumerate(mesh.cells):
        cell_type = cell_block.type
        data = cell_block.data
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
        vtp_type = numpy_to_vtp_type[data.dtype]
        fmt = "{:.11e}" if vtp_type.startswith("Float") else "{:d}"
        da = ET.SubElement(parent, "DataArray", type=vtp_type, Name=name)
        if len(data.shape) == 2:
            da.set("NumberOfComponents", f"{data.shape[1]}")

        def text_writer_compressed(f):
            max_block_size = 32768
            data_bytes = data.tobytes()

            # round up
            num_blocks = -int(-len(data_bytes) // max_block_size)
            last_block_size = len(data_bytes) - (num_blocks - 1) * max_block_size

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
            header = np.array(
                [num_blocks, max_block_size, last_block_size]
                + [len(b) for b in compressed_blocks],
                dtype=vtp_to_numpy_type[header_type],
            )
            f.write(base64.b64encode(header.tobytes()).decode())
            f.write(base64.b64encode(b"".join(compressed_blocks)).decode())

        def text_writer_uncompressed(f):
            data_bytes = data.tobytes()
            # collect header
            header = np.array(len(data_bytes), dtype=vtp_to_numpy_type[header_type])
            f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

        def text_writer_ascii(f):
            # This write() loop is the bottleneck for the write. Alternatives:
            # savetxt is super slow:
            #   np.savetxt(f, data.reshape(-1), fmt=fmt)
            # joining and writing is a bit faster, but consumes huge amounts of
            # memory:
            #   f.write("\n".join(map(fmt.format, data.reshape(-1))))
            for item in data.reshape(-1):
                f.write((fmt + "\n").format(item))

        if binary:
            da.set("format", "binary")
            da.text_writer = (
                text_writer_compressed if compression else text_writer_uncompressed
            )
        else:
            da.set("format", "ascii")
            da.text_writer = text_writer_ascii

    comment = ET.Comment(f"This file was created by meshio v{__version__}")
    vtk_file.insert(1, comment)

    grid = ET.SubElement(vtk_file, "PolyData")

    num_verts = 0
    num_lines = 0
    num_strips = 0
    num_polys = 0
    for cell_arr in mesh.cells:
        if cell_arr.type == "vertex":
            num_verts = len(cell_arr.data)
        elif cell_arr.type == "line":
            num_lines = len(cell_arr.data)
        elif cell_arr.type in ["triangle", "quad", "polygon"]:
            num_polys += len(cell_arr.data)

    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=f"{len(points)}",
        NumberOfVerts=f"{num_verts}",
        NumberOfLines=f"{num_lines}",
        NumberOfStrips=f"{num_strips}",
        NumberOfPolys=f"{num_polys}",
    )

    # points
    if points is not None:
        pts = ET.SubElement(piece, "Points")
        numpy_to_xml_array(pts, "Points", points)

    if mesh.cells is not None and len(mesh.cells) > 0:

        poly_types = []
        for cell_arr in mesh.cells:
            if cell_arr.type == "vertex":
                cls = ET.SubElement(piece, "Verts")
                connectivity, offsets = _format_polydata(mesh, ["vertex"])
            elif cell_arr.type == "line":
                cls = ET.SubElement(piece, "Lines")
                connectivity, offsets = _format_polydata(mesh, ["line"])
            elif cell_arr.type in ["triangle", "quad", "polygon"]:
                if cell_arr.type not in poly_types:
                    poly_types.append(cell_arr.type)
                continue
            numpy_to_xml_array(cls, "connectivity", connectivity)
            numpy_to_xml_array(cls, "offsets", offsets)

        if poly_types:
            cls = ET.SubElement(piece, "Polys")
            connectivity, offsets = _format_polydata(mesh, poly_types)
            numpy_to_xml_array(cls, "connectivity", connectivity)
            numpy_to_xml_array(cls, "offsets", offsets)

    if mesh.point_data:
        pd = ET.SubElement(piece, "PointData")
        for name, data in mesh.point_data.items():
            numpy_to_xml_array(pd, name, data)

    if mesh.cell_data:
        cd = ET.SubElement(piece, "CellData")
        for name, data in raw_from_cell_data(mesh.cell_data).items():
            numpy_to_xml_array(cd, name, data)

    if mesh.point_sets or mesh.cell_sets:
        warn(
            "VTP format cannot write sets. "
            + "Consider converting them to \\[point,cell]_data.",
            highlight=False,
        )

    # write_xml(filename, vtk_file, pretty_xml)
    tree = ET.ElementTree(vtk_file)
    tree.write(filename)


register_format("vtp", [".vtp"], read, {"vtp": write})
