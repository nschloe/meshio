"""
I/O for VTU.
<https://vtk.org/Wiki/VTK_XML_Formats>
<https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf>
"""

import base64
import re
import sys
import zlib

import numpy as np

from ..__about__ import __version__
from .._common import info, join_strings, raw_from_cell_data, replace_space, warn
from .._exceptions import CorruptionError, ReadError
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


def _polyhedron_cells_from_data(offsets, faces, faceoffsets, cell_data_raw):
    # In general the number of faces will vary between cells, and the
    # number of nodes vary between faces for each cell. The information
    # will be stored as a List (one item per cell) of lists (one item
    # per face of the cell) of np-arrays of node indices.

    cells = {}
    cell_data = {}

    # The data format for face-cells is:
    # num_faces_cell_0,
    #   num_nodes_face_0, node_ind_0, node_ind_1, ..
    #   num_nodes_face_1, node_ind_0, node_ind_1, ..
    #   ...
    # num_faces_cell_1,
    #   ...
    # See https://vtk.org/Wiki/VTK/Polyhedron_Support for more.

    # The faceoffsets describes the end of the face description for each
    # cell. Switch faceoffsets to give start points, not end points
    faceoffsets = np.append([0], faceoffsets[:-1])

    # Double loop over cells then faces.
    # This will be slow, but seems necessary to cover all cases
    for cell_start in faceoffsets:
        num_faces_this_cell = faces[cell_start]
        faces_this_cell = []
        next_face = cell_start + 1
        for _ in range(num_faces_this_cell):
            num_nodes_this_face = faces[next_face]
            faces_this_cell.append(
                np.array(
                    faces[next_face + 1 : (next_face + num_nodes_this_face + 1)],
                    dtype=int,
                )
            )
            # Increase by number of nodes just read, plus the item giving
            # number of nodes per face
            next_face += num_nodes_this_face + 1

        # Done with this cell
        # Find number of nodes for this cell
        num_nodes_this_cell = np.unique(np.hstack([v for v in faces_this_cell])).size

        key = f"polyhedron{num_nodes_this_cell}"
        if key not in cells.keys():
            cells[key] = []
        cells[key].append(faces_this_cell)

    # The cells will be assigned to blocks according to their number of nodes.
    # This is potentially a reordering, compared to the ordering in faces.
    # Cell data must be reorganized accordingly.

    # Start of the cell-node relations
    start_cn = np.hstack((0, offsets))
    size = np.diff(start_cn)

    # Loop over all cell sizes, find all cells with this size, and store
    # cell data.
    for sz in np.unique(size):
        # Cells with this number of nodes.
        items = np.where(size == sz)[0]

        # Store cell data for this set of cells
        for name, d in cell_data_raw.items():
            if name not in cell_data:
                cell_data[name] = []
            cell_data[name].append(d[items])

    return cells, cell_data


def _organize_cells(point_offsets, cells, cell_data_raw):
    if len(point_offsets) != len(cells):
        raise ReadError("Inconsistent data!")

    out_cells = []

    # IMPLEMENTATION NOTE: The treatment of polyhedral cells is quite a bit different
    # from the other cells; moreover, there are some strong (?) assumptions on such
    # cells. The processing of such cells is therefore moved to a dedicated function for
    # the time being, while all other cell types are treated by the same function.
    # There are still similarities between processing of polyhedral and the rest, so it
    # may be possible to unify the implementations at a later stage.

    # Check if polyhedral cells are present.
    polyhedral_mesh = False
    for c in cells:
        if np.any(c["types"] == 42):  # vtk type 42 is polyhedral
            polyhedral_mesh = True
            break

    if polyhedral_mesh:
        # The current implementation assumes a single set of cells, and cannot mix
        # polyhedral cells with other cell types. It may be possible to do away with
        # these limitations, but for the moment, this is what is available.
        if len(cells) > 1:
            raise ValueError("Implementation assumes single set of cells")
        if np.any(cells[0]["types"] != 42):
            raise ValueError("Cannot handle combinations of polyhedra with other cells")

        # Polyhedra are specified by their faces and faceoffsets; see the function
        # _polyhedron_cells_from_data for more information.
        faces = cells[0]["faces"]
        faceoffsets = cells[0]["faceoffsets"]
        cls, cell_data = _polyhedron_cells_from_data(
            cells[0]["offsets"], faces, faceoffsets, cell_data_raw[0]
        )
        # Organize polyhedra in cell blocks according to the number of nodes per cell.
        for tp, c in cls.items():
            out_cells.append(CellBlock(tp, c))

    else:
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

    dtype = vtu_to_numpy_type[root.get("header_type", "UInt32")]
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

        # raise ReadError("Compressed raw binary VTU files not supported.")
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


vtu_to_numpy_type = {
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
}
numpy_to_vtu_type = {v: k for k, v in vtu_to_numpy_type.items()}


class VtuReader:
    """Helper class for reading VTU files. Some properties are global to the file (e.g.,
    byte_order), and instead of passing around these parameters, make them properties of
    this class.
    """

    def __init__(self, filename):
        self._init_root(filename)
        self._parse_grid()
        self._merge_pieces()

    def _init_root(self, filename):
        """Initialize and validate the XML root element"""
        from xml.etree import ElementTree as ET

        parser = ET.XMLParser()
        try:
            tree = ET.parse(str(filename), parser)
            root = tree.getroot()
        except ET.ParseError:
            root = _parse_raw_binary(str(filename))

        if root.tag != "VTKFile":
            raise ReadError(f"Expected tag 'VTKFile', found {root.tag}")
        if root.attrib["type"] != "UnstructuredGrid":
            tpe = root.attrib["type"]
            raise ReadError(f"Expected type UnstructuredGrid, found {tpe}")

        self._validate_version(root)
        self._fix_empty_components(root)
        self._init_compression(root)
        self.header_type = root.attrib.get("header_type", "UInt32")
        self.byte_order = self._get_byte_order(root)
        
        self.grid, self.appended_data = get_grid(root)

    def _validate_version(self, root):
        if "version" in root.attrib:
            version = root.attrib["version"]
            if version not in ["0.1", "1.0"]:
                raise ReadError(f"Unknown VTU file version '{version}'.")

    def _fix_empty_components(self, root):
        # fix empty NumberOfComponents attributes as produced by Firedrake
        for da_tag in root.findall(".//DataArray[@NumberOfComponents='']"):
            da_tag.attrib.pop("NumberOfComponents")

    def _init_compression(self, root):
        if "compressor" in root.attrib:
            assert root.attrib["compressor"] in [
                "vtkLZMADataCompressor",
                "vtkZLibDataCompressor",
            ]
            self.compression = root.attrib["compressor"]
        else:
            self.compression = None

    def _get_byte_order(self, root):
        try:
            byte_order = root.attrib["byte_order"]
            if byte_order not in ["LittleEndian", "BigEndian"]:
                raise ReadError(f"Unknown byte order '{byte_order}'.")
            return byte_order
        except KeyError:
            return None

    def _parse_grid(self):
        """Parse the grid pieces and field data"""
        self.pieces = []
        self.field_data = {}
        
        for c in self.grid:
            if c.tag == "Piece":
                self.pieces.append(c)
            elif c.tag == "FieldData":
                self._parse_field_data(c)
            else:
                raise ReadError(f"Unknown grid subtag '{c.tag}'.")

        if not self.pieces:
            raise ReadError("No Piece found.")

    def _parse_field_data(self, field_data_elem):
        for data_array in field_data_elem:
            self.field_data[data_array.attrib["Name"]] = self.read_data(data_array)

    def _merge_pieces(self):
        """Merge data from all pieces"""
        points = []
        cells = []
        point_data = []
        cell_data_raw = []

        for piece in self.pieces:
            piece_data = self._process_piece(piece)
            points.append(piece_data["points"])
            cells.append(piece_data["cells"]) 
            point_data.append(piece_data["point_data"])
            cell_data_raw.append(piece_data["cell_data_raw"])

        if not cell_data_raw:
            cell_data_raw = [{}] * len(cells)

        if len(cell_data_raw) != len(cells):
            raise ReadError()

        self._set_merged_data(points, cells, point_data, cell_data_raw)

    def _process_piece(self, piece):
        """Process a single piece element"""
        num_points = int(piece.attrib["NumberOfPoints"])
        num_cells = int(piece.attrib["NumberOfCells"])
        
        piece_data = {
            "points": None,
            "cells": {},
            "point_data": {},
            "cell_data_raw": {}
        }

        for child in piece:
            if child.tag == "Points":
                piece_data["points"] = self._process_points(child, num_points)
            elif child.tag == "Cells":
                piece_data["cells"] = self._process_cells(child, num_cells)
            elif child.tag == "PointData":
                piece_data["point_data"] = self._process_point_data(child)
            elif child.tag == "CellData":
                piece_data["cell_data_raw"] = self._process_cell_data(child)
            else:
                raise ReadError(f"Unknown tag '{child.tag}'.")
                
        return piece_data

    def _set_merged_data(self, points, cells, point_data, cell_data_raw):
        point_offsets = np.cumsum([0] + [pts.shape[0] for pts in points][:-1])

        if not points:
            raise ReadError()
        self.points = np.concatenate(points)

        if point_data[0]:
            self.point_data = {
                key: np.concatenate([pd[key] for pd in point_data])
                for key in point_data[0]
            }
        else:
            self.point_data = None

        self.cells, self.cell_data = _organize_cells(
            point_offsets, cells, cell_data_raw
        )

    def _process_points(self, points_elem, num_points):
        data_arrays = list(points_elem)
        if len(data_arrays) != 1:
            raise ReadError()
        data_array = data_arrays[0]

        if data_array.tag != "DataArray":
            raise ReadError()

        pts = self.read_data(data_array)
        num_components = int(data_array.attrib["NumberOfComponents"])
        return pts.reshape(num_points, num_components)

    def _process_cells(self, cells_elem, num_cells):
        piece_cells = {}
        for data_array in cells_elem:
            if data_array.tag != "DataArray":
                raise ReadError()
            piece_cells[data_array.attrib["Name"]] = self.read_data(data_array)

        if len(piece_cells["offsets"]) != num_cells:
            raise ReadError()
        if len(piece_cells["types"]) != num_cells:
            raise ReadError()

        return piece_cells

    def _process_point_data(self, point_data_elem):
        piece_point_data = {}
        for c in point_data_elem:
            if c.tag != "DataArray":
                raise ReadError()
            try:
                piece_point_data[c.attrib["Name"]] = self.read_data(c)
            except CorruptionError as e:
                warn(e.args[0] + " Skipping.")
        return piece_point_data

    def _process_cell_data(self, cell_data_elem):
        piece_cell_data_raw = {}
        for c in cell_data_elem:
            if c.tag != "DataArray":
                raise ReadError()
            piece_cell_data_raw[c.attrib["Name"]] = self.read_data(c)
        return piece_cell_data_raw

    def read_uncompressed_binary(self, data, dtype):
        byte_string = base64.b64decode(data)

        # the first item is the total_num_bytes, given in header_dtype
        header_dtype = vtu_to_numpy_type[self.header_type]
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
        header_dtype = vtu_to_numpy_type[self.header_type]
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
            dtype = vtu_to_numpy_type[data_type]
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
                    "VTU file corrupt. "
                    + f"The size of the data array '{name}' is {data.size} "
                    + f"which doesn't fit the number of components {nc}."
                )
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


def _make_vtu_file():
    """Create the VTK XML file element."""
    from .._cxml import etree as ET
    vtk_file = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="0.1",
        byte_order=("LittleEndian" if sys.byteorder == "little" else "BigEndian"),
    )
    comment = ET.Comment(f"This file was created by meshio v{__version__}")
    vtk_file.insert(1, comment)
    return vtk_file

def _numpy_to_xml_array(parent, name, data, binary=True, compression=None, header_type=None):
    """Write numpy array to XML element."""
    from .._cxml import etree as ET
    vtu_type = numpy_to_vtu_type[data.dtype]
    fmt = "{:.11e}" if vtu_type.startswith("Float") else "{:d}"
    da = ET.SubElement(parent, "DataArray", type=vtu_type, Name=name)
    if len(data.shape) == 2:
        da.set("NumberOfComponents", f"{data.shape[1]}")

    def text_writer_compressed(f):
        max_block_size = 32768
        data_bytes = data.tobytes()
        num_blocks = -int(-len(data_bytes) // max_block_size)
        last_block_size = len(data_bytes) - (num_blocks - 1) * max_block_size

        c = {"lzma": lzma, "zlib": zlib}[compression]
        compressed_blocks = [
            c.compress(block)
            for block in _chunk_it(data_bytes, max_block_size)
        ]

        header = np.array(
            [num_blocks, max_block_size, last_block_size]
            + [len(b) for b in compressed_blocks],
            dtype=vtu_to_numpy_type[header_type],
        )
        f.write(base64.b64encode(header.tobytes()).decode())
        f.write(base64.b64encode(b"".join(compressed_blocks)).decode())

    def text_writer_uncompressed(f):
        data_bytes = data.tobytes()
        header = np.array(len(data_bytes), dtype=vtu_to_numpy_type[header_type])
        f.write(base64.b64encode(header.tobytes() + data_bytes).decode())

    def text_writer_ascii(f):
        for item in data.reshape(-1):
            f.write((fmt + "\n").format(item))

    if binary:
        da.set("format", "binary")
        da.text_writer = text_writer_compressed if compression else text_writer_uncompressed
    else:
        da.set("format", "ascii")
        da.text_writer = text_writer_ascii

def write(filename, mesh, binary=True, compression="zlib", header_type=None):
    """Write mesh to VTU file."""
    from .._cxml import etree as ET
    
    if not binary:
        warn("VTU ASCII files are only meant for debugging.")

    # Process points
    points = mesh.points
    if points.shape[1] == 2:
        warn("VTU requires 3D points, but 2D points given. Appending 0 third component.")
        points = np.column_stack([points, np.zeros_like(points[:, 0])])

    # Handle sets
    if mesh.point_sets:
        info("VTU format cannot write point_sets. Converting them to point_data...", highlight=False)
        key, _ = join_strings(list(mesh.point_sets.keys()))
        key, _ = replace_space(key)
        mesh.point_sets_to_data(key)

    if mesh.cell_sets:
        info("VTU format cannot write cell_sets. Converting them to cell_data...", highlight=False)
        key, _ = join_strings(list(mesh.cell_sets.keys()))
        key, _ = replace_space(key) 
        mesh.cell_sets_to_data(key)

    # Create VTK file
    vtk_file = _make_vtu_file()
    
    if header_type is not None:
        vtk_file.set("header_type", header_type)
    if binary and compression:
        compressions = {
            "lzma": "vtkLZMADataCompressor",
            "zlib": "vtkZLibDataCompressor",
        }
        assert compression in compressions
        vtk_file.set("compressor", compressions[compression])

    # Create grid
    grid = ET.SubElement(vtk_file, "UnstructuredGrid")
    total_num_cells = sum(len(c.data) for c in mesh.cells)
    piece = ET.SubElement(
        grid,
        "Piece", 
        NumberOfPoints=f"{len(points)}",
        NumberOfCells=f"{total_num_cells}",
    )

    # Write points
    if points is not None:
        pts = ET.SubElement(piece, "Points")
        _numpy_to_xml_array(pts, "Points", points, binary, compression, header_type)

    # Write cells
    if mesh.cells:
        cls = ET.SubElement(piece, "Cells")
        connectivity = np.concatenate([v.data.flatten() for v in mesh.cells])
        offsets = np.concatenate([np.arange(1, len(v.data) + 1) * v.data.shape[1] for v in mesh.cells])
        types = np.concatenate([np.full(len(v), meshio_to_vtk_type[v.type]) for v in mesh.cells])
        
        _numpy_to_xml_array(cls, "connectivity", connectivity, binary, compression, header_type)
        _numpy_to_xml_array(cls, "offsets", offsets, binary, compression, header_type)
        _numpy_to_xml_array(cls, "types", types, binary, compression, header_type)

    # Write data
    if mesh.point_data:
        pd = ET.SubElement(piece, "PointData")
        for name, data in mesh.point_data.items():
            _numpy_to_xml_array(pd, name, data, binary, compression, header_type)

    if mesh.cell_data:
        cd = ET.SubElement(piece, "CellData")
        for name, data in raw_from_cell_data(mesh.cell_data).items():
            _numpy_to_xml_array(cd, name, data, binary, compression, header_type)

    # Write file
    tree = ET.ElementTree(vtk_file)
    tree.write(filename)


register_format("vtu", [".vtu"], read, {"vtu": write})
