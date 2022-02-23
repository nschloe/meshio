"""
I/O UNV mesh format
"""
# pylint: disable=too-many-locals
import math
import re
from typing import BinaryIO, Dict, List, TextIO

import numpy as np

from .._common import warn
from .._exceptions import ReadError, WriteError
from .._files import open_file
from .._mesh import CellBlock, Mesh
from .._helpers import register_format


unv_type_to_numnodes = {
    # line
    11: 2,  # rod
    21: 2,  # linear beam
    22: 3,  # tapered beam
    24: 3,  # parabolic beam
    # triangle
    41: 3,  # plane stress linear triangle
    42: 6,  # plane stress parabolic triangle
    91: 3,  # thin shell linear triangle
    92: 6,  # thin shell parabolic triangle
    # quad
    44: 4,  # plane stress linear quadrilateral
    45: 8,  # plane stress parabolic quadrilateral
    94: 4,  # thin shell linear quadrilateral
    95: 8,  # thin shell parabolic quadrilateral
    122: 4,  # rigid element
    # tetrahedron
    111: 4,
    118: 10,
    # wedge
    112: 6,
    # hexahedron
    115: 8,
    116: 20,
}

unv_to_meshio_type = {
    11: "line",
    21: "line",
    22: "line3",
    24: "line3",
    41: "triangle",
    42: "triangle6",
    91: "triangle",
    92: "triangle6",
    44: "quad",
    45: "quad8",
    94: "quad",
    95: "quad8",
    122: "quad",
    111: "tetra",
    118: "tetra10",
    112: "wedge",
    115: "hexahedron",
    116: "hexahedron20",
}

meshio_to_unv_type = {
    "line": 11,
    "line3": 22,
    "triangle": 41,
    "triangle6": 42,
    "quad": 44,
    "quad8": 45,
    "tetra": 111,
    "tetra10": 118,
    "wedge": 112,
    "hexahedron": 115,
    "hexahedron20": 116,
}

# supported UNV tags
_UNV_SEPARATOR = "-1"
_UNV_UNITS = "164"
_UNV_HEADER = "151"
_UNV_POINTS = "2411"
_UNV_CELLS = "2412"
_UNV_GROUPS = ("2452", "2467")
_UNV_DOFS = "757"

unv_supported_tags = (
    _UNV_SEPARATOR,
    _UNV_HEADER,
    _UNV_UNITS,
    _UNV_POINTS,
    _UNV_CELLS,
    *_UNV_GROUPS,
    _UNV_DOFS,
)

_UNV_POINT_GROUP = 7
_UNV_CELL_GROUP = 8

unv_units_codes = {
    1: "SI: Meter (newton)",
    2: "BG: Foot (pound f)",
    3: "MG: Meter (kilogram f)",
    4: "BA: Foot (poundal)",
    5: "MM: mm (milli newton)",
    6: "CM: cm (centi newton)",
    7: "IN: Inch (pound f)",
    8: "GM: mm (kilogram f)",
    9: "US: USER_DEFINED",
    10: "MN: mm (newton)",
}


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class UNVReader:
    """A UNV mesh file reader"""

    def __init__(self, unv_fh: TextIO) -> None:
        self.unv_fh = unv_fh

        # units data
        self.units_code = -1

        # points data
        self.points = np.array([])
        self.unv_point_idx_map: Dict[int, int] = {}

        # cells data
        self.element_type_to_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.unv_element_idx_map: Dict[int, int] = {}

        # groups data
        self.element_id_to_tag_number: Dict[int, int] = {}
        self.point_id_to_tag_number: Dict[int, int] = {}
        self.cell_tags: Dict[int, List] = {}
        self.point_tags: Dict[int, List] = {}

        self._read_buffer()

    @staticmethod
    def _is_end_or_separator(line: str) -> bool:
        """Check for EOF, UNV separator or empty space"""
        return not line or line.strip() == _UNV_SEPARATOR

    @staticmethod
    def is_beam_type(element_type: int) -> bool:
        """Check if the given element type is a beam type (line element)"""
        return element_type in (11, 21, 22, 24)

    def _jump_to_next_tag(self) -> None:
        """Skip current tag and jump to the next one (or EOF)"""
        while True:
            line = self.unv_fh.readline()
            if self._is_end_or_separator(line):
                return

    def _read_units(self) -> None:
        """Read system of units, tag 164"""
        line = self.unv_fh.readline()
        try:
            units_code = int(re.search(r"\d+", line).group())  # type: ignore
        except AttributeError:
            warn("cannot read mesh system of units, skipping...")
            self._jump_to_next_tag()
            return

        while True:
            line = self.unv_fh.readline()

            if self._is_end_or_separator(line):
                break

        if units_code not in unv_units_codes:
            warn("mesh has an unknown system of units, skipping reading tag (164)...")
            return

        self.units_code = units_code

    def _read_points(self) -> None:
        """Read points tag 2411

        Example:
        --------------
        -1
        2411
        121 1 1 11
        5 1 0
        -1
        --------------
        record (121 1 1 11) contains the following data:
            1) 121  -> node label
            2) 1    -> export coordinate system number
            3) 3    -> displacement coordinate system number
            4) 11   -> color

        record (5 1 0)
        contains the 3D cartesian coordinates of point number 121.

        The relevant data in each point entry are node number and 3D coordinates.
        Note that UNV numbering of nodes (points) may not start from 1.
        """
        points = []  # list of parsed 3D coordinates, before reshaping

        # meshio point index
        current_point_id = 0

        while True:
            line = self.unv_fh.readline()

            if self._is_end_or_separator(line):
                break

            point_unv_index = int(line.split()[0])

            # get coordinates
            line = self.unv_fh.readline()
            points.append([float(x) for x in line.split()])

            self.unv_point_idx_map[point_unv_index] = current_point_id
            current_point_id += 1

        if not points:
            raise ReadError("tag 2411 exists, however no points were found in mesh")

        # reshape the points list to a 3 columns numpy matrix,
        # each row represents a 3D coordinate of a point in the mesh.
        self.points = np.array(points).reshape(-1, 3)

    def _read_cells(self) -> None:
        """Read cells tag 2412

        Example:
        ----------------
        -1
        2412
        1 11 1 5380 7 3
        11 18 12
        -1
        -----------------
        record (1 11 1 5380 7 2) contains the following data:
            1) 1    -> element label
            2) 11   -> fe descriptor id (cell type)
            3) 1    -> physical property table number
            4) 5380 -> material property table number
            5) 7    -> color
            6) 2    -> number of nodes on element
        only relevant data in that line is (1), (2) and (6),
        which are element label, type and number of nodes respectively.

        record (11 18 12) contains the UNV points indices that define the cell.
        """
        if not self.unv_point_idx_map:
            raise ReadError("attempting to read cells for mesh with no points")

        # As we did with point indexing, we follow the same for UNV cells indexing.
        unv_elements_indices = []
        element_type_to_data: Dict[str, Dict[str, List]] = {}

        def append_cell_type_points_and_cell_id(
            cell_type: str, points: List, cell_id: int
        ) -> None:
            """Append a cell (array of meshio points) and its meshio cell index
            to our cell-type-points map
            """
            if cell_type not in element_type_to_data:
                # new cell type
                element_type_to_data[cell_type] = {
                    "points": [],
                    "id": [],
                }

            # update existing cell type
            element_type_to_data[cell_type]["points"].append(points)
            element_type_to_data[cell_type]["id"].append(cell_id)

        meshio_element_index = 0

        while True:
            # Unfortunately, there is no way to know beforehand the size of
            # elements in a UNV mesh, so no pre-allocations can be performed,
            # and we have to go line by line and parse elements, until
            # we meet a separator.
            line = self.unv_fh.readline()

            if self._is_end_or_separator(line):
                break

            # read element fields, expecting 6 entries.
            # fields = np.fromstring(line, sep=" ", count=6, dtype=np.int64)
            fields = [int(x) for x in line.split()]
            unv_element_idx, element_type, *_, element_nodes_count = fields

            if element_type not in unv_type_to_numnodes:
                raise ReadError(
                    f"meshio does not support element type ({element_type})"
                )

            if element_nodes_count != unv_type_to_numnodes[element_type]:
                raise ReadError(
                    f"Number of nodes for element {unv_to_meshio_type[element_type]} "
                    f"should be {unv_type_to_numnodes[element_type]}, "
                    f"found {element_nodes_count} instead"
                )

            # get next record
            line = self.unv_fh.readline()

            # beam type elements have an extra irrelevant record
            if self.is_beam_type(element_type):
                # ignore the last record
                line = self.unv_fh.readline()

            current_element_points = [
                self.unv_point_idx_map[int(x)] for x in line.split()
            ]

            if len(current_element_points) != element_nodes_count:
                raise ReadError(
                    f"Number of nodes for element {unv_to_meshio_type[element_type]} "
                    f"should be {unv_type_to_numnodes[element_type]}, "
                    f"found {len(current_element_points)} instead"
                )

            unv_elements_indices.append(unv_element_idx)

            append_cell_type_points_and_cell_id(
                unv_to_meshio_type[element_type],
                current_element_points,
                meshio_element_index,
            )
            meshio_element_index += 1

        if not unv_elements_indices:
            raise ReadError("tag 2412 exists, however no cells were found")

        for meshio_element_type, element_data in element_type_to_data.items():
            unv_type = meshio_to_unv_type[meshio_element_type]
            num_nodes = unv_type_to_numnodes[unv_type]
            self.element_type_to_data[meshio_element_type] = {}
            self.element_type_to_data[meshio_element_type]["points"] = np.array(
                element_data["points"]
            ).reshape(-1, num_nodes)
            self.element_type_to_data[meshio_element_type]["id"] = np.array(
                element_data["id"]
            )

        self.unv_element_idx_map = {
            unv_elements_indices[i]: i
            for i in np.arange(start=0, stop=len(unv_elements_indices), dtype=np.int64)
        }

    def _read_groups(self) -> None:
        """Read groups tags 2452, 2467

        Example:
        -------------------------
        -1
        2467
        1 0 0 0 0 0 0 7
        inlet
        8 28 0 0 8 29 0 0
        8 30 0 0 8 31 0 0
        8 32 0 0 8 33 0 0
        8 34 0 0
        point1
        7 1 0 0
        -1
        -------------------------
        record (1 0 0 0 0 0 0 7) contains two relevant fields,
        the first field is the group index and last field is
        number of vertices defining the group (7 in our case).

        record (inlet) is the name of the group.

        record (8 28 0 0 8 29 0 0) contains three relevant fields,
        specifically at index 0, 1 and index 5 (starting from zero),
        in our case: 8, 28 & 29, which can be interpreted as follows:
        - (8) is the group type (8 = element group) and (7 = point group)
        - (28 & 29) the UNV item (element or point indices) for items in the group.
        """
        # index to keep track of last group index
        current_group_tag = 0

        # maps an element meshio index to its tag (patch) number
        # used later when assigning mesh.cell_data["cell_tags"]
        self.element_id_to_tag_number = {
            i: 0
            for i in np.arange(
                start=0, stop=len(self.unv_element_idx_map), dtype=np.int64
            )
        }

        # maps a point meshio index to its tag (patch) number
        # used later when assigning mesh.point_data["cell_tags"]
        self.point_id_to_tag_number = {
            i: 0
            for i in np.arange(
                start=0, stop=len(self.unv_point_idx_map), dtype=np.int64
            )
        }

        while True:
            line = self.unv_fh.readline()

            if self._is_end_or_separator(line):
                break

            n_elements = int(line.split()[-1])  # number of elements defining the group
            group_name = self.unv_fh.readline().strip()

            group_elements_rows = math.floor(n_elements / 2)
            has_remainder = (n_elements % 2) != 0

            group_elements = np.fromfile(
                file=self.unv_fh, dtype=np.int64, count=group_elements_rows * 8, sep=" "
            ).reshape(group_elements_rows, 8)

            # UNV stores the type of group at index 0, try to get it
            group_type = -1
            if group_elements.shape[0] >= 1:
                group_type = group_elements[0, 0]

            group_elements = np.concatenate(
                [group_elements[:, 1], group_elements[:, 5]]
            )

            if has_remainder:
                unv_group_record = np.fromstring(
                    self.unv_fh.readline(), count=4, sep=" ", dtype=np.int64
                )

                # double check for group type
                # because if we have a single item group,
                # until now we don't know the group type
                if group_type == -1:
                    group_type = unv_group_record[0]
                group_elements = np.concatenate([group_elements, [unv_group_record[1]]])

            if group_type not in (_UNV_POINT_GROUP, _UNV_CELL_GROUP):
                raise ReadError(
                    f"meshio does not support group type {group_type} "
                    f"for group {group_name}"
                )

            # Convert group elements UNV indices to meshio indices
            if group_type == _UNV_CELL_GROUP:
                group_elements = self._unv_index_translate(
                    group_elements, self.unv_element_idx_map
                )
                map_to_update = self.element_id_to_tag_number
            else:
                group_elements = self._unv_index_translate(
                    group_elements, self.unv_point_idx_map
                )
                map_to_update = self.point_id_to_tag_number

            # Assign tag and link group elements to its tag
            current_group_tag += 1

            for element in group_elements:
                map_to_update[element] = current_group_tag

            if group_type == _UNV_CELL_GROUP:
                self.cell_tags[current_group_tag] = [group_name]
            else:
                self.point_tags[current_group_tag] = [group_name]

    def _read_dofs(self) -> None:
        """Read DOF tag 757
        DOF tag is similar to group tag for defining boundaries or patches
        yet it's definition is different.

        DOF tag might be repeated several times, so we provide dofs dict
        as an "out" argument, to update instead of returning a new dict.

        Example:
        ------------------------
        757
        1
        inlet_dof ( 1)
        15 4 1 1 1 1 1 1
        16 4 1 1 1 1 1 1
        21 4 1 1 1 1 1 1
        22 4 1 1 1 1 1 1
        ------------------------

        record (1) is the index of the group.

        record (inlet_dof ( 1)) contains the name of the group.

        record (15 4 1 1 1 1 1 1) has only one relevant field at index 0,
        which is the UNV point indices for points belonging to the group.
        """
        current_point_tag_number: int = 0

        if self.point_tags:
            current_point_tag_number = max(self.point_tags.keys())

        while True:
            line = self.unv_fh.readline()
            if self._is_end_or_separator(line):
                break

            # current line holds patch id, we will ignore and read next line.
            patch_name = self.unv_fh.readline().strip().split()[0]
            current_point_tag_number += 1
            self.point_tags[current_point_tag_number] = [patch_name]

            patch_points = []

            # read patch points
            while True:
                line = self.unv_fh.readline()
                if self._is_end_or_separator(line):
                    break

                patch_points.append(np.fromstring(line, sep=" ", dtype=np.int64)[0])

            patch_points_translated = self._unv_index_translate(
                np.array(patch_points), self.unv_point_idx_map
            )

            for point in patch_points_translated:
                self.point_id_to_tag_number[point] = current_point_tag_number

    @staticmethod
    def _unv_index_translate(
        points_array: np.ndarray, translate_map: Dict[int, int]
    ) -> np.ndarray:
        """Convert UNV index to ordered index"""
        # https://stackoverflow.com/a/43917704/2839539
        translator = np.array(
            [list(translate_map.keys()), list(translate_map.values())]
        )
        mask = np.in1d(points_array, translator[0, :])
        points_array[mask] = translator[
            1, np.searchsorted(translator[0, :], points_array[mask])
        ]

        return points_array

    def _check_empty_mesh(self) -> None:
        """Check empty mesh"""
        if self.points.size == 0 or not self.unv_element_idx_map:
            raise ReadError("either mesh points or cells were not found")

    # pylint: disable=too-many-branches, too-many-statements
    def _read_buffer(self) -> None:
        """Extract relevant data from UNV file"""
        tag_handler_map = {
            _UNV_HEADER: self._jump_to_next_tag,
            _UNV_POINTS: self._read_points,
            _UNV_CELLS: self._read_cells,
            _UNV_UNITS: self._read_units,
            _UNV_GROUPS[0]: self._read_groups,
            _UNV_GROUPS[1]: self._read_groups,
            _UNV_DOFS: self._read_dofs,
        }

        while True:
            line = self.unv_fh.readline()

            if not line:
                break

            tag = line.strip()

            if tag == _UNV_SEPARATOR:
                continue

            if tag not in unv_supported_tags:
                warn(f"meshio does not support tag ({tag}), skipping...")
                self._jump_to_next_tag()
                continue

            tag_handler_map[tag]()

    def create_mesh(self) -> Mesh:
        """Create Mesh object

        Returns:
            Mesh: meshio.Mesh
        """
        # Sanity check
        self._check_empty_mesh()

        # Create cell blocks & assign cell tags
        cell_blocks = []
        cell_data: Dict[str, List] = {"cell_tags": []}

        if not self.element_id_to_tag_number:
            self.element_id_to_tag_number = {
                i: 0
                for i in np.arange(
                    start=0, stop=len(self.unv_element_idx_map), dtype=np.int64
                )
            }

        for cell_type, cells_points_and_ids in self.element_type_to_data.items():
            cell_block = CellBlock(
                cell_type=cell_type, data=cells_points_and_ids["points"]
            )
            cell_blocks.append(cell_block)

            cell_data["cell_tags"].append(
                self._unv_index_translate(
                    cells_points_and_ids["id"], self.element_id_to_tag_number
                )
            )

        mesh = Mesh(points=self.points, cells=cell_blocks, cell_data=cell_data)
        mesh.cell_tags = self.cell_tags

        # Assign point tags
        mesh.point_tags = {}

        if self.point_tags:
            point_data: Dict[str, np.ndarray] = {"point_tags": np.array([])}
            point_data["point_tags"] = np.array(
                list(self.point_id_to_tag_number.values()), dtype=np.int64
            )
            mesh.point_data = point_data
            mesh.point_tags = self.point_tags

        if self.units_code in unv_units_codes:
            mesh.field_data["unit_system"] = unv_units_codes[self.units_code]

        return mesh


def read(filename: str) -> Mesh:
    """Read UNV file, and convert it to meshio Mesh type.

    Args:
        filename (str): UNV file name

    Returns:
        Mesh: meshio representation of UNV mesh
    """
    with open_file(filename, "r") as unv_fh:
        reader = UNVReader(unv_fh)
        mesh = reader.create_mesh()
    return mesh


def write(filename: str, mesh: Mesh) -> None:
    """Write a UNV mesh file from a meshio.Mesh

    Args:
        filename (str): file name
        mesh (Mesh): input mesh
    """
    with open_file(filename, "wb") as unv_fh:
        _write_points(unv_fh, mesh.points)
        _write_cells(unv_fh, mesh.cells)


_SP = "         "


def _write_tag_header(unv_fh: BinaryIO, tag_id: int) -> None:
    """Write a tag header"""
    unv_fh.write(f"    -1\n  {tag_id}\n".encode())


def _write_separator(unv_fh: BinaryIO) -> None:
    """Write a separator"""
    unv_fh.write("    -1\n".encode())


def _write_points(unv_fh: BinaryIO, points: np.ndarray) -> None:
    """Write UNV points tag number 2411

    Args:
        unv_fh (BinaryIO): UNV output file handler
        points (np.ndarray): array of 3D coordinates
    """
    _write_tag_header(unv_fh, 2411)
    is_3d = points.shape[1] == 3

    for i, row in enumerate(points):
        unv_fh.write(_SP.join([str(x) for x in [i + 1, 1, 1, 11]]).encode())
        if is_3d:
            unv_fh.write(
                f"\n   {row[0]:.16E}   {row[1]:.16E}   {row[2]:.16E}\n".encode()
            )
        else:
            unv_fh.write(f"\n   {row[0]:.16E}   {row[1]:.16E}   {0:.16E}\n".encode())
    _write_separator(unv_fh)


def _write_cells(unv_fh: BinaryIO, cell_blocks: List):
    """Write UNV cells tag 2412

    Args:
        unv_fh (BinaryIO): UNV output file handler
        cell_blocks (List): list of meshio CellBlocks

    Raises:
        WriteError: When an unsupported element type is found
    """
    _write_tag_header(unv_fh, 2412)
    i = 0
    for cell_block in cell_blocks:
        if cell_block.type not in meshio_to_unv_type:
            raise WriteError(f"UNV does not support element type {cell_block.type}")

        unv_element_type = meshio_to_unv_type[cell_block.type]

        for cell in cell_block.data:
            n_points = len(cell)
            unv_fh.write(
                _SP.join(
                    [str(x) for x in [i + 1, unv_element_type, 2, 1, 7, n_points]]
                ).encode()
            )
            unv_fh.write("\n".encode())

            i += 1

            if UNVReader.is_beam_type(unv_element_type):
                unv_fh.write(_SP.join([str(x) for x in [0, 1, 1]]).encode())
                unv_fh.write("\n".encode())

            # write cell points indices
            for point in cell:
                unv_fh.write(f"{point + 1}{_SP}".encode())
            unv_fh.write("\n".encode())
    _write_separator(unv_fh)

register_format("su2", [".su2"], read, {"su2": write})
