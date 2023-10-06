"""
I/O SU2 mesh format
<https://su2code.github.io/docs_v7/Mesh-File/>
"""
from itertools import chain, islice

import numpy as np

from .._common import _pick_first_int_data, warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

# follows VTK conventions
su2_type_to_numnodes = {
    3: 2,  # line
    5: 3,  # triangle
    9: 4,  # quad
    10: 4,  # tetra
    12: 8,  # hexahedron
    13: 6,  # wedge
    14: 5,  # pyramid
}
su2_to_meshio_type = {
    3: "line",
    5: "triangle",
    9: "quad",
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}
meshio_to_su2_type = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
    "tetra": 10,
    "hexahedron": 12,
    "wedge": 13,
    "pyramid": 14,
}


def read(filename):

    with open_file(filename, "r") as f:
        mesh = read_buffer(f)
    return mesh


def read_buffer(f):
    cells = []
    cell_data = {"su2:tag": []}

    itype = "i8"
    ftype = "f8"
    dim = 0

    next_tag_id = 0
    expected_nmarkers = 0
    markers_found = 0
    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == "%":
            continue

        try:
            name, rest_of_line = line.split("=")
        except ValueError:
            warn(f"meshio could not parse line\n {line}\n skipping.....")
            continue

        if name == "NDIME":
            dim = int(rest_of_line)
            if dim != 2 and dim != 3:
                raise ReadError(f"Invalid dimension value {line}")

        elif name == "NPOIN":
            # according to documentation rest_of_line should just be a int,
            # and the next block should be just the coordinates of the points
            # However, some file have one or two extra indices not related to the
            # actual coordinates.
            # So lets read the next line to find its actual number of columns
            #
            first_line = f.readline()
            first_line = first_line.split()
            first_line = np.array(first_line, dtype=ftype)

            extra_columns = first_line.shape[0] - dim

            num_verts = int(rest_of_line.split()[0]) - 1
            points = np.fromfile(
                f, count=num_verts * (dim + extra_columns), dtype=ftype, sep=" "
            ).reshape(num_verts, dim + extra_columns)

            # save off any extra info
            if extra_columns > 0:
                first_line = first_line[:-extra_columns]
                points = points[:, :-extra_columns]

            # add the first line we read separately
            points = np.vstack([first_line, points])

        elif name == "NELEM" or name == "MARKER_ELEMS":
            # we cannot? read at once using numpy because we do not know the
            # total size. Read, instead next num_elems as is and re-use the
            # translate_cells function from vtk reader

            num_elems = int(rest_of_line)
            gen = islice(f, num_elems)

            # some files has an extra int column while other not
            # We do not need it so make sure we will skip it
            first_line_str = next(gen)
            first_line = first_line_str.split()
            nnodes = su2_type_to_numnodes[int(first_line[0])]
            has_extra_column = False
            if nnodes + 1 == len(first_line):
                has_extra_column = False
            elif nnodes + 2 == len(first_line):
                has_extra_column = True
            else:
                raise ReadError(f"Invalid number of columns for {name} field")

            # reset generator
            gen = chain([first_line_str], gen)

            cell_array = " ".join([line.rstrip("\n") for line in gen])
            cell_array = np.fromiter(cell_array.split(), dtype=itype)

            cells_, _ = _translate_cells(cell_array, has_extra_column)

            for eltype, data in cells_.items():
                cells.append(CellBlock(eltype, data))
                num_block_elems = len(data)
                if name == "NELEM":
                    cell_data["su2:tag"].append(
                        np.full(num_block_elems, 0, dtype=np.int32)
                    )
                else:
                    tags = np.full(num_block_elems, next_tag_id, dtype=np.int32)
                    cell_data["su2:tag"].append(tags)

        elif name == "NMARK":
            expected_nmarkers = int(rest_of_line)
        elif name == "MARKER_TAG":
            next_tag = rest_of_line
            try:
                next_tag_id = int(next_tag)
            except ValueError:
                next_tag_id += 1
                warn(
                    "meshio does not support tags of string type.\n"
                    f"    Surface tag {rest_of_line} will be replaced by {next_tag_id}"
                )
            markers_found += 1

    if markers_found != expected_nmarkers:
        warn(
            f"expected {expected_nmarkers} markers according to NMARK value "
            f"but found only {markers_found}"
        )

    # merge boundary elements in a single cellblock per cell type
    if dim == 2:
        types = ["line"]
    else:
        types = ["triangle", "quad"]

    indices_to_merge = {}
    for t in types:
        indices_to_merge[t] = []

    for index, cell_block in enumerate(cells):
        if cell_block.type in types:
            indices_to_merge[cell_block.type].append(index)

    cdata = cell_data["su2:tag"]
    for type, indices in indices_to_merge.items():
        if len(indices) > 1:
            cells[indices[0]] = CellBlock(
                type, np.concatenate([cells[i].data for i in indices])
            )
            cdata[indices[0]] = np.concatenate([cdata[i] for i in indices])

    # delete merged blocks
    idelete = []
    for type, indices in indices_to_merge.items():
        idelete += indices[1:]

    for i in sorted(idelete, reverse=True):
        del cells[i]
        del cdata[i]

    cell_data["su2:tag"] = cdata
    return Mesh(points, cells, cell_data=cell_data)


def _translate_cells(data, has_extra_column=False):
    # adapted from _vtk.py
    # Translate input array  into the cells dictionary.
    # `data` is a one-dimensional vector with
    # (vtk cell type, p0, p1, ... ,pk, vtk cell type, p10, p11, ..., p1k, ...

    entry_offset = 1
    if has_extra_column:
        entry_offset += 1

    # Collect types into bins.
    # See <https://stackoverflow.com/q/47310359/353337> for better
    # alternatives.
    types = []
    i = 0
    while i < len(data):
        types.append(data[i])
        i += su2_type_to_numnodes[data[i]] + entry_offset

    types = np.array(types)
    bins = {u: np.where(types == u)[0] for u in np.unique(types)}

    # Deduct offsets from the cell types. This is much faster than manually
    # going through the data array. Slight disadvantage: This doesn't work for
    # cells with a custom number of points.
    numnodes = np.empty(len(types), dtype=int)
    for tpe, idx in bins.items():
        numnodes[idx] = su2_type_to_numnodes[tpe]
    offsets = np.cumsum(numnodes + entry_offset) - (numnodes + entry_offset)

    cells = {}
    cell_data = {}
    for tpe, b in bins.items():
        meshio_type = su2_to_meshio_type[tpe]
        nnodes = su2_type_to_numnodes[tpe]
        indices = np.add.outer(offsets[b], np.arange(1, nnodes + 1))
        cells[meshio_type] = data[indices]

    return cells, cell_data


def write(filename, mesh):

    with open_file(filename, "wb") as f:
        dim = mesh.points.shape[1]
        f.write(f"NDIME= {dim}\n".encode())

        # Write points
        num_points = mesh.points.shape[0]
        f.write(f"NPOIN= {num_points}\n".encode())
        np.savetxt(f, mesh.points)

        # Through warnings about unsupported types
        for cell_block in mesh.cells:
            if cell_block.type not in meshio_to_su2_type:
                warn(
                    ".su2 does not support tags elements of type {}.\n"
                    "Skipping ...".format(type)
                )

        # Write `internal` cells

        types = None
        if dim == 2:
            # `internal` cells are considered to be triangles and quads
            types = ["triangle", "quad"]
        else:
            types = ["tetra", "hexahedron", "wedge", "pyramid"]

        cells = [c for c in mesh.cells if c.type in types]
        total_num_volume_cells = sum(len(c.data) for c in cells)
        f.write(f"NELEM= {total_num_volume_cells}\n".encode())

        for cell_block in cells:
            cell_type = meshio_to_su2_type[cell_block.type]
            # create a column with the value cell_type
            type_column = np.full(
                cell_block.data.shape[0],
                cell_type,
                dtype=cell_block.data.dtype,
            )

            # prepend a column with the value cell_type
            cell_block_to_write = np.column_stack([type_column, cell_block.data])
            np.savetxt(f, cell_block_to_write, fmt="%d")

        # write boundary information

        labels_key, other = _pick_first_int_data(mesh.cell_data)
        if labels_key and other:
            warn(
                "su2 file format can only write one cell data array. "
                "Picking {}, skipping {}.".format(labels_key, ", ".join(other))
            )

        if dim == 2:
            types = ["line"]
        else:
            types = ["triangle", "quad"]

        tags_per_cell_block = dict()

        # We want to separate boundary elements in groups of same tag

        # First, find unique tags and how many elements per tags we have
        for index, cell_block in enumerate(mesh.cells):

            if cell_block.type not in types:
                continue

            labels = (
                mesh.cell_data[labels_key][index]
                if labels_key
                else np.ones(len(cell_block), dtype=cell_block.data.dtype)
            )

            # Get unique tags and number of instances of each tag for this Cell block
            tags_tmp, counts_tmp = np.unique(labels, return_counts=True)

            for tag, count in zip(tags_tmp, counts_tmp):
                if tag not in tags_per_cell_block:
                    tags_per_cell_block[tag] = count
                else:
                    tags_per_cell_block[tag] += count

        f.write(f"NMARK= {len(tags_per_cell_block)}\n".encode())

        # write the blocks you found in previous step
        for tag, count in tags_per_cell_block.items():

            f.write(f"MARKER_TAG= {tag}\n".encode())
            f.write(f"MARKER_ELEMS= {count}\n".encode())

            for index, (cell_type, data) in enumerate(mesh.cells):

                if cell_type not in types:
                    continue

                labels = (
                    mesh.cell_data[labels_key][index]
                    if labels_key
                    else np.ones(len(data), dtype=data.dtype)
                )

                # Pick elements with given tag
                mask = np.where(labels == tag)

                cells_to_write = data[mask]

                cell_type = meshio_to_su2_type[cell_type]

                # create a column with the value cell_type
                type_column = np.full(
                    cells_to_write.shape[0],
                    cell_type,
                    dtype=cells_to_write.dtype,
                )

                # prepend a column with the value cell_type
                cell_block_to_write = np.column_stack([type_column, cells_to_write])
                np.savetxt(f, cell_block_to_write, fmt="%d")

    return


register_format("su2", [".su2"], read, {"su2": write})
