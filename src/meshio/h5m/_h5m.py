"""
I/O for h5m, cf.
<https://www.mcs.anl.gov/~fathom/moab-docs/html/h5mmain.html>.
"""
from datetime import datetime

import numpy as np

from .. import __about__
from .._common import warn
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

# def _int_to_bool_list(num):
#     # From <https://stackoverflow.com/a/33608387/353337>.
#     bin_string = format(num, '04b')
#     return [x == '1' for x in bin_string[::-1]]


def read(filename):
    import h5py

    f = h5py.File(filename, "r")
    dset = f["tstt"]

    points = dset["nodes"]["coordinates"][()]
    # read point data
    point_data = {}
    if "tags" in dset["nodes"]:
        for name, dataset in dset["nodes"]["tags"].items():
            point_data[name] = dataset[()]

    # # Assert that the GLOBAL_IDs are contiguous.
    # point_gids = dset['nodes']['tags']['GLOBAL_ID'][()]
    # point_start_gid = dset['nodes']['coordinates'].attrs['start_id']
    # point_end_gid = point_start_gid + len(point_gids) - 1
    # assert all(point_gids == range(point_start_gid, point_end_gid + 1))

    h5m_to_meshio_type = {
        "Edge2": "line",
        "Hex8": "hexahedron",
        "Prism6": "wedge",
        "Pyramid5": "pyramid",
        "Quad4": "quad",
        "Tri3": "triangle",
        "Tet4": "tetra",
    }
    cells = []
    cell_data = {}
    for h5m_type, data in dset["elements"].items():
        meshio_type = h5m_to_meshio_type[h5m_type]
        conn = data["connectivity"]
        # Note that the indices are off by 1 in h5m.
        cells.append(CellBlock(meshio_type, conn[()] - 1))

        # TODO bring cell data back
        # if 'tags' in data:
        #     for name, dataset in data['tags'].items():
        #         cell_data[name] = dataset[()]

    # The `sets` in H5M are special in that they represent a segration of data
    # in the current file, particularly by a load balancer (Metis, Zoltan,
    # etc.). This segregation has no equivalent in other data types, but is
    # certainly worthwhile visualizing.
    # Hence, we will translate the sets into cell data with the prefix "set::"
    # here.
    field_data = {}
    # TODO deal with sets
    # if 'sets' in dset and 'contents' in dset['sets']:
    #     # read sets
    #     sets_contents = dset['sets']['contents'][()]
    #     sets_list = dset['sets']['list'][()]
    #     sets_tags = dset['sets']['tags']

    #     cell_start_gid = conn.attrs['start_id']
    #     cell_gids = cell_start_gid + elems['tags']['GLOBAL_ID'][()]
    #     cell_end_gid = cell_start_gid + len(cell_gids) - 1
    #     assert all(cell_gids == range(cell_start_gid, cell_end_gid + 1))

    #     # create the sets
    #     for key, value in sets_tags.items():
    #         mod_key = 'set::' + key
    #         cell_data[mod_key] = np.empty(len(cells), dtype=int)
    #         end = 0
    #         for k, row in enumerate(sets_list):
    #             bits = _int_to_bool_list(row[3])
    #             # is_owner = bits[0]
    #             # is_unique = bits[1]
    #             # is_ordered = bits[2]
    #             is_range_compressed = bits[3]
    #             if is_range_compressed:
    #                 start_gids = sets_contents[end:row[0]+1:2]
    #                 lengths = sets_contents[end+1:row[0]+1:2]
    #                 for start_gid, length in zip(start_gids, lengths):
    #                     end_gid = start_gid + length - 1
    #                     if start_gid >= cell_start_gid and \
    #                             end_gid <= cell_end_gid:
    #                         i0 = start_gid - cell_start_gid
    #                         i1 = end_gid - cell_start_gid + 1
    #                         cell_data[mod_key][i0:i1] = value[k]
    #                     else:
    #                         # TODO deal with point data
    #                         raise RuntimeError('')
    #             else:
    #                 gids = sets_contents[end:row[0]+1]
    #                 cell_data[mod_key][gids - cell_start_gid] = value[k]

    #             end = row[0] + 1

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def write(filename, mesh, add_global_ids=True, compression="gzip", compression_opts=4):
    import h5py

    f = h5py.File(filename, "w")

    tstt = f.create_group("tstt")

    # The base index for h5m is 1.
    global_id = 1

    # add nodes
    nodes = tstt.create_group("nodes")
    coords = nodes.create_dataset(
        "coordinates",
        data=mesh.points,
        compression=compression,
        compression_opts=compression_opts,
    )
    coords.attrs.create("start_id", global_id)
    global_id += len(mesh.points)

    # Global tags
    tstt_tags = tstt.create_group("tags")

    # The GLOBAL_ID associated with a point is used to identify points if
    # distributed across several processes. mbpart automatically adds them,
    # too.
    # Copy to pd to avoid changing point_data. The items are not deep-copied.
    pd = mesh.point_data.copy()
    if "GLOBAL_ID" not in pd and add_global_ids:
        pd["GLOBAL_ID"] = np.arange(1, len(mesh.points) + 1)

    # add point data
    if pd:
        tags = nodes.create_group("tags")
    for key, data in pd.items():
        if len(data.shape) == 1:
            dtype = data.dtype
            tags.create_dataset(
                key,
                data=data,
                compression=compression,
                compression_opts=compression_opts,
            )
        else:
            # H5M doesn't accept n-x-k arrays as data; it wants an n-x-1
            # array with k-tuples as entries.
            n, k = data.shape
            dtype = np.dtype((data.dtype, (k,)))
            dset = tags.create_dataset(
                key,
                (n,),
                dtype=dtype,
                compression=compression,
                compression_opts=compression_opts,
            )
            dset[:] = data

        # Create entry in global tags
        g = tstt_tags.create_group(key)
        g["type"] = dtype
        # Add a class tag:
        # From
        # <https://lists.mcs.anl.gov/pipermail/moab-dev/2015/007104.html>:
        # ```
        # /* Was dense tag data in mesh database */
        #  define mhdf_DENSE_TYPE   2
        # /** \brief Was sparse tag data in mesh database */
        # #define mhdf_SPARSE_TYPE  1
        # /** \brief Was bit-field tag data in mesh database */
        # #define mhdf_BIT_TYPE     0
        # /** \brief Unused */
        # #define mhdf_MESH_TYPE    3
        #
        g.attrs["class"] = 2

    # add elements
    elements = tstt.create_group("elements")

    elem_dt = h5py.special_dtype(
        enum=(
            "i",
            {
                "Edge": 1,
                "Tri": 2,
                "Quad": 3,
                "Polygon": 4,
                "Tet": 5,
                "Pyramid": 6,
                "Prism": 7,
                "Knife": 8,
                "Hex": 9,
                "Polyhedron": 10,
            },
        )
    )

    tstt["elemtypes"] = elem_dt

    tstt.create_dataset(
        "history",
        data=[
            __name__.encode(),
            __about__.__version__.encode(),
            str(datetime.now()).encode(),
        ],
        compression=compression,
        compression_opts=compression_opts,
    )

    # number of nodes to h5m name, element type
    meshio_to_h5m_type = {
        "line": {"name": "Edge2", "type": 1},
        "triangle": {"name": "Tri3", "type": 2},
        "tetra": {"name": "Tet4", "type": 5},
    }
    for cell_block in mesh.cells:
        key = cell_block.type
        data = cell_block.data
        if key not in meshio_to_h5m_type:
            warn("Unsupported H5M element type '%s'. Skipping.", key)
            continue
        this_type = meshio_to_h5m_type[key]
        elem_group = elements.create_group(this_type["name"])
        elem_group.attrs.create("element_type", this_type["type"], dtype=elem_dt)
        # h5m node indices are 1-based
        conn = elem_group.create_dataset(
            "connectivity",
            data=(data + 1),
            compression=compression,
            compression_opts=compression_opts,
        )
        conn.attrs.create("start_id", global_id)
        global_id += len(data)

    # add cell data
    for cell_type, cd in mesh.cell_data.items():
        if cd:
            tags = elem_group.create_group("tags")
            for key, value in cd.items():
                tags.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )

    # add empty set -- MOAB wants this
    sets = tstt.create_group("sets")
    sets.create_group("tags")

    # set max_id
    tstt.attrs.create("max_id", global_id, dtype="u8")


register_format("h5m", [".h5m"], read, {"h5m": write})
