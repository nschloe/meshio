# -*- coding: utf-8 -*-
#
'''
I/O for MED/Salome, cf.
<http://docs.salome-platform.org/latest/dev/MEDCoupling/med-file.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from datetime import datetime
import logging
import numpy

from . import __about__


def read(filename):
    import h5py

    f = h5py.File(filename, 'r')

    # def print_group(grp, indent=0):
    #     for key in grp:
    #         if isinstance(key, int) or isinstance(key, float) or \
    #                 isinstance(key, numpy.int32):
    #             print(indent*' ' + repr(key))
    #         elif isinstance(key, numpy.ndarray):
    #             print(indent*' ' + repr(key))
    #         else:
    #             try:
    #                 print(indent*' ' + str(grp[key]))
    #                 print_group(grp[key], indent=indent+4)
    #             except TypeError:
    #                 print(type(key))
    #                 break
    #     return

    # print_group(f)

    # <HDF5 group "/ENS_MAA" (1 members)>
    ens_maa = f['ENS_MAA']
    meshes = ens_maa.keys()
    assert len(meshes) == 1

    mesh = ens_maa[list(meshes)[0]]

    submeshes = mesh.keys()
    assert len(submeshes) == 1
    submesh = mesh[list(submeshes)[0]]

    pts_dataset = submesh['NOE']['COO']
    points = pts_dataset[()].reshape(3, -1).T

    cells = {}

    mai = submesh['MAI']
    if 'TR3' in mai:
        cells['triangle'] = mai['TR3']['NOD'][()].reshape(3, -1).T - 1

    if 'TE4' in mai:
        cells['tetra'] = mai['TE4']['NOD'][()].reshape(4, -1).T - 1

    return points, cells, {}, {}, {}


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        add_global_ids=True
        ):
    '''Writes H5M files, cf.
    https://trac.mcs.anl.gov/projects/ITAPS/wiki/MOAB/h5m.
    '''
    import h5py

    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    f = h5py.File(filename, 'w')

    tstt = f.create_group('tstt')

    # The base index for h5m is 1.
    global_id = 1

    # add nodes
    nodes = tstt.create_group('nodes')
    coords = nodes.create_dataset('coordinates', data=points)
    coords.attrs.create('start_id', global_id)
    global_id += len(points)

    # Global tags
    tstt_tags = tstt.create_group('tags')

    # The GLOBAL_ID associated with a point is used to identify points if
    # distributed across several processes. mbpart automatically adds them,
    # too.
    if 'GLOBAL_ID' not in point_data and add_global_ids:
        point_data['GLOBAL_ID'] = numpy.arange(1, len(points)+1, )

    # add point data
    if point_data is not None:
        tags = nodes.create_group('tags')
        for key, data in point_data.items():
            if len(data.shape) == 1:
                dtype = data.dtype
                tags.create_dataset(key, data=data)
            else:
                # H5M doesn't accept n-x-k arrays as data; it wants an n-x-1
                # array with k-tuples as entries.
                n, k = data.shape
                dtype = numpy.dtype((data.dtype, (k,)))
                dset = tags.create_dataset(key, (n,), dtype=dtype)
                dset[:] = data

            # Create entry in global tags
            g = tstt_tags.create_group(key)
            g['type'] = dtype
            # Add a class tag:
            # From
            # <http://lists.mcs.anl.gov/pipermail/moab-dev/2015/007104.html>:
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
            g.attrs['class'] = 2

    # add elements
    elements = tstt.create_group('elements')

    elem_dt = h5py.special_dtype(
        enum=('i', {
            'Edge': 1,
            'Tri': 2,
            'Quad': 3,
            'Polygon': 4,
            'Tet': 5,
            'Pyramid': 6,
            'Prism': 7,
            'Knife': 8,
            'Hex': 9,
            'Polyhedron': 10
            })
        )

    tstt['elemtypes'] = elem_dt

    tstt.create_dataset(
        'history',
        data=[
         __name__.encode('utf-8'),
         __about__.__version__.encode('utf-8'),
         str(datetime.now()).encode('utf-8')
        ]
        )

    # number of nodes to h5m name, element type
    meshio_to_h5m_type = {
            'line': {'name': 'Edge2', 'type': 1},
            'triangle': {'name': 'Tri3', 'type': 2},
            'tetra': {'name': 'Tet4', 'type': 5}
            }
    for key, data in cells.items():
        if key not in meshio_to_h5m_type:
            logging.warning(
                    'Unsupported H5M element type \'%s\'. Skipping.', key
                    )
            continue
        this_type = meshio_to_h5m_type[key]
        elem_group = elements.create_group(this_type['name'])
        elem_group.attrs.create(
                'element_type',
                this_type['type'],
                dtype=elem_dt
                )
        # h5m node indices are 1-based
        conn = elem_group.create_dataset('connectivity', data=(data + 1))
        conn.attrs.create('start_id', global_id)
        global_id += len(data)

    # add cell data
    if cell_data:
        tags = elem_group.create_group('tags')
        for key, value in cell_data.items():
            tags.create_dataset(key, data=value)

    # add empty set -- MOAB wants this
    sets = tstt.create_group('sets')
    sets.create_group('tags')

    # set max_id
    tstt.attrs.create('max_id', global_id, dtype='u8')

    return
