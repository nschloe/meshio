# -*- coding: utf-8 -*-
#
'''
I/O for MED/Salome, cf.
<http://docs.salome-platform.org/latest/dev/MEDCoupling/med-file.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy


def read(filename):
    import h5py

    f = h5py.File(filename, 'r')

    # def print_group(grp, indent=0):
    #     for key in grp:
    #         if isinstance(key, int) or \
    #                 isinstance(key, float) or \
    #                 isinstance(key, numpy.int32) or \
    #                 isinstance(key, numpy.int64):
    #             print(indent*' ' + repr(key))
    #         elif isinstance(key, numpy.ndarray):
    #             print(indent*' ' + repr(key))
    #         else:
    #             try:
    #                 print(indent*' ' + str(grp[key]))
    #                 print(indent*' ' + str(list(grp[key].attrs.items())))
    #                 print_group(grp[key], indent=indent+4)
    #             except TypeError:
    #                 print(type(key))
    #                 break
    #     return
    #
    # print_group(f)

    # <HDF5 group "/ENS_MAA" (1 members)>
    ens_maa = f['ENS_MAA']
    meshes = ens_maa.keys()
    assert len(meshes) == 1

    mesh = ens_maa[list(meshes)[0]]

    if 'NOE' not in mesh:
        # One needs NOE (node) and MAI (french maillage, meshing) data. If they
        # are not available in the mesh, check for the submesh.
        submeshes = mesh.keys()
        assert len(submeshes) == 1
        mesh = mesh[list(submeshes)[0]]

    pts_dataset = mesh['NOE']['COO']
    number = pts_dataset.attrs['NBR']
    points = pts_dataset[()].reshape(-1, number).T
    if points.shape[1] == 2:
        points = numpy.column_stack([points, numpy.zeros(len(points))])

    cells = {}

    mai = mesh['MAI']

    if 'PO1' in mai:
        cells['vertex'] = mai['PO1']['NOD'][()].reshape(1, -1).T - 1

    if 'SE2' in mai:
        cells['line'] = mai['SE2']['NOD'][()].reshape(2, -1).T - 1

    if 'TR3' in mai:
        cells['triangle'] = mai['TR3']['NOD'][()].reshape(3, -1).T - 1

    if 'TE4' in mai:
        cells['tetra'] = mai['TE4']['NOD'][()].reshape(4, -1).T - 1

    if 'HE8' in mai:
        cells['hexahedron'] = mai['HE8']['NOD'][()].reshape(8, -1).T - 1

    if 'QU4' in mai:
        cells['quad'] = mai['QU4']['NOD'][()].reshape(4, -1).T - 1

    # Read nodal and cell data if they exist
    try:
        cha = f['CHA']  # champs (fields) in french
        point_data, cell_data, field_data = _read_data(cha)
    except KeyError:
        point_data, cell_data, field_data = {}, {}, {}

    return points, cells, point_data, cell_data, field_data


def _read_data(cha):
    point_data = {}
    cell_data = {}
    field_data = {}
    for name, data in cha.items():
        supps = ['NOE', 'MAI']
        if all(supp not in data for supp in supps):
            submeshes = data.keys()
            assert len(submeshes) == 1
            data = data[list(submeshes)[0]]
        supp = list(data.keys())[0]

        if supp == 'NOE':  # continuous nodal data
            point_data[name] = _read_nodal_data(data)
        else:
            pass

    return point_data, cell_data, field_data


def _read_nodal_data(data):
    nodal_dataset = data['NOE'][data['NOE'].attrs['PFL']]
    number = nodal_dataset.attrs['NBR']
    values = nodal_dataset['CO'][()].reshape(-1, number).T
    if values.shape[1] == 1:  # cut off for 1d arrays
        values = values[:, 0]
    return values


def _read_cell_data(data):
    pass


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        add_global_ids=True
        ):
    import h5py

    point_data = {} if point_data is None else point_data
    cell_data = {} if cell_data is None else cell_data
    field_data = {} if field_data is None else field_data

    f = h5py.File(filename, 'w')

    # Strangely the version must be 3.0.x
    # Any version >= 3.1.0 will NOT work with SALOME 8.3
    info = f.create_group('INFOS_GENERALES')
    info.attrs.create('MAJ', 3)
    info.attrs.create('MIN', 0)
    info.attrs.create('REL', 0)

    # Meshes
    ens_maa = f.create_group('ENS_MAA')
    mesh_name = 'mesh'
    mesh = ens_maa.create_group(mesh_name)
    mesh.attrs.create('DIM', points.shape[1])  # mesh dimension
    mesh.attrs.create('ESP', points.shape[1])  # spatial dimension
    mesh.attrs.create('REP', 0)  # cartesian coordinate system (repère in french)
    mesh.attrs.create('UNT', b'')  # time unit
    mesh.attrs.create('UNI', b'')  # spatial unit
    mesh.attrs.create('SRT', 1)  # sorting type MED_SORT_ITDT
    mesh.attrs.create('NOM', _comp_nom(points.shape[1]).encode('ascii'))  # component names
    mesh.attrs.create('DES', b'Mesh created with meshio')
    mesh.attrs.create('TYP', 0)  # mesh type (MED_NON_STRUCTURE)

    # Time-step
    step = '-0000000000000000001-0000000000000000001'  # NDT NOR
    ts = mesh.create_group(step)
    ts.attrs.create('CGT', 1)
    ts.attrs.create('NDT', -1)  # no time step (-1)
    ts.attrs.create('NOR', -1)  # no iteration step (-1)
    ts.attrs.create('PDT', -1.0)  # current time

    # Points
    noe_group = ts.create_group('NOE')
    noe_group.attrs.create('CGT', 1)
    noe_group.attrs.create('CGS', 1)
    profile = 'MED_NO_PROFILE_INTERNAL'
    noe_group.attrs.create('PFL', profile.encode('ascii'))
    coo = noe_group.create_dataset('COO', data=points.T.flatten())
    coo.attrs.create('CGT', 1)
    coo.attrs.create('NBR', len(points))

    # Elements (mailles in french)
    mai_group = ts.create_group('MAI')
    mai_group.attrs.create('CGT', 1)

    d = {
        'triangle': 'TR3',
        'tetra': 'TE4',
        'hexahedron': 'HE8',
        'quad': 'QU4',
        'vertex': 'PO1',
        'line': 'SE2',
        }

    for key in d:
        if key in cells:
            mai = mai_group.create_group(d[key])
            mai.attrs.create('CGT', 1)
            mai.attrs.create('CGS', 1)
            mai.attrs.create('PFL', profile.encode('ascii'))
            nod = mai.create_dataset('NOD', data=cells[key].T.flatten() + 1)
            nod.attrs.create('CGT', 1)
            nod.attrs.create('NBR', len(cells[key]))

    # Subgroups (familles in french)
    fas = f.create_group('FAS')
    fm = fas.create_group(mesh_name)
    fz = fm.create_group('FAMILLE_ZERO')  # must be defined in any case
    fz.attrs.create('NUM', 0)

    # Write nodal data
    if point_data:
        cha = f.create_group('CHA')
        for name, data in point_data.items():
            # Field
            field = cha.create_group(name)
            field.attrs.create('MAI', mesh_name.encode('ascii'))
            field.attrs.create('TYP', 6)  # MED_FLOAT64
            field.attrs.create('UNI', b'')  # physical unit
            field.attrs.create('UNT', b'')
            nco = 1 if data.ndim == 1 else data.shape[1]
            field.attrs.create('NCO', nco)  # number of components
            field.attrs.create('NOM', _comp_nom(nco).encode('ascii'))  # component names

            # Time-step
            step = '0000000000000000000100000000000000000001'
            ts = field.create_group(step)
            ts.attrs.create('NDT', 1)  # time step 1
            ts.attrs.create('NOR', 1)  # iteration step 1
            ts.attrs.create('PDT', 0.0)  # current time
            ts.attrs.create('RDT', -1)  # NDT of the mesh
            ts.attrs.create('ROR', -1)  # NOR of the mesh

            # Values
            noe = ts.create_group('NOE')
            noe.attrs.create('GAU', b'')  # no associated Gauss points
            noe.attrs.create('PFL', profile.encode('ascii'))
            pfl = noe.create_group(profile)
            pfl.attrs.create('NBR', len(data))  # number of points
            pfl.attrs.create('NGA', 1)  # number of Gauss points (by default 1)
            pfl.attrs.create('GAU', b'')
            pfl.create_dataset('CO', data=data.T.flatten())

    return


def _comp_nom(nco):
    '''
    To be correctly read in a MED viewer, each component must be a
    string of width 16. Since we do not know the physical nature of
    the data, we just use V1, V2, ...
    '''
    return ''.join(['V%-15d' % (i+1) for i in range(nco)])
