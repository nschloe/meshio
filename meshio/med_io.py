# -*- coding: utf-8 -*-
#
'''
I/O for MED/Salome, cf.
<http://docs.salome-platform.org/latest/dev/MEDCoupling/med-file.html>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
                  Tianyi Li <tianyikillua@gmail.com>
'''
import numpy


meshio_to_med_type = {
    'triangle': 'TR3',
    'tetra': 'TE4',
    'hexahedron': 'HE8',
    'quad': 'QU4',
    'vertex': 'PO1',
    'line': 'SE2',
    }
med_to_meshio_type = {v: k for k, v in meshio_to_med_type.items()}


def read(filename):
    import h5py

    f = h5py.File(filename, 'r')

    # Mesh ensemble
    ens_maa = f['ENS_MAA']
    meshes = ens_maa.keys()
    assert len(meshes) == 1, 'Must only contain exactly 1 mesh'
    mesh = ens_maa[list(meshes)[0]]

    # Possible time-stepping
    if 'NOE' not in mesh:
        # One needs NOE (node) and MAI (french maillage, meshing) data. If they
        # are not available in the mesh, check for time-steppings.
        ts = mesh.keys()
        assert len(ts) == 1, 'Must only contain exactly 1 time-step'
        mesh = mesh[list(ts)[0]]

    # Points
    pts_dataset = mesh['NOE']['COO']
    number = pts_dataset.attrs['NBR']
    points = pts_dataset[()].reshape(-1, number).T
    if points.shape[1] == 2:
        points = numpy.column_stack([points, numpy.zeros(len(points))])

    # Cells
    cells = {}
    mai = mesh['MAI']
    for key, med_type in meshio_to_med_type.items():
        if med_type in mai:
            nn = int(med_type[-1])
            cells[key] = mai[med_type]['NOD'][()].reshape(nn, -1).T - 1

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
        ts = data.keys()
        data = data[list(ts)[-1]]  # only read the last time-step

        # MED field can contain multiple types of data
        for supp in data:
            if supp == 'NOE':  # continuous nodal (NOEU) data
                point_data[name] = _read_nodal_data(data)
            else:  # Gauss points (ELGA) or DG (ELNO) data
                cell_data = _read_cell_data(cell_data, name, supp, data)

    return point_data, cell_data, field_data


def _read_nodal_data(data):
    nodal_dataset = data['NOE'][data['NOE'].attrs['PFL']]
    nbr = nodal_dataset.attrs['NBR']
    values = nodal_dataset['CO'][()].reshape(-1, nbr).T
    if values.shape[1] == 1:  # cut off for scalars
        values = values[:, 0]
    return values


def _read_cell_data(cell_data, name, supp, data):
    med_type = supp.partition('.')[2]
    cell_dataset = data[supp][data[supp].attrs['PFL']]
    nbr = cell_dataset.attrs['NBR']  # number of cells
    nga = cell_dataset.attrs['NGA']  # number of Gauss points

    # Only 1 Gauss/elemental nodal point per cell
    if nga == 1:
        values = cell_dataset['CO'][()].reshape(-1, nbr).T
        if values.shape[1] == 1:  # cut off for scalars
            values = values[:, 0]
    # Multiple Gauss/elemental nodal points per cell
    # In general at each cell the value shape will be (nco, nga)
    else:
        values = cell_dataset['CO'][()].reshape(-1, nbr, nga)
        values = numpy.swapaxes(values, 0, 1)

    try:  # cell type already exists
        key = med_to_meshio_type[med_type]
        cell_data[key][name] = values
    except KeyError:
        cell_data[key] = {name: values}
    return cell_data


def write(filename,
          points,
          cells,
          point_data=None,
          cell_data=None,
          field_data=None,
          add_global_ids=True):
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
    pfl = 'MED_NO_PROFILE_INTERNAL'
    noe_group.attrs.create('PFL', pfl.encode('ascii'))
    coo = noe_group.create_dataset('COO', data=points.T.flatten())
    coo.attrs.create('CGT', 1)
    coo.attrs.create('NBR', len(points))

    # Cells (mailles in french)
    mai_group = ts.create_group('MAI')
    mai_group.attrs.create('CGT', 1)
    for key, med_type in meshio_to_med_type.items():
        if key in cells:
            mai = mai_group.create_group(med_type)
            mai.attrs.create('CGT', 1)
            mai.attrs.create('CGS', 1)
            mai.attrs.create('PFL', pfl.encode('ascii'))
            nod = mai.create_dataset('NOD', data=cells[key].T.flatten() + 1)
            nod.attrs.create('CGT', 1)
            nod.attrs.create('NBR', len(cells[key]))

    # Subgroups (familles in french)
    fas = f.create_group('FAS')
    fm = fas.create_group(mesh_name)
    fz = fm.create_group('FAMILLE_ZERO')  # must be defined in any case
    fz.attrs.create('NUM', 0)

    # Write nodal/cell data
    if point_data or cell_data:
        cha = f.create_group('CHA')

        # Nodal data
        for name, data in point_data.items():
            supp = 'NOEU'  # nodal data
            _write_data(cha, mesh_name, pfl, name, supp, data)

        # Cell data
        # Only support writing ELEM fields with only 1 Gauss point per cell
        # Or ELNO (DG) fields defined at every node per cell
        for cell_type, d in cell_data.items():
            for name, data in d.items():

                # Determine the nature of the cell data
                # Either data.shape = (nbr, ) or (nbr, nco) -> ELEM
                # or data.shape = (nbr, nco, nga) -> ELNO or ELGA
                med_type = meshio_to_med_type[cell_type]
                nn = int(med_type[-1])  # number of nodes per cell
                if data.ndim <= 2:
                    supp = 'ELEM'
                elif data.shape[2] == nn:
                    supp = 'ELNO'
                else:  # general ELGA data defined at unknown Gauss points
                    supp = 'ELGA'
                _write_data(cha, mesh_name, pfl, name, supp, data, med_type)

    return


def _write_data(cha, mesh_name, pfl, name, supp, data, med_type=None):
    # Skip for general ELGA fields defined at unknown Gauss points
    if supp == 'ELGA':
        return

    # Field
    try:  # a same MED field may contain fields of different natures
        field = cha.create_group(name)
        field.attrs.create('MAI', mesh_name.encode('ascii'))
        field.attrs.create('TYP', 6)  # MED_FLOAT64
        field.attrs.create('UNI', b'')  # physical unit
        field.attrs.create('UNT', b'')  # time unit
        nco = 1 if data.ndim == 1 else data.shape[1]
        field.attrs.create('NCO', nco)  # number of components
        field.attrs.create('NOM', _comp_nom(nco).encode('ascii'))

        # Time-step
        step = '0000000000000000000100000000000000000001'
        ts = field.create_group(step)
        ts.attrs.create('NDT', 1)  # time step 1
        ts.attrs.create('NOR', 1)  # iteration step 1
        ts.attrs.create('PDT', 0.0)  # current time
        ts.attrs.create('RDT', -1)  # NDT of the mesh
        ts.attrs.create('ROR', -1)  # NOR of the mesh

    except ValueError:  # name already exists
        field = cha[name]
        ts_name = list(field.keys())[-1]
        ts = field[ts_name]

    # Field information
    if supp == 'NOEU':
        typ = ts.create_group('NOE')
    elif supp == 'ELNO':
        typ = ts.create_group('NOE.' + med_type)
    else:  # 'ELEM' with only 1 Gauss points!
        typ = ts.create_group('MAI.' + med_type)

    typ.attrs.create('GAU', b'')  # no associated Gauss points
    typ.attrs.create('PFL', pfl.encode('ascii'))
    pfl = typ.create_group(pfl)
    pfl.attrs.create('NBR', len(data))  # number of points
    if supp == 'ELNO':
        pfl.attrs.create('NGA', data.shape[2])
    else:
        pfl.attrs.create('NGA', 1)
    pfl.attrs.create('GAU', b'')

    # Data
    if supp == 'NOEU' or supp == 'ELEM':
        pfl.create_dataset('CO', data=data.T.flatten())
    else:  # ELNO fields
        data = numpy.swapaxes(data, 0, 1)
        pfl.create_dataset('CO', data=data.flatten())


def _comp_nom(nco):
    '''
    To be correctly read in a MED viewer, each component must be a
    string of width 16. Since we do not know the physical nature of
    the data, we just use V1, V2, ...
    '''
    return ''.join(['V%-15d' % (i+1) for i in range(nco)])
