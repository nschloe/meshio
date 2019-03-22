# -*- coding: utf-8 -*-
#
"""
I/O for MED/Salome, cf.
<https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html>.
"""
import numpy

from .common import num_nodes_per_cell
from .mesh import Mesh

# https://bitbucket.org/code_aster/codeaster-src/src/default/catalo/cataelem/Commons/mesh_types.py
meshio_to_med_type = {
    "vertex": "PO1",
    "line": "SE2",
    "line3": "SE3",
    "triangle": "TR3",
    "triangle6": "TR6",
    "quad": "QU4",
    "quad8": "QU8",
    "tetra": "TE4",
    "tetra10": "T10",
    "pyramid": "PY5",
    "pyramid13": "P13",
    "hexahedron": "HE8",
}
med_to_meshio_type = {v: k for k, v in meshio_to_med_type.items()}


def read(filename):
    import h5py

    f = h5py.File(filename, "r")

    # Mesh ensemble
    ens_maa = f["ENS_MAA"]
    meshes = ens_maa.keys()
    assert len(meshes) == 1, "Must only contain exactly 1 mesh"
    mesh_name = list(meshes)[0]
    mesh = ens_maa[mesh_name]

    # Possible time-stepping
    if "NOE" not in mesh:
        # One needs NOE (node) and MAI (french maillage, meshing) data. If they
        # are not available in the mesh, check for time-steppings.
        ts = mesh.keys()
        assert len(ts) == 1, "Must only contain exactly 1 time-step"
        mesh = mesh[list(ts)[0]]

    # Read nodal and cell data if they exist
    try:
        cha = f["CHA"]  # champs (fields) in french
    except KeyError:
        point_data, cell_data, field_data = {}, {}, {}
    else:
        point_data, cell_data, field_data = _read_data(cha)

    # Points
    pts_dataset = mesh["NOE"]["COO"]
    number = pts_dataset.attrs["NBR"]
    points = pts_dataset[()].reshape(-1, number).T

    # Point tags
    if "FAM" in mesh["NOE"]:
        tags = mesh["NOE"]["FAM"][()]
        point_data["point_tags"] = tags  # replacing previous "point_tags"

    # Information for point tags
    point_tags = {}
    fas = f["FAS"][mesh_name]
    if "NOEUD" in fas:
        point_tags = _read_families(fas["NOEUD"])

    # Cells
    cells = {}
    mai = mesh["MAI"]
    for med_cell_type, med_cell_type_group in mai.items():
        cell_type = med_to_meshio_type[med_cell_type]
        cells[cell_type] = (
            med_cell_type_group["NOD"][()].reshape(num_nodes_per_cell[cell_type], -1).T
            - 1
        )

        # Cell tags
        if "FAM" in med_cell_type_group:
            tags = med_cell_type_group["FAM"][()]
            if cell_type not in cell_data:
                cell_data[cell_type] = {}
            cell_data[cell_type]["cell_tags"] = tags  # replacing previous "cell_tags"

    # Information for cell tags
    cell_tags = {}
    if "ELEME" in fas:
        cell_tags = _read_families(fas["ELEME"])

    # Construct the mesh object
    mesh = Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )
    mesh.point_tags = point_tags
    mesh.cell_tags = cell_tags
    return mesh


def _read_data(cha):
    point_data = {}
    cell_data = {}
    field_data = {}
    for name, data in cha.items():
        ts = sorted(data.keys())  # associated time-steps
        if len(ts) == 1:  # single time-step
            names = [name]  # do not change field name
        else:  # many time-steps
            names = [None] * len(ts)
            for i, key in enumerate(ts):
                t = data[key].attrs["PDT"]  # current time
                names[i] = name + "[{:d}] - {:g}".format(i, t)

        # MED field can contain multiple types of data
        for i, key in enumerate(ts):
            datum = data[key]  # at a particular time step
            name = names[i]
            for supp in datum:
                if supp == "NOE":  # continuous nodal (NOEU) data
                    point_data[name] = _read_nodal_data(datum)
                else:  # Gauss points (ELGA) or DG (ELNO) data
                    cell_data = _read_cell_data(cell_data, name, supp, datum)

    return point_data, cell_data, field_data


def _read_nodal_data(data):
    nodal_dataset = data["NOE"][data["NOE"].attrs["PFL"]]
    nbr = nodal_dataset.attrs["NBR"]
    values = nodal_dataset["CO"][()].reshape(-1, nbr).T
    if values.shape[1] == 1:  # cut off for scalars
        values = values[:, 0]
    return values


def _read_cell_data(cell_data, name, supp, data):
    med_type = supp.partition(".")[2]
    cell_dataset = data[supp][data[supp].attrs["PFL"]]
    nbr = cell_dataset.attrs["NBR"]  # number of cells
    nga = cell_dataset.attrs["NGA"]  # number of Gauss points

    # Only 1 Gauss/elemental nodal point per cell
    if nga == 1:
        values = cell_dataset["CO"][()].reshape(-1, nbr).T
        if values.shape[1] == 1:  # cut off for scalars
            values = values[:, 0]
    # Multiple Gauss/elemental nodal points per cell
    # In general at each cell the value shape will be (nco, nga)
    else:
        values = cell_dataset["CO"][()].reshape(-1, nbr, nga)
        values = numpy.swapaxes(values, 0, 1)

    try:  # cell type already exists
        key = med_to_meshio_type[med_type]
        cell_data[key][name] = values
    except KeyError:
        cell_data[key] = {name: values}
    return cell_data


def _read_families(fas_data):
    families = {}
    for _, node_set in fas_data.items():
        num = node_set.attrs["NUM"]  # unique set id
        nbr = node_set["GRO"].attrs["NBR"]  # number of subsets
        nom_dataset = node_set["GRO"]["NOM"][()]  # (nbr, 80) of int8
        nom = [None] * nbr
        for i in range(nbr):
            nom[i] = "".join([chr(x) for x in nom_dataset[i] if x != 0])
        families[num] = nom
    return families


def write(filename, mesh, add_global_ids=True):
    import h5py

    f = h5py.File(filename, "w")

    # Strangely the version must be 3.0.x
    # Any version >= 3.1.0 will NOT work with SALOME 8.3
    info = f.create_group("INFOS_GENERALES")
    info.attrs.create("MAJ", 3)
    info.attrs.create("MIN", 0)
    info.attrs.create("REL", 0)

    # Meshes
    ens_maa = f.create_group("ENS_MAA")
    mesh_name = "mesh"
    med_mesh = ens_maa.create_group(mesh_name)
    med_mesh.attrs.create("DIM", mesh.points.shape[1])  # mesh dimension
    med_mesh.attrs.create("ESP", mesh.points.shape[1])  # spatial dimension
    med_mesh.attrs.create("REP", 0)  # cartesian coordinate system (repère in french)
    med_mesh.attrs.create("UNT", b"")  # time unit
    med_mesh.attrs.create("UNI", b"")  # spatial unit
    med_mesh.attrs.create("SRT", 1)  # sorting type MED_SORT_ITDT
    med_mesh.attrs.create(
        "NOM", _comp_nom(mesh.points.shape[1]).encode("ascii")
    )  # component names
    med_mesh.attrs.create("DES", b"Mesh created with meshio")
    med_mesh.attrs.create("TYP", 0)  # mesh type (MED_NON_STRUCTURE)

    # Time-step
    step = "-0000000000000000001-0000000000000000001"  # NDT NOR
    ts = med_mesh.create_group(step)
    ts.attrs.create("CGT", 1)
    ts.attrs.create("NDT", -1)  # no time step (-1)
    ts.attrs.create("NOR", -1)  # no iteration step (-1)
    ts.attrs.create("PDT", -1.0)  # current time

    # Points
    noe_group = ts.create_group("NOE")
    noe_group.attrs.create("CGT", 1)
    noe_group.attrs.create("CGS", 1)
    pfl = "MED_NO_PROFILE_INTERNAL"
    noe_group.attrs.create("PFL", pfl.encode("ascii"))
    coo = noe_group.create_dataset("COO", data=mesh.points.T.flatten())
    coo.attrs.create("CGT", 1)
    coo.attrs.create("NBR", len(mesh.points))

    # Point tags
    if "point_tags" in mesh.point_data:  # works only for med -> med
        fam = noe_group.create_dataset("FAM", data=mesh.point_data["point_tags"])
        fam.attrs.create("CGT", 1)
        fam.attrs.create("NBR", len(mesh.points))

    # Cells (mailles in french)
    mai_group = ts.create_group("MAI")
    mai_group.attrs.create("CGT", 1)
    for cell_type, cells in mesh.cells.items():
        med_type = meshio_to_med_type[cell_type]
        mai = mai_group.create_group(med_type)
        mai.attrs.create("CGT", 1)
        mai.attrs.create("CGS", 1)
        mai.attrs.create("PFL", pfl.encode("ascii"))
        nod = mai.create_dataset("NOD", data=cells.T.flatten() + 1)
        nod.attrs.create("CGT", 1)
        nod.attrs.create("NBR", len(cells))

        # Cell tags
        if cell_type in mesh.cell_data:
            if "cell_tags" in mesh.cell_data[cell_type]:  # works only for med -> med
                fam = mai.create_dataset(
                    "FAM", data=mesh.cell_data[cell_type]["cell_tags"]
                )
                fam.attrs.create("CGT", 1)
                fam.attrs.create("NBR", len(cells))

    # Information about point and cell sets (familles in french)
    fas = f.create_group("FAS")
    fm = fas.create_group(mesh_name)
    fz = fm.create_group("FAMILLE_ZERO")  # must be defined in any case
    fz.attrs.create("NUM", 0)

    # For point tags
    try:
        if len(mesh.point_tags) > 0:
            noeud = fm.create_group("NOEUD")
            _write_families(noeud, mesh.point_tags)
    except AttributeError:
        pass

    # For cell tags
    try:
        if len(mesh.cell_tags) > 0:
            eleme = fm.create_group("ELEME")
            _write_families(eleme, mesh.cell_tags)
    except AttributeError:
        pass

    # Write nodal/cell data
    cha = f.create_group("CHA")

    # Nodal data
    for name, data in mesh.point_data.items():
        if name == "point_tags":  # ignore point_tags already written under FAS
            continue
        supp = "NOEU"  # nodal data
        _write_data(cha, mesh_name, pfl, name, supp, data)

    # Cell data
    # Only support writing ELEM fields with only 1 Gauss point per cell
    # Or ELNO (DG) fields defined at every node per cell
    for cell_type, d in mesh.cell_data.items():
        for name, data in d.items():
            if name == "cell_tags":  # ignore cell_tags already written under FAS
                continue

            # Determine the nature of the cell data
            # Either data.shape = (nbr, ) or (nbr, nco) -> ELEM
            # or data.shape = (nbr, nco, nga) -> ELNO or ELGA
            med_type = meshio_to_med_type[cell_type]
            nn = int(med_type[-1])  # number of nodes per cell
            if data.ndim <= 2:
                supp = "ELEM"
            elif data.shape[2] == nn:
                supp = "ELNO"
            else:  # general ELGA data defined at unknown Gauss points
                supp = "ELGA"
            _write_data(cha, mesh_name, pfl, name, supp, data, med_type)

    return


def _write_data(cha, mesh_name, pfl, name, supp, data, med_type=None):
    # Skip for general ELGA fields defined at unknown Gauss points
    if supp == "ELGA":
        return

    # Field
    try:  # a same MED field may contain fields of different natures
        field = cha.create_group(name)
        field.attrs.create("MAI", mesh_name.encode("ascii"))
        field.attrs.create("TYP", 6)  # MED_FLOAT64
        field.attrs.create("UNI", b"")  # physical unit
        field.attrs.create("UNT", b"")  # time unit
        nco = 1 if data.ndim == 1 else data.shape[1]
        field.attrs.create("NCO", nco)  # number of components
        field.attrs.create("NOM", _comp_nom(nco).encode("ascii"))

        # Time-step
        step = "0000000000000000000100000000000000000001"
        ts = field.create_group(step)
        ts.attrs.create("NDT", 1)  # time step 1
        ts.attrs.create("NOR", 1)  # iteration step 1
        ts.attrs.create("PDT", 0.0)  # current time
        ts.attrs.create("RDT", -1)  # NDT of the mesh
        ts.attrs.create("ROR", -1)  # NOR of the mesh

    except ValueError:  # name already exists
        field = cha[name]
        ts_name = list(field.keys())[-1]
        ts = field[ts_name]

    # Field information
    if supp == "NOEU":
        typ = ts.create_group("NOE")
    elif supp == "ELNO":
        typ = ts.create_group("NOE." + med_type)
    else:  # 'ELEM' with only 1 Gauss points!
        typ = ts.create_group("MAI." + med_type)

    typ.attrs.create("GAU", b"")  # no associated Gauss points
    typ.attrs.create("PFL", pfl.encode("ascii"))
    pfl = typ.create_group(pfl)
    pfl.attrs.create("NBR", len(data))  # number of points
    if supp == "ELNO":
        pfl.attrs.create("NGA", data.shape[2])
    else:
        pfl.attrs.create("NGA", 1)
    pfl.attrs.create("GAU", b"")

    # Data
    if supp == "NOEU" or supp == "ELEM":
        pfl.create_dataset("CO", data=data.T.flatten())
    else:  # ELNO fields
        data = numpy.swapaxes(data, 0, 1)
        pfl.create_dataset("CO", data=data.flatten())


def _comp_nom(nco):
    """
    To be correctly read in a MED viewer, each component must be a
    string of width 16. Since we do not know the physical nature of
    the data, we just use V1, V2, ...
    """
    return "".join(["V%-15d" % (i + 1) for i in range(nco)])


def _family_name(num, nom):
    """
    Return the FAM object name corresponding to
    the unique set id and a list of subset names
    """
    return "FAM" + "_" + str(num) + "_" + "_".join(nom)


def _write_families(fm_group, tags):
    """
    Write point/cell tag information under FAS/[mesh_name]
    """
    for num, nom in tags.items():
        fam = fm_group.create_group(_family_name(num, nom))
        fam.attrs.create("NUM", num)
        gro = fam.create_group("GRO")
        gro.attrs.create("NBR", len(nom))  # number of subsets
        dataset = gro.create_dataset("NOM", (len(nom),), dtype="80int8")
        for i in range(len(nom)):
            nom_80 = nom[i] + "\x00" * (80 - len(nom[i]))  # make nom 80 characters
            dataset[i] = [ord(x) for x in nom_80]
