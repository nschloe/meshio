# -*- coding: utf-8 -*-
#
"""
I/O for MED/Salome, cf.
<https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html>.
"""
from .common import num_nodes_per_cell
from .mesh import Mesh
import numpy

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
    mesh_ensemble = f["ENS_MAA"]
    meshes = mesh_ensemble.keys()
    assert len(meshes) == 1, "Must only contain exactly 1 mesh"
    mesh_name = list(meshes)[0]
    mesh = mesh_ensemble[mesh_name]

    # Possible time-stepping
    if "NOE" not in mesh:
        # One needs NOE (node) and MAI (french maillage, meshing) data. If they
        # are not available in the mesh, check for time-steppings.
        time_step = mesh.keys()
        assert len(time_step) == 1, "Must only contain exactly 1 time-step"
        mesh = mesh[list(time_step)[0]]

    # Read nodal and cell data if they exist
    try:
        fields = f["CHA"]  # champs (fields) in french
    except KeyError:
        point_data, cell_data, field_data = {}, {}, {}
    else:
        profiles = f["PROFILS"] if "PROFILS" in f else None
        point_data, cell_data, field_data = _read_data(fields, profiles)

    # Points
    pts_dataset = mesh["NOE"]["COO"]
    n_points = pts_dataset.attrs["NBR"]
    points = pts_dataset[()].reshape(n_points, -1, order="F")

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
    med_cells = mesh["MAI"]
    for med_cell_type, med_cell_type_group in med_cells.items():
        cell_type = med_to_meshio_type[med_cell_type]
        nod = med_cell_type_group["NOD"]
        n_cells = nod.attrs["NBR"]
        cells[cell_type] = nod[()].reshape(n_cells, -1, order="F") - 1

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


def _read_data(fields, profiles):
    point_data = {}
    cell_data = {}
    field_data = {}
    for name, data in fields.items():
        time_step = sorted(data.keys())  # associated time-steps
        if len(time_step) == 1:  # single time-step
            names = [name]  # do not change field name
        else:  # many time-steps
            names = [None] * len(time_step)
            for i, key in enumerate(time_step):
                t = data[key].attrs["PDT"]  # current time
                names[i] = name + "[{:d}] - {:g}".format(i, t)

        # MED field can contain multiple types of data
        for i, key in enumerate(time_step):
            med_data = data[key]  # at a particular time step
            name = names[i]
            for supp in med_data:
                if supp == "NOE":  # continuous nodal (NOEU) data
                    point_data[name] = _read_nodal_data(med_data, profiles)
                else:  # Gauss points (ELGA) or DG (ELNO) data
                    cell_data = _read_cell_data(
                        cell_data, name, supp, med_data, profiles
                    )

    return point_data, cell_data, field_data


def _read_nodal_data(med_data, profiles):
    profile = med_data["NOE"].attrs["PFL"]
    data_profile = med_data["NOE"][profile]
    n_points = data_profile.attrs["NBR"]
    if profile.decode() == "MED_NO_PROFILE_INTERNAL":  # default profile with everything
        values = data_profile["CO"][()].reshape(n_points, -1, order="F")
    else:
        n_data = profiles[profile].attrs["NBR"]
        index_profile = profiles[profile]["PFL"][()] - 1
        values_profile = data_profile["CO"][()].reshape(n_data, -1, order="F")
        values = numpy.full((n_points, values_profile.shape[1]), numpy.nan)
        values[index_profile] = values_profile
    if values.shape[-1] == 1:  # cut off for scalars
        values = values[:, 0]
    return values


def _read_cell_data(cell_data, name, supp, med_data, profiles):
    med_type = supp.partition(".")[2]
    profile = med_data[supp].attrs["PFL"]
    data_profile = med_data[supp][profile]
    n_cells = data_profile.attrs["NBR"]
    n_gauss_points = data_profile.attrs["NGA"]
    if profile.decode() == "MED_NO_PROFILE_INTERNAL":  # default profile with everything
        values = data_profile["CO"][()].reshape(n_cells, n_gauss_points, -1, order="F")
    else:
        n_data = profiles[profile].attrs["NBR"]
        index_profile = profiles[profile]["PFL"][()] - 1
        values_profile = data_profile["CO"][()].reshape(
            n_data, n_gauss_points, -1, order="F"
        )
        values = numpy.full(
            (n_cells, values_profile.shape[1], values_profile.shape[2]), numpy.nan
        )
        values[index_profile] = values_profile

    # Only 1 data point per cell, shape -> (n_cells, n_components)
    if n_gauss_points == 1:
        values = values[:, 0, :]
        if values.shape[-1] == 1:  # cut off for scalars
            values = values[:, 0]

    try:  # cell type already exists
        key = med_to_meshio_type[med_type]
        cell_data[key][name] = values
    except KeyError:
        cell_data[key] = {name: values}
    return cell_data


def _read_families(fas_data):
    families = {}
    for _, node_set in fas_data.items():
        set_id = node_set.attrs["NUM"]  # unique set id
        n_subsets = node_set["GRO"].attrs["NBR"]  # number of subsets
        nom_dataset = node_set["GRO"]["NOM"][()]  # (n_subsets, 80) of int8
        name = [None] * n_subsets
        for i in range(n_subsets):
            name[i] = "".join([chr(x) for x in nom_dataset[i]]).strip().rstrip("\x00")
        families[set_id] = name
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
    mesh_ensemble = f.create_group("ENS_MAA")
    mesh_name = "mesh"
    med_mesh = mesh_ensemble.create_group(mesh_name)
    med_mesh.attrs.create("DIM", mesh.points.shape[1])  # mesh dimension
    med_mesh.attrs.create("ESP", mesh.points.shape[1])  # spatial dimension
    med_mesh.attrs.create("REP", 0)  # cartesian coordinate system (repère in french)
    med_mesh.attrs.create("UNT", b"")  # time unit
    med_mesh.attrs.create("UNI", b"")  # spatial unit
    med_mesh.attrs.create("SRT", 1)  # sorting type MED_SORT_ITDT
    med_mesh.attrs.create(
        "NOM", _component_names(mesh.points.shape[1]).encode("ascii")
    )  # component names
    med_mesh.attrs.create("DES", b"Mesh created with meshio")
    med_mesh.attrs.create("TYP", 0)  # mesh type (MED_NON_STRUCTURE)

    # Time-step
    step = "-0000000000000000001-0000000000000000001"  # NDT NOR
    time_step = med_mesh.create_group(step)
    time_step.attrs.create("CGT", 1)
    time_step.attrs.create("NDT", -1)  # no time step (-1)
    time_step.attrs.create("NOR", -1)  # no iteration step (-1)
    time_step.attrs.create("PDT", -1.0)  # current time

    # Points
    nodes_group = time_step.create_group("NOE")
    nodes_group.attrs.create("CGT", 1)
    nodes_group.attrs.create("CGS", 1)
    profile = "MED_NO_PROFILE_INTERNAL"
    nodes_group.attrs.create("PFL", profile.encode("ascii"))
    coo = nodes_group.create_dataset("COO", data=mesh.points.flatten(order="F"))
    coo.attrs.create("CGT", 1)
    coo.attrs.create("NBR", len(mesh.points))

    # Point tags
    if "point_tags" in mesh.point_data:  # works only for med -> med
        family = nodes_group.create_dataset("FAM", data=mesh.point_data["point_tags"])
        family.attrs.create("CGT", 1)
        family.attrs.create("NBR", len(mesh.points))

    # Cells (mailles in french)
    cells_group = time_step.create_group("MAI")
    cells_group.attrs.create("CGT", 1)
    for cell_type, cells in mesh.cells.items():
        med_type = meshio_to_med_type[cell_type]
        med_cells = cells_group.create_group(med_type)
        med_cells.attrs.create("CGT", 1)
        med_cells.attrs.create("CGS", 1)
        med_cells.attrs.create("PFL", profile.encode("ascii"))
        nod = med_cells.create_dataset("NOD", data=cells.flatten(order="F") + 1)
        nod.attrs.create("CGT", 1)
        nod.attrs.create("NBR", len(cells))

        # Cell tags
        if cell_type in mesh.cell_data:
            if "cell_tags" in mesh.cell_data[cell_type]:  # works only for med -> med
                family = med_cells.create_dataset(
                    "FAM", data=mesh.cell_data[cell_type]["cell_tags"]
                )
                family.attrs.create("CGT", 1)
                family.attrs.create("NBR", len(cells))

    # Information about point and cell sets (familles in french)
    fas = f.create_group("FAS")
    families = fas.create_group(mesh_name)
    family_zero = families.create_group("FAMILLE_ZERO")  # must be defined in any case
    family_zero.attrs.create("NUM", 0)

    # For point tags
    try:
        if len(mesh.point_tags) > 0:
            node = families.create_group("NOEUD")
            _write_families(node, mesh.point_tags)
    except AttributeError:
        pass

    # For cell tags
    try:
        if len(mesh.cell_tags) > 0:
            element = families.create_group("ELEME")
            _write_families(element, mesh.cell_tags)
    except AttributeError:
        pass

    # Write nodal/cell data
    fields = f.create_group("CHA")

    # Nodal data
    for name, data in mesh.point_data.items():
        if name == "point_tags":  # ignore point_tags already written under FAS
            continue
        supp = "NOEU"  # nodal data
        _write_data(fields, mesh_name, profile, name, supp, data)

    # Cell data
    # Only support writing ELEM fields with only 1 Gauss point per cell
    # Or ELNO (DG) fields defined at every node per cell
    for cell_type, d in mesh.cell_data.items():
        for name, data in d.items():
            if name == "cell_tags":  # ignore cell_tags already written under FAS
                continue

            # Determine the nature of the cell data
            # Either shape = (n_data, ) or (n_data, n_components) -> ELEM
            # or shape = (n_data, n_gauss_points, n_components) -> ELNO or ELGA
            med_type = meshio_to_med_type[cell_type]
            if data.ndim <= 2:
                supp = "ELEM"
            elif data.shape[1] == num_nodes_per_cell[cell_type]:
                supp = "ELNO"
            else:  # general ELGA data defined at unknown Gauss points
                supp = "ELGA"
            _write_data(fields, mesh_name, profile, name, supp, data, med_type)

    return


def _write_data(fields, mesh_name, profile, name, supp, data, med_type=None):
    # Skip for general ELGA fields defined at unknown Gauss points
    if supp == "ELGA":
        return

    # Field
    try:  # a same MED field may contain fields of different natures
        field = fields.create_group(name)
        field.attrs.create("MAI", mesh_name.encode("ascii"))
        field.attrs.create("TYP", 6)  # MED_FLOAT64
        field.attrs.create("UNI", b"")  # physical unit
        field.attrs.create("UNT", b"")  # time unit
        n_components = 1 if data.ndim == 1 else data.shape[-1]
        field.attrs.create("NCO", n_components)  # number of components
        field.attrs.create("NOM", _component_names(n_components).encode("ascii"))

        # Time-step
        step = "0000000000000000000100000000000000000001"
        time_step = field.create_group(step)
        time_step.attrs.create("NDT", 1)  # time step 1
        time_step.attrs.create("NOR", 1)  # iteration step 1
        time_step.attrs.create("PDT", 0.0)  # current time
        time_step.attrs.create("RDT", -1)  # NDT of the mesh
        time_step.attrs.create("ROR", -1)  # NOR of the mesh

    except ValueError:  # name already exists
        field = fields[name]
        ts_name = list(field.keys())[-1]
        time_step = field[ts_name]

    # Field information
    if supp == "NOEU":
        typ = time_step.create_group("NOE")
    elif supp == "ELNO":
        typ = time_step.create_group("NOE." + med_type)
    else:  # 'ELEM' with only 1 Gauss points!
        typ = time_step.create_group("MAI." + med_type)

    typ.attrs.create("GAU", b"")  # no associated Gauss points
    typ.attrs.create("PFL", profile.encode("ascii"))
    profile = typ.create_group(profile)
    profile.attrs.create("NBR", len(data))  # number of data
    if supp == "ELNO":
        profile.attrs.create("NGA", data.shape[1])
    else:
        profile.attrs.create("NGA", 1)
    profile.attrs.create("GAU", b"")

    # Dataset
    profile.create_dataset("CO", data=data.flatten(order="F"))


def _component_names(n_components):
    """
    To be correctly read in a MED viewer, each component must be a
    string of width 16. Since we do not know the physical nature of
    the data, we just use V1, V2, ...
    """
    return "".join(["V%-15d" % (i + 1) for i in range(n_components)])


def _family_name(set_id, name):
    """
    Return the FAM object name corresponding to
    the unique set id and a list of subset names
    """
    return "FAM" + "_" + str(set_id) + "_" + "_".join(name)


def _write_families(fm_group, tags):
    """
    Write point/cell tag information under FAS/[mesh_name]
    """
    for set_id, name in tags.items():
        family = fm_group.create_group(_family_name(set_id, name))
        family.attrs.create("NUM", set_id)
        group = family.create_group("GRO")
        group.attrs.create("NBR", len(name))  # number of subsets
        dataset = group.create_dataset("NOM", (len(name),), dtype="80int8")
        for i in range(len(name)):
            name_80 = name[i] + "\x00" * (80 - len(name[i]))  # make name 80 characters
            dataset[i] = [ord(x) for x in name_80]
