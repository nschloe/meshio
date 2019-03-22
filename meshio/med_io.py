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
        point_data, cell_data, field_data = _read_data(fields)

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
    med_cells = mesh["MAI"]
    for med_cell_type, med_cell_type_group in med_cells.items():
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


def _read_data(fields):
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
    n_components = nodal_dataset.attrs["NBR"]
    values = nodal_dataset["CO"][()].reshape(-1, n_components).T
    if values.shape[1] == 1:  # cut off for scalars
        values = values[:, 0]
    return values


def _read_cell_data(cell_data, name, supp, data):
    med_type = supp.partition(".")[2]
    cell_dataset = data[supp][data[supp].attrs["PFL"]]
    n_components = cell_dataset.attrs["NBR"]
    n_gauss_points = cell_dataset.attrs["NGA"]

    # Only 1 Gauss/elemental nodal point per cell
    if n_gauss_points == 1:
        values = cell_dataset["CO"][()].reshape(-1, n_components).T
        if values.shape[1] == 1:  # cut off for scalars
            values = values[:, 0]
    # Multiple Gauss/elemental nodal points per cell
    # In general at each cell the value shape will be (n_components, n_gauss_points)
    else:
        values = cell_dataset["CO"][()].reshape(-1, n_components, n_gauss_points)
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
        set_id = node_set.attrs["NUM"]  # unique set id
        n_subsets = node_set["GRO"].attrs["NBR"]  # number of subsets
        nom_dataset = node_set["GRO"]["NOM"][()]  # (n_subsets, 80) of int8
        name = [None] * n_subsets
        for i in range(n_subsets):
            name[i] = "".join([chr(x) for x in nom_dataset[i] if x != 0])
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
    coo = nodes_group.create_dataset("COO", data=mesh.points.T.flatten())
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
        nod = med_cells.create_dataset("NOD", data=cells.T.flatten() + 1)
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
            # Either data.shape = (n_cells, ) or (n_cells, n_components) -> ELEM
            # or data.shape = (n_cells, n_components, n_gauss_points) -> ELNO or ELGA
            med_type = meshio_to_med_type[cell_type]
            if data.ndim <= 2:
                supp = "ELEM"
            elif data.shape[2] == num_nodes_per_cell[cell_type]:
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
        n_components = 1 if data.ndim == 1 else data.shape[1]
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
    profile.attrs.create("NBR", len(data))  # number of points
    if supp == "ELNO":
        profile.attrs.create("NGA", data.shape[2])
    else:
        profile.attrs.create("NGA", 1)
    profile.attrs.create("GAU", b"")

    # Data
    if supp == "NOEU" or supp == "ELEM":
        profile.create_dataset("CO", data=data.T.flatten())
    else:  # ELNO fields
        data = numpy.swapaxes(data, 0, 1)
        profile.create_dataset("CO", data=data.flatten())


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
