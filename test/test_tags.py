import os
import tempfile

import numpy
import pytest

import meshio


@pytest.mark.parametrize("mesh_format", [meshio.med, meshio.medit, meshio.avsucd])
def test_cell_tags(mesh_format):
    # Reference cell tags given by a GMSH file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "msh", "insulated-2.2.msh")
    mesh = meshio.read(filename)
    cell_tags_ref = mesh.cell_tags

    tmpfile = tempfile.NamedTemporaryFile().name
    mesh_format.write(tmpfile, mesh)
    cell_tags = mesh_format.read(tmpfile).cell_tags
    assert cell_tags is not None
    assert len(cell_tags) == len(cell_tags_ref)
    for i in range(len(cell_tags)):
        assert numpy.allclose(cell_tags[i], cell_tags_ref[i])


@pytest.mark.parametrize("mesh_format", [meshio.med, meshio.medit])
def test_tag_tags(mesh_format):
    # Reference cell tags given by a MED file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "med", "cylinder.med")
    mesh = meshio.read(filename)
    point_tags_ref = mesh.point_tags

    tmpfile = tempfile.NamedTemporaryFile().name
    mesh_format.write(tmpfile, mesh)
    point_tags = mesh_format.read(tmpfile).point_tags
    assert point_tags is not None
    assert numpy.allclose(point_tags, point_tags_ref)
