import pytest
import os
import numpy

import helpers
import meshio


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.tri_mesh,
        helpers.quad_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
    ],
)
@pytest.mark.parametrize(
    "accuracy,ext",
    [
        (1.0e-7, ".ugrid"),
        (1.0e-15, ".b8.ugrid"),
        (1.0e-7, ".b4.ugrid"),
        (1.0e-15, ".lb8.ugrid"),
        (1.0e-7, ".lb4.ugrid"),
        (1.0e-15, ".r8.ugrid"),
        (1.0e-7, ".r4.ugrid"),
        (1.0e-15, ".lr8.ugrid"),
        (1.0e-7, ".lr4.ugrid"),
    ],
)
def test_io(mesh, accuracy, ext):
    helpers.write_read(meshio.ugrid.write, meshio.ugrid.read, mesh, accuracy, ext)


def test_generic_io():
   helpers.generic_io("test.lb8.ugrid")
   # With additional, insignificant suffix:
   helpers.generic_io("test.0.lb8.ugrid")

# sphere_mixed.1.lb8.ugrid and hch_strct.4.lb8.ugrid created
# using the codes from http://cfdbooks.com
@pytest.mark.parametrize(
        "filename, ref_num_points, ref_num_triangle, ref_num_quad, ref_num_wedge, ref_num_tet, ref_num_hex, ref_tag_counts", [
            ("sphere_mixed.1.lb8.ugrid", 3270, 864,0, 3024, 9072,0, { 1:432,2:216,3:216} ),
            ("hch_strct.4.lb8.ugrid", 306,12,178,96,0,144,{ 1:15,2:15,3:160})
            ]
)
def test_reference_file(filename, ref_num_points, ref_num_triangle, ref_num_quad,ref_num_wedge, ref_num_tet, ref_num_hex,ref_tag_counts):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(this_dir, "meshes", "ugrid", filename)

    mesh = meshio.read(filename)
    assert mesh.points.shape[0] == ref_num_points 
    assert mesh.points.shape[1] == 3
    print(mesh)
    # validate element counts
    if ref_num_triangle > 0 :
        assert mesh.cells["triangle"].shape[0] == ref_num_triangle
        assert mesh.cells["triangle"].shape[1] == 3
    else:
        assert "triangle" not in mesh.cells.keys()
    if ref_num_quad > 0 :
        assert mesh.cells["quad"].shape[0] == ref_num_quad
        assert mesh.cells["quad"].shape[1] == 4
    else:
        assert "quad" not in mesh.cells.keys()
    if ref_num_tet > 0 :
        assert mesh.cells["tetra"].shape[0] == ref_num_tet
        assert mesh.cells["tetra"].shape[1] == 4
    else:
        assert "tetra" not in mesh.cells.keys()
    if ref_num_wedge > 0 :
        assert mesh.cells["wedge"].shape[0] == ref_num_wedge
        assert mesh.cells["wedge"].shape[1] == 6
    else:
        assert "wedge" not in mesh.cells.keys()
    if ref_num_hex > 0 :
        assert mesh.cells["hexahedron"].shape[0] == ref_num_hex
        assert mesh.cells["hexahedron"].shape[1] == 8
    else:
        assert "hexahedron" not in mesh.cells.keys()
    
    # validate boundary tags
    
    # gather tags
    all_tags = []
    for surf_element in ["triangle","quad"]:
        if surf_element in mesh.cells:
            assert surf_element in mesh.cell_data.keys()
            assert "ugrid:ref" in mesh.cell_data[surf_element].keys()
            all_tags.append(mesh.cell_data[surf_element]["ugrid:ref"])
        
    all_tags = numpy.concatenate(all_tags)
    
    # validate against known values
    unique, counts = numpy.unique( all_tags,  return_counts=True)
    tags = dict( zip(unique,counts))
    assert tags.keys() == ref_tag_counts.keys()
    for key in tags.keys():
        assert tags[key] == ref_tag_counts[key]
