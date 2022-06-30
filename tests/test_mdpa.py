import pytest

import meshio

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.triangle6_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.tri_quad_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        # helpers.add_point_data(helpers.tri_mesh, 1), # NOTE: Data not supported yet
        # helpers.add_point_data(helpers.tri_mesh, 3),
        # helpers.add_point_data(helpers.tri_mesh, 9),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (), np.float64)]),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (3,), np.float64)]),
        # helpers.add_cell_data(helpers.tri_mesh, [("a", (9,), np.float64)]),
        # helpers.add_field_data(helpers.tri_mesh, [1, 2], int),
        # helpers.add_field_data(helpers.tet_mesh, [1, 3], int),
        # helpers.add_field_data(helpers.hex_mesh, [1, 3], int),
    ],
)
def test_io(mesh, tmp_path):
    helpers.write_read(tmp_path, meshio.mdpa.write, meshio.mdpa.read, mesh, 1.0e-15)


def test_generic_io(tmp_path):
    helpers.generic_io(tmp_path / "test.mesh")
    # With additional, insignificant suffix:
    helpers.generic_io(tmp_path / "test.0.mesh")


def test_write_from_gmsh(tmp_path):
    fg = tmp_path / "test.msh"
    fg.write_text(msh_mesh)
    m = meshio.read(fg, "gmsh")
    fk = tmp_path / "test.mdpa"
    m.write(fk, "mdpa")
    mdpa_mesh = fk.read_text().split("\n")
    assert mdpa_mesh == mdpa_mesh_ref


msh_mesh = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
6
2 2 "Inlet"
2 3 "Outlet"
2 4 "SYMM-Y0"
2 5 "Wall"
2 6 "SYMM-Z0"
3 1 "Fluid"
$EndPhysicalNames
$Entities
6 9 5 1
1 0 0 0 0
2 0 0 0.2 0
3 0 0.2 0 0
4 0.2 0 0 0
5 0.2 0 0.2 0
10 0.2 0.2 0 0
1 0 0 0 0 0 0.2 0 2 1 -2
2 0 0 0 0 0.2 0 0 2 3 -1
3 0 0 1.387778780781446e-17 0 0.2 0.2 0 2 2 -3
7 0.2 0 0 0.2 0 0.2 0 2 4 -5
8 0.2 0 1.387778780781446e-17 0.2 0.2 0.2 0 2 5 -10
9 0.2 0 0 0.2 0.2 0 0 2 10 -4
11 0 0 0 0.2 0 0 0 2 1 -4
12 0 0 0.2 0.2 0 0.2 0 2 2 -5
16 0 0.2 0 0.2 0.2 0 0 2 3 -10
5 0 0 0 0 0.2 0.2 1 2 3 1 3 2
13 0 0 0 0.2 0 0.2 1 4 4 1 12 -7 -11
17 0 0 0 0.2 0.2 0.2 1 5 4 3 16 -8 -12
21 0 0 0 0.2 0.2 0 1 6 4 2 11 -9 -16
22 0.2 0 0 0.2 0.2 0.2 1 3 3 7 8 9
1 0 0 0 0.2 0.2 0.2 1 1 5 -5 22 13 17 21
$EndEntities
$Nodes
18 24 1 24
0 1 0 1
1
0 0 0
0 2 0 1
2
0 0 0.2
0 3 0 1
3
0 0.2 0
0 4 0 1
4
0.2 0 0
0 5 0 1
5
0.2 0 0.2
0 10 0 1
6
0.2 0.2 0
1 1 0 2
7
8
0 0 0.06666666666650216
0 0 0.1333333333331544
1 2 0 2
9
10
0 0.1333333333335178 0
0 0.06666666666685292 0
1 3 0 2
11
12
0 0.1000000002601682 0.1732050806066796
0 0.1732050809154458 0.09999999972536942
1 7 0 2
13
14
0.2 0 0.06666666666650216
0.2 0 0.1333333333331544
1 8 0 2
15
16
0.2 0.1000000002601682 0.1732050806066796
0.2 0.1732050809154458 0.09999999972536942
1 9 0 2
17
18
0.2 0.1333333333335178 0
0.2 0.06666666666685292 0
2 5 0 3
19
20
21
0 0.1094024180527431 0.06355262272139157
0 0.06303928009977956 0.1102111067340192
0 0.05514522520565465 0.05557793336458285
2 13 0 0
2 17 0 0
2 21 0 0
2 22 0 3
22
23
24
0.2 0.1094024180527431 0.06355262272139157
0.2 0.06303928009977956 0.1102111067340192
0.2 0.05514522520565465 0.05557793336458285
3 1 0 0
$EndNodes
$Elements
9 30 1 30
2 5 2 1
1 20 19 21
2 5 3 6
2 9 19 12 3
3 10 21 19 9
4 20 21 7 8
5 11 20 8 2
6 11 12 19 20
7 1 7 21 10
2 13 3 3
8 1 7 13 4
9 7 8 14 13
10 8 2 5 14
2 17 3 3
11 2 11 15 5
12 11 12 16 15
13 12 3 6 16
2 21 3 3
14 3 9 17 6
15 9 10 18 17
16 10 1 4 18
2 22 2 1
17 23 22 24
2 22 3 6
18 17 22 16 6
19 18 24 22 17
20 23 24 13 14
21 15 23 14 5
22 15 16 22 23
23 4 13 24 18
3 1 5 6
24 12 19 9 3 16 22 17 6
25 19 21 10 9 22 24 18 17
26 7 21 20 8 13 24 23 14
27 8 20 11 2 14 23 15 5
28 19 12 11 20 22 16 15 23
29 21 7 1 10 24 13 4 18
3 1 6 1
30 19 20 21 22 23 24
$EndElements
"""

mdpa_mesh_ref = """Begin ModelPartData
End ModelPartData

Begin Properties 0
End Properties

Begin Nodes
 1 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
 2 0.0000000000000000e+00 0.0000000000000000e+00 2.0000000000000001e-01
 3 0.0000000000000000e+00 2.0000000000000001e-01 0.0000000000000000e+00
 4 2.0000000000000001e-01 0.0000000000000000e+00 0.0000000000000000e+00
 5 2.0000000000000001e-01 0.0000000000000000e+00 2.0000000000000001e-01
 6 2.0000000000000001e-01 2.0000000000000001e-01 0.0000000000000000e+00
 7 0.0000000000000000e+00 0.0000000000000000e+00 6.6666666666502158e-02
 8 0.0000000000000000e+00 0.0000000000000000e+00 1.3333333333315439e-01
 9 0.0000000000000000e+00 1.3333333333351780e-01 0.0000000000000000e+00
 10 0.0000000000000000e+00 6.6666666666852920e-02 0.0000000000000000e+00
 11 0.0000000000000000e+00 1.0000000026016820e-01 1.7320508060667961e-01
 12 0.0000000000000000e+00 1.7320508091544581e-01 9.9999999725369423e-02
 13 2.0000000000000001e-01 0.0000000000000000e+00 6.6666666666502158e-02
 14 2.0000000000000001e-01 0.0000000000000000e+00 1.3333333333315439e-01
 15 2.0000000000000001e-01 1.0000000026016820e-01 1.7320508060667961e-01
 16 2.0000000000000001e-01 1.7320508091544581e-01 9.9999999725369423e-02
 17 2.0000000000000001e-01 1.3333333333351780e-01 0.0000000000000000e+00
 18 2.0000000000000001e-01 6.6666666666852920e-02 0.0000000000000000e+00
 19 0.0000000000000000e+00 1.0940241805274310e-01 6.3552622721391575e-02
 20 0.0000000000000000e+00 6.3039280099779563e-02 1.1021110673401920e-01
 21 0.0000000000000000e+00 5.5145225205654652e-02 5.5577933364582853e-02
 22 2.0000000000000001e-01 1.0940241805274310e-01 6.3552622721391575e-02
 23 2.0000000000000001e-01 6.3039280099779563e-02 1.1021110673401920e-01
 24 2.0000000000000001e-01 5.5145225205654652e-02 5.5577933364582853e-02
End Nodes

Begin Conditions Triangle3D3
  1 0 20 19 21
End Conditions

Begin Conditions Quadrilateral3D4
  2 0 9 19 12 3
  3 0 10 21 19 9
  4 0 20 21 7 8
  5 0 11 20 8 2
  6 0 11 12 19 20
  7 0 1 7 21 10
End Conditions

Begin Conditions Quadrilateral3D4
  8 0 1 7 13 4
  9 0 7 8 14 13
  10 0 8 2 5 14
End Conditions

Begin Conditions Quadrilateral3D4
  11 0 2 11 15 5
  12 0 11 12 16 15
  13 0 12 3 6 16
End Conditions

Begin Conditions Quadrilateral3D4
  14 0 3 9 17 6
  15 0 9 10 18 17
  16 0 10 1 4 18
End Conditions

Begin Conditions Triangle3D3
  17 0 23 22 24
End Conditions

Begin Conditions Quadrilateral3D4
  18 0 17 22 16 6
  19 0 18 24 22 17
  20 0 23 24 13 14
  21 0 15 23 14 5
  22 0 15 16 22 23
  23 0 4 13 24 18
End Conditions

Begin Elements Hexahedra3D8
  1 0 12 19 9 3 16 22 17 6
  2 0 19 21 10 9 22 24 18 17
  3 0 7 21 20 8 13 24 23 14
  4 0 8 20 11 2 14 23 15 5
  5 0 19 12 11 20 22 16 15 23
  6 0 21 7 1 10 24 13 4 18
End Elements

Begin Elements Prism3D6
  7 0 19 20 21 22 23 24
End Elements

Begin SubModelPart Inlet
    Begin SubModelPartNodes
        1
        2
        3
        7
        8
        9
        10
        11
        12
        19
        20
        21
    End SubModelPartNodes
    Begin SubModelPartElements
    End SubModelPartElements
    Begin SubModelPartConditions
        1
        2
        3
        4
        5
        6
        7
    End SubModelPartConditions
End SubModelPart

Begin SubModelPart Outlet
    Begin SubModelPartNodes
        4
        5
        6
        13
        14
        15
        16
        17
        18
        22
        23
        24
    End SubModelPartNodes
    Begin SubModelPartElements
    End SubModelPartElements
    Begin SubModelPartConditions
        17
        18
        19
        20
        21
        22
        23
    End SubModelPartConditions
End SubModelPart

Begin SubModelPart SYMM-Y0
    Begin SubModelPartNodes
        1
        2
        4
        5
        7
        8
        13
        14
    End SubModelPartNodes
    Begin SubModelPartElements
    End SubModelPartElements
    Begin SubModelPartConditions
        8
        9
        10
    End SubModelPartConditions
End SubModelPart

Begin SubModelPart Wall
    Begin SubModelPartNodes
        2
        3
        5
        6
        11
        12
        15
        16
    End SubModelPartNodes
    Begin SubModelPartElements
    End SubModelPartElements
    Begin SubModelPartConditions
        11
        12
        13
    End SubModelPartConditions
End SubModelPart

Begin SubModelPart SYMM-Z0
    Begin SubModelPartNodes
        1
        3
        4
        6
        9
        10
        17
        18
    End SubModelPartNodes
    Begin SubModelPartElements
    End SubModelPartElements
    Begin SubModelPartConditions
        14
        15
        16
    End SubModelPartConditions
End SubModelPart

Begin SubModelPart Fluid
    Begin SubModelPartNodes
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
        11
        12
        13
        14
        15
        16
        17
        18
        19
        20
        21
        22
        23
        24
    End SubModelPartNodes
    Begin SubModelPartElements
        1
        2
        3
        4
        5
        6
        7
    End SubModelPartElements
    Begin SubModelPartConditions
    End SubModelPartConditions
End SubModelPart

""".split(
    "\n"
)
