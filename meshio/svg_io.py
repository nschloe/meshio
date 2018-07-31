# -*- coding: utf-8 -*-
#
import numpy


def write(filename, mesh):
    from lxml import etree as ET

    if mesh.points.shape[1] == 3:
        assert (
            numpy.all(mesh.points[:, 2]) < 1.0e-14
        ), "SVG can only handle flat 2D meshes (shape: {})".format(
            mesh.points.shape
        )
        mesh.points = mesh.points[:, :2]

    assert mesh.points.shape[1] == 2

    # flip y, move corner point to (0, 0)
    pts = mesh.points.copy()
    pts[:, 1] *= -1
    pts[:, 0] -= numpy.min(pts[:, 0])
    pts[:, 1] -= numpy.min(pts[:, 1])

    width = numpy.max(pts[:, 0])
    height = numpy.max(pts[:, 1])

    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        height=str(height),
        width=str(width),
    )

    style = ET.SubElement(svg, "style")
    style.text = "polygon {fill: none; stroke: black;}"

    for cell_type in ["line", "triangle", "quad"]:
        if cell_type not in mesh.cells:
            continue
        for cell in mesh.cells[cell_type]:
            ET.SubElement(
                svg,
                "polygon",
                points=" ".join(
                    ["{:.1f},{:.1f}".format(pts[c, 0], pts[c, 1]) for c in cell]
                ),
            )

    tree = ET.ElementTree(svg)
    tree.write(filename, pretty_print=True)
    return
