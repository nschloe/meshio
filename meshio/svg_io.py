# -*- coding: utf-8 -*-
#
import numpy


def write(filename, mesh):
    from lxml import etree as ET

    if mesh.points.shape[1] == 3:
        assert numpy.allclose(
            mesh.points[:, 2], 0.0, rtol=0.0, atol=1.0e-14
        ), "SVG can only handle flat 2D meshes (shape: {})".format(mesh.points.shape)

    pts = mesh.points[:, :2].copy()
    pts[:, 1] = numpy.max(pts[:, 1]) - pts[:, 1]

    min_x = numpy.min(pts[:, 0])
    min_y = numpy.min(pts[:, 1])
    width = numpy.max(pts[:, 0]) - min_x
    height = numpy.max(pts[:, 1]) - min_y

    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        viewBox="{:.3f} {:.3f} {:.3f} {:.3f}".format(min_x, min_y, width, height),
    )

    style = ET.SubElement(svg, "style")
    style.text = "polygon {fill: none; stroke: black; stroke-width: 2%;}"

    for cell_type in ["line", "triangle", "quad"]:
        if cell_type not in mesh.cells:
            continue
        for cell in mesh.cells[cell_type]:
            ET.SubElement(
                svg,
                "polygon",
                points=" ".join(
                    ["{:.3f},{:.3f}".format(pts[c, 0], pts[c, 1]) for c in cell]
                ),
            )

    tree = ET.ElementTree(svg)
    tree.write(filename, pretty_print=True)
    return
