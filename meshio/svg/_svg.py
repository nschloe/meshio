import xml.etree.ElementTree as ET

import numpy

from .._exceptions import WriteError
from .._helpers import register


def write(filename, mesh, float_fmt=".3f", stroke_width="1", force_width=None):
    if mesh.points.shape[1] == 3 and not numpy.allclose(
        mesh.points[:, 2], 0.0, rtol=0.0, atol=1.0e-14
    ):
        raise WriteError(
            "SVG can only handle flat 2D meshes (shape: {})".format(mesh.points.shape)
        )

    pts = mesh.points[:, :2].copy()
    pts[:, 1] = numpy.max(pts[:, 1]) - pts[:, 1]

    min_x = numpy.min(pts[:, 0])
    min_y = numpy.min(pts[:, 1])
    width = numpy.max(pts[:, 0]) - min_x
    height = numpy.max(pts[:, 1]) - min_y

    if force_width is not None:
        scaling_factor = force_width / width
        min_x *= scaling_factor
        min_y *= scaling_factor
        width *= scaling_factor
        height *= scaling_factor
        pts *= scaling_factor

    fmt = " ".join(4 * ["{{:{}}}".format(float_fmt)])
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        viewBox=fmt.format(min_x, min_y, width, height),
    )

    style = ET.SubElement(svg, "style")
    opts = [
        "fill: none",
        "stroke: black",
        "stroke-width: {}".format(stroke_width),
        "stroke-linejoin:bevel",
    ]
    # Use path, not polygon, because svgo converts polygons to paths and doesn't convert
    # the style alongside. No problem it's paths all along.
    style.text = "path {" + "; ".join(opts) + "}"

    for cell_block in mesh.cells:
        if cell_block.type not in ["line", "triangle", "quad"]:
            continue
        if cell_block.type == "line":
            fmt = "M {{:{}}} {{:{}}}".format(
                float_fmt, float_fmt
            ) + "L {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
        elif cell_block.type == "triangle":
            fmt = (
                "M {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "L {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "L {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "Z"
            )
        elif cell_block.type == "quad":
            fmt = (
                "M {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "L {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "L {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "L {{:{}}} {{:{}}}".format(float_fmt, float_fmt)
                + "Z"
            )
        for cell in cell_block.data:
            ET.SubElement(
                svg, "path", d=fmt.format(*pts[cell].flatten()),
            )

    tree = ET.ElementTree(svg)
    tree.write(filename)


register("svg", [".svg"], None, {"svg": write})
