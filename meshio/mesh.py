# -*- coding: utf-8 -*-
#
import numpy


class Mesh(object):
    def __init__(
        self,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        node_sets=None,
        gmsh_periodic=None,
    ):
        self.points = points
        self.cells = cells
        self.point_data = point_data if point_data else {}
        self.cell_data = cell_data if cell_data else {}
        self.field_data = field_data if field_data else {}
        self.node_sets = node_sets if node_sets else {}
        self.gmsh_periodic = gmsh_periodic
        return

    def __repr__(self):
        lines = []
        lines.append("Number of points: {}".format(len(self.points)))
        lines.append("Elements:")
        for tpe, elems in self.cells.items():
            lines.append("  Number of {}s: {}".format(tpe, len(elems)))

        if self.node_sets:
            lines.append("Node sets: {}".format(", ".join(self.node_sets.keys())))

        if self.point_data:
            lines.append("Point data: {}".format(", ".join(self.point_data.keys())))

        cell_data_keys = set()
        for cell_type in self.cell_data:
            cell_data_keys = cell_data_keys.union(self.cell_data[cell_type].keys())
        if cell_data_keys:
            lines.append("Cell data: {}".format(", ".join(cell_data_keys)))

        return "\n".join(lines)

    def prune(self):
        self.cells.pop("vertex", None)
        self.cells.pop("line", None)
        if "tetra" in self.cells:
            # remove_lower_order_cells
            self.cells.pop("triangle", None)
        # remove_orphaned_nodes.
        # find which nodes are not mentioned in the cells and remove them
        all_cells_flat = numpy.concatenate(
            [vals for vals in self.cells.values()]
        ).flatten()
        orphaned_nodes = numpy.setdiff1d(numpy.arange(len(self.points)), all_cells_flat)
        self.points = numpy.delete(self.points, orphaned_nodes, axis=0)
        # also adapt the point data
        for key in self.point_data:
            self.point_data[key] = numpy.delete(
                self.point_data[key], orphaned_nodes, axis=0
            )

        # reset GLOBAL_ID
        if "GLOBAL_ID" in self.point_data:
            self.point_data["GLOBAL_ID"] = numpy.arange(1, len(self.points) + 1)

        # We now need to adapt the cells too.
        diff = numpy.zeros(len(all_cells_flat), dtype=all_cells_flat.dtype)
        for orphan in orphaned_nodes:
            diff[numpy.argwhere(all_cells_flat > orphan)] += 1
        all_cells_flat -= diff
        k = 0
        for key in self.cells:
            s = self.cells[key].shape
            n = numpy.prod(s)
            self.cells[key] = all_cells_flat[k : k + n].reshape(s)
            k += n
        return
