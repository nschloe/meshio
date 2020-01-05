import numpy


class Mesh:
    def __init__(
        self,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        point_sets=None,
        cell_sets=None,
        gmsh_periodic=None,
        info=None,
    ):
        self.points = points
        self.cells = cells
        self.point_data = {} if point_data is None else point_data
        self.cell_data = {} if cell_data is None else cell_data
        self.field_data = {} if field_data is None else field_data
        self.point_sets = {} if point_sets is None else point_sets
        self.cell_sets = {} if cell_sets is None else cell_sets
        self.gmsh_periodic = gmsh_periodic
        self.info = info

    def __repr__(self):
        lines = [
            "<meshio mesh object>",
            "  Number of points: {}".format(len(self.points)),
        ]
        if len(self.cells) > 0:
            lines.append("  Number of cells:")
            for tpe, elems in self.cells.items():
                lines.append("    {}: {}".format(tpe, len(elems)))
        else:
            lines.append("  No cells.")

        if self.point_sets:
            lines.append("  Point sets: {}".format(", ".join(self.point_sets.keys())))

        if self.point_data:
            lines.append("  Point data: {}".format(", ".join(self.point_data.keys())))

        cell_data_keys = set()
        for cell_type in self.cell_data:
            cell_data_keys = cell_data_keys.union(self.cell_data[cell_type].keys())
        if cell_data_keys:
            lines.append("  Cell data: {}".format(", ".join(cell_data_keys)))

        return "\n".join(lines)

    def prune(self):
        prune_list = []

        for cell_type in ["vertex", "line", "line3"]:
            if cell_type in self.cells:
                prune_list.append(cell_type)

        self.cells.pop("vertex", None)
        self.cells.pop("line", None)
        self.cells.pop("line3", None)
        if "tetra" in self.cells or "tetra10" in self.cells:
            # remove_lower_order_cells
            for cell_type in ["triangle", "triangle6"]:
                if cell_type in self.cells:
                    prune_list.append(cell_type)

        for cell_type in prune_list:
            self.cells.pop(cell_type, None)
            self.cell_data.pop(cell_type, None)

        print("Pruned cell types: {}".format(", ".join(prune_list)))

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

    def write(self, path_or_buf, file_format=None, **kwargs):
        # avoid circular import
        from ._helpers import write

        write(path_or_buf, self, file_format, **kwargs)

    @classmethod
    def read(cls, path_or_buf, file_format=None):
        # avoid circular import
        from ._helpers import read

        return read(path_or_buf, file_format)

    def iterpoints(self):
        for i, point in enumerate(self.points):
            data = (
                {label: data_array[i] for label, data_array in self.point_data.items()}
                if self.point_data
                else {}
            )
            yield (point, data)

    def itercells(self):
        from ._common import num_nodes_per_cell

        for cell_type in sorted(self.cells.keys(), key=lambda x: num_nodes_per_cell[x]):
            for i, corner in enumerate(self.cells[cell_type]):
                data = (
                    {
                        label: data_array[i]
                        for label, data_array in self.cell_data[cell_type].items()
                    }
                    if cell_type in self.cell_data
                    else {}
                )
                yield (corner, data, cell_type)
