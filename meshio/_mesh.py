import collections

import numpy

from ._common import _topological_dimension


class CellBlock(collections.namedtuple("CellBlock", ["type", "data"])):
    def __repr__(self):
        return f"<meshio CellBlock, type: {self.type}, num cells: {len(self.data)}>"


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
        if isinstance(cells, dict):
            # Let's not deprecate this for now.
            # import warnings
            # warnings.warn(
            #     "cell dictionaries are deprecated, use list of tuples, e.g., "
            #     '[("triangle", [[0, 1, 2], ...])]',
            #     DeprecationWarning,
            # )
            # old dict, deprecated
            self.cells = [
                CellBlock(cell_type, data) for cell_type, data in cells.items()
            ]
        else:
            self.cells = [CellBlock(cell_type, data) for cell_type, data in cells]
        self.point_data = {} if point_data is None else point_data
        self.cell_data = {} if cell_data is None else cell_data
        self.field_data = {} if field_data is None else field_data
        self.point_sets = {} if point_sets is None else point_sets
        self.cell_sets = {} if cell_sets is None else cell_sets
        self.gmsh_periodic = gmsh_periodic
        self.info = info

        for key, data in self.cell_data.items():
            assert len(data) == len(cells), (
                "Incompatible cell data. "
                f"{len(cells)} cell blocks, but '{key}' has {len(data)} blocks."
            )

    def __repr__(self):
        lines = ["<meshio mesh object>", f"  Number of points: {len(self.points)}"]
        if len(self.cells) > 0:
            lines.append("  Number of cells:")
            for tpe, elems in self.cells:
                lines.append(f"    {tpe}: {len(elems)}")
        else:
            lines.append("  No cells.")

        if self.point_sets:
            names = ", ".join(self.point_sets.keys())
            lines.append(f"  Point sets: {names}")

        if self.cell_sets:
            names = ", ".join(self.cell_sets.keys())
            lines.append(f"  Cell sets: {names}")

        if self.point_data:
            names = ", ".join(self.point_data.keys())
            lines.append(f"  Point data: {names}")

        if self.cell_data:
            names = ", ".join(self.cell_data.keys())
            lines.append(f"  Cell data: {names}")

        return "\n".join(lines)

    def remove_lower_dimensional_cells(self):
        """Remove all cells of topological dimension lower than the max dimension in the
        mesh, i.e., in a mesh that contains tetrahedra, remove triangles, lines, etc.
        """
        max_topological_dim = max(_topological_dimension[c.type] for c in self.cells)
        new_cells = []
        new_cell_data = {}
        new_cell_sets = {}
        prune_set = set()
        for idx, c in enumerate(self.cells):
            if _topological_dimension[c.type] == max_topological_dim:
                new_cells.append(c)

                for name, data in self.cell_data.items():
                    if name not in new_cell_data:
                        new_cell_data[name] = []
                    new_cell_data[name] += [data[idx]]

                for name, data in self.cell_sets.items():
                    if name not in new_cell_sets:
                        new_cell_sets[name] = []
                    new_cell_sets[name] += [data[idx]]
            else:
                prune_set.add(c.type)

        self.cells = new_cells
        self.cell_data = new_cell_data
        self.cell_sets = new_cell_sets
        return prune_set

    def remove_orphaned_nodes(self):
        """Remove nodes which don't belong to any cell."""
        all_cells_flat = numpy.concatenate([c.data.flat for c in self.cells])
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
        for k, c in enumerate(self.cells):
            s = c.data.shape
            n = numpy.prod(s)
            self.cells[k] = CellBlock(c.type, all_cells_flat[k : k + n].reshape(s))
            k += n

    def prune_z_0(self, tol=1.0e-13):
        """Remove third (z) component of points if it is 0 everywhere (up to a
        tolerance).
        """
        if self.points.shape[1] == 3 and numpy.all(numpy.abs(self.points[:, 2]) < tol):
            self.points = self.points[:, :2]

    def write(self, path_or_buf, file_format=None, **kwargs):
        # avoid circular import
        from ._helpers import write

        write(path_or_buf, self, file_format, **kwargs)

    def get_cells_type(self, cell_type):
        if not any(c.type == cell_type for c in self.cells):
            return numpy.array([], dtype=int)
        return numpy.concatenate([c.data for c in self.cells if c.type == cell_type])

    def get_cell_data(self, name, cell_type):
        return numpy.concatenate(
            [d for c, d in zip(self.cells, self.cell_data[name]) if c.type == cell_type]
        )

    @property
    def cells_dict(self):
        cells_dict = {}
        for cell_type, data in self.cells:
            if cell_type not in cells_dict:
                cells_dict[cell_type] = []
            cells_dict[cell_type].append(data)
        # concatenate
        for key, value in cells_dict.items():
            cells_dict[key] = numpy.concatenate(value)
        return cells_dict

    @property
    def cell_data_dict(self):
        cell_data_dict = {}
        for key, value_list in self.cell_data.items():
            cell_data_dict[key] = {}
            for value, (cell_type, _) in zip(value_list, self.cells):
                if cell_type not in cell_data_dict[key]:
                    cell_data_dict[key][cell_type] = []
                cell_data_dict[key][cell_type].append(value)

            for cell_type, val in cell_data_dict[key].items():
                cell_data_dict[key][cell_type] = numpy.concatenate(val)
        return cell_data_dict

    @property
    def cell_sets_dict(self):
        sets_dict = {}
        for key, member_list in self.cell_sets.items():
            sets_dict[key] = {}
            offsets = {}
            for members, cells in zip(member_list, self.cells):
                if cells.type in offsets:
                    offset = offsets[cells.type]
                    offsets[cells.type] += cells.data.shape[0]
                else:
                    offset = 0
                    offsets[cells.type] = cells.data.shape[0]
                if cells.type in sets_dict[key]:
                    sets_dict[key][cells.type].append(members + offset)
                else:
                    sets_dict[key][cells.type] = [members + offset]
        return {
            key: {
                cell_type: numpy.concatenate(members)
                for cell_type, members in sets.items()
                if sum(map(numpy.size, members))
            }
            for key, sets in sets_dict.items()
        }

    @classmethod
    def read(cls, path_or_buf, file_format=None):
        # avoid circular import
        from ._helpers import read

        return read(path_or_buf, file_format)

    def sets_to_int_data(self):
        # If possible, convert cell sets to integer cell data. This is possible if all
        # cells appear exactly in one group.
        intfun = []
        for k, c in enumerate(zip(*self.cell_sets.values())):
            # `c` contains the values of all cell sets for a particular cell block
            c = [([] if cc is None else cc) for cc in c]
            # check if all numbers appear exactly once in the groups
            d = numpy.sort(numpy.concatenate(c))
            if numpy.all(d == numpy.arange(len(d))):
                arr = numpy.empty(len(d), dtype=int)
                arr[:] = numpy.nan
                for k, cc in enumerate(c):
                    arr[cc] = k
            else:
                # We could just append None, but some mesh formats expect _something_
                # here. Go for an array of NaNs.
                arr = numpy.empty(len(self.cells[k]), dtype=int)
                arr[:] = numpy.nan

            intfun.append(arr)

        data_name = "-".join(self.cell_sets.keys())
        self.cell_data = {data_name: intfun}
        self.cell_sets = {}

    def int_data_to_sets(self):
        """Convert all int data to {point,cell}_sets, where possible."""
        keys = []
        for key, data in self.cell_data.items():
            # handle all int and uint data
            if not numpy.all(v.dtype.kind in ["i", "u"] for v in data):
                continue

            keys.append(key)

            # this call can be rather expensive
            tags = numpy.unique(numpy.concatenate(data))

            # try and get the names by splitting the key along "-" (this is how
            # sets_to_int_data() forms the key
            names = sorted(list(set(key.split("-"))))
            if len(names) != len(tags):
                # alternative names
                names = [f"set{tag}" for tag in tags]

            for name, tag in zip(names, tags):
                self.cell_sets[name] = []
                self.cell_sets[name] = [numpy.where(d == tag)[0] for d in data]

        # remove the cell data
        for key in keys:
            del self.cell_data[key]
