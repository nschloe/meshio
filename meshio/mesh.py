# -*- coding: utf-8 -*-
#


class Mesh(object):
    def __init__(self, points, cells, point_data=None, cell_data=None, field_data=None):
        self.points = points
        self.cells = cells
        self.point_data = point_data if point_data else {}
        self.cell_data = cell_data if cell_data else {}
        self.field_data = field_data if field_data else {}
        return

    def __repr__(self):
        string = []
        string.append("meshio mesh:")
        string.append("    num points: {}".format(self.points.shape[0]))
        string.append("    num cells:")
        for key, data in self.cells.items():
            string.append("        {}: {}".format(key, data.shape[0]))

        if self.point_data:
            string.append("    point data: {}".format(", ".join(self.point_data.keys())))
        if self.field_data:
            string.append("    field data: {}".format(", ".join(self.field_data.keys())))
        # TODO cell data
        return "\n".join(string)
