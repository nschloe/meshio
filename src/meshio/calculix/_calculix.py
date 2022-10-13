from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import numpy as np

from .._helpers import register_format
from .._mesh import CellBlock, Mesh


class BaseData(Enum):
    @classmethod
    def from_str(cls, value: str):
        key_map = {x.value.lower(): x for x in cls}
        return key_map.get(value.lower(), None)


class PointData(BaseData):
    DISP = "DISP"
    FORC = "FORC"
    STRESS = "STRESS"
    PE = "PE"
    ERROR = "ERROR"


class FieldData(BaseData):
    pass


class ElemShape(Enum):
    WEDGE = "wedge"
    HEX = "hexahedron"

    @staticmethod
    def get_type_from_elem_array_shape(elements: np.ndarray) -> ElemShape:
        shape = elements.shape
        if shape[1] == 10:
            return ElemShape.WEDGE
        elif shape[1] == 12:
            return ElemShape.HEX
        else:
            raise NotImplementedError()


@dataclass
class CcxResult:
    name: str
    step: int
    components: list[str]
    values: list[tuple] = field(repr=False)


@dataclass
class CcxResultModel:
    file: Iterator

    nodes: np.ndarray = None
    elements: list[tuple] = None
    results: list[CcxResult] = field(default_factory=list)

    _curr_step: int = None

    def collect_nodes(self):
        while True:
            data = next(self.file)
            stripped = data.strip()
            if stripped.startswith("-1") is False:
                break
            split = stripped.split()
            data = [float(x) for x in split[1:]]
            yield data

    def collect_elements(self):
        elements = []
        curr_element = []
        while True:
            data = next(self.file)

            stripped = data.strip()
            if stripped.startswith("-1") is False:
                if stripped.startswith("-2") is False:
                    break

            split = stripped.split()

            if stripped.startswith("-1"):
                data = [int(x) for x in split[1:]]
                curr_element = data
            elif stripped.startswith("-2"):
                data = [int(x) for x in split[1:]]
                curr_element += data
                elements.append(tuple(curr_element))
                curr_element = []
            else:
                break

        self.elements = elements
        self.eval_flags(data)

    def collect_results(self, first_line):
        name = first_line.split()[1]

        component_names = []
        component_data = []
        while True:
            data = next(self.file)
            stripped = data.strip()

            if stripped.startswith("-5"):
                split = stripped.split()
                component_names.append(split[1])
            elif stripped.startswith("-1"):
                slen = len(stripped)
                x = 12
                res = [stripped[i : i + x] for i in range(0, slen, x)]
                first_batch = res[0].split()
                nid = int(float(first_batch[1]))
                component_data.append((nid, *[float(x) for x in res[1:]]))
            else:
                break

        self.results.append(
            CcxResult(name, self._curr_step, component_names, component_data)
        )
        self.eval_flags(data)

    def eval_flags(self, data: str):
        stripped = data.strip()
        if stripped.startswith("2C"):
            split = stripped.split()
            num_len = int(float(split[1]))
            self.nodes = np.fromiter(
                self.collect_nodes(), dtype=np.dtype((float, 4)), count=num_len
            )

        if stripped.startswith("3C"):
            self.collect_elements()

        if stripped.startswith("1PSTEP"):
            split_data = stripped.split()
            self._curr_step = int(float(split_data[2]))

        if stripped.startswith("-4"):
            self.collect_results(stripped)

    def load(self):
        while True:
            try:
                curr = next(self.file)
                self.eval_flags(curr)
            except StopIteration:
                break

    def get_last_step_results(self, data_type: BaseData) -> list[CcxResult]:
        results = dict()
        for res in self.results:
            if data_type.from_str(res.name) is None:
                continue
            existing_step = results.get(res.name)
            if existing_step is not None:
                if res.step > existing_step.step:
                    results[res.name] = res
            else:
                results[res.name] = res

        return list(results.values())

    def to_meshio_mesh(self) -> Mesh:
        # Points
        nodes = self.nodes
        points = nodes[:, 1:]
        monotonic_point_map = dict()
        for i, x in enumerate(nodes[:, 0].astype(int), start=0):
            monotonic_point_map[x] = i

        # Cells
        elements = np.asarray(self.elements)
        cells = elements[:, 4:]
        for original_num, new_num in monotonic_point_map.items():
            cells[cells == original_num] = new_num

        shape = ElemShape.get_type_from_elem_array_shape(elements)
        cell_block = CellBlock(str(shape.value), cells)

        # Point Data
        # Multiple steps are AFAIK not supported in the meshio Mesh object. So only the last step is used
        point_data = dict()
        for res in self.get_last_step_results(PointData):
            values = np.asarray(res.values)[:, 1:]
            point_data[res.name] = values

        # Field Data
        cell_data = dict()
        for res in self.get_last_step_results(FieldData):
            values = np.asarray(res.values)[:, 1:]
            cell_data[res.name] = [values]

        mesh = Mesh(
            points=points,
            cells=[cell_block],
            cell_data=cell_data,
            point_data=point_data,
        )
        return mesh


def read(frd_file) -> Mesh:
    with open(frd_file, "r") as f:
        ccx_res_model = CcxResultModel(f)
        ccx_res_model.load()

    mesh = ccx_res_model.to_meshio_mesh()
    return mesh


register_format("calculix", [".frd"], read, {"calculix": None})
