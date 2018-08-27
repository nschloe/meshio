# -*- coding: utf-8 -*-
#
import shlex
import numpy


def _read_physical_names(f, field_data):
    line = f.readline().decode("utf-8")
    num_phys_names = int(line)
    for _ in range(num_phys_names):
        line = shlex.split(f.readline().decode("utf-8"))
        key = line[2]
        value = numpy.array(line[1::-1], dtype=int)
        field_data[key] = value
    line = f.readline().decode("utf-8")
    assert line.strip() == "$EndPhysicalNames"
    return
