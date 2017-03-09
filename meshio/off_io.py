# -*- coding: utf-8 -*-
#
'''
I/O for the OFF surface format, cf.
<https://en.wikipedia.org/wiki/OFF_(file_format)>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
from itertools import islice
import numpy


def read(filename):
    with open(filename) as f:
        points, cells = read_buffer(f)

    return points, cells, {}, {}, {}


def read_buffer(f):
    # assert that the first line reads `OFF`
    line = next(islice(f, 1))
    assert line.strip() == 'OFF'

    # fast forward to the next significant line
    while True:
        line = next(islice(f, 1))
        stripped = line.strip()
        if len(stripped) > 0 and stripped[0] != '#':
            break

    # This next line contains:
    # <number of vertices> <number of faces> <number of edges>
    # 2775 5558 0
    num_verts, num_faces, num_edges = stripped.split(' ')
    num_verts = int(num_verts)
    num_faces = int(num_faces)
    num_edges = int(num_edges)

    verts = numpy.empty((num_verts, 3), dtype=float)

    # read vertices
    k = 0
    while True:
        if k >= num_verts:
            break

        try:
            line = next(islice(f, 1))
        except StopIteration:
            break
        stripped = line.strip()
        # skip comments and empty lines
        if len(stripped) == 0 or stripped[0] == '#':
            continue

        x, y, z = stripped.split()
        verts[k] = [float(x), float(y), float(z)]
        k += 1

    # read cells
    triangles = []
    k = 0
    while True:
        if k >= num_faces:
            break

        try:
            line = next(islice(f, 1))
        except StopIteration:
            break

        stripped = line.strip()

        # skip comments and empty lines
        if len(stripped) == 0 or stripped[0] == '#':
            continue

        data = stripped.split()
        num_points = int(data[0])
        assert num_points == len(data) - 1

        if not num_points == 3:
            raise RuntimeError('Can only handle triangular faces')

        data = [int(data[1]), int(data[2]), int(data[3])]
        triangles.append(data)

    cells = {}
    if len(triangles):
        cells['triangle'] = numpy.array(triangles)

    return verts, cells


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None
        ):

    for key in cells:
        if key not in ['triangle']:
            raise RuntimeError('Can only deal with triangular faces')

    tri = cells['triangle']

    with open(filename, 'wb') as fh:
        fh.write(b'OFF\n')
        fh.write(b'# Created by meshio\n\n')

        # counts
        c = '%d %d %d\n\n' % (len(points), len(tri), 0)
        fh.write(c.encode('utf-8'))

        # vertices
        numpy.savetxt(fh, points, '%r')

        # triangles
        data_with_label = numpy.c_[
                tri.shape[1] * numpy.ones(tri.shape[0]),
                tri
                ]
        numpy.savetxt(fh, data_with_label, '%d  %d %d %d')

    return
