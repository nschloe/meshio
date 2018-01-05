# -*- coding: utf-8 -*-
#
'''
I/O for the STL format, cf.
<https://en.wikipedia.org/wiki/STL_(file_format)>.

.. moduleauthor:: Nico Schlömer <nico.schloemer@gmail.com>
'''
import numpy


def read(filename):
    '''Reads a Gmsh msh file.
    '''
    with open(filename, 'rb') as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # The format is specified at
    # <http://gmsh.info//doc/texinfo/gmsh.html#MSH-ASCII-file-format>.

    # Initialize the optional data fields
    points = []
    cells = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    line = f.readline().decode('utf-8')
    print(line)
    assert line[:5] == 'solid'

    facets = []
    while True:
        line = f.readline().decode('utf-8')

        if line[:8] == 'endsolid':
            break

        line = line.strip()
        assert line[:5] == 'facet'
        facets.append(_read_facet(f))
        line = f.readline().decode('utf-8')
        assert line.strip() == 'endfacet'

    print(facets)
    exit(1)
    return points, cells, point_data, cell_data, field_data


def _read_facet(f):
    line = f.readline().decode('utf-8')
    assert line.strip() == 'outer loop'

    facet = numpy.empty((3, 3))

    flt = numpy.vectorize(float)
    for k in range(3):
        parts = f.readline().decode('utf-8').split()
        assert len(parts) == 4
        assert parts[0] == 'vertex'
        facet[k] = flt(parts[1:])

    line = f.readline().decode('utf-8')
    assert line.strip() == 'endloop'
    return facet


def write(
        filename,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        write_binary=True,
        ):

    return
