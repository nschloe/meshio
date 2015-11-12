# -*- coding: utf-8 -*-
#
__version__ = '0.1.4'
__author__ = 'Nico Schlömer'
__author_email__ = 'nico.schloemer@gmail.com'
__website__ = 'https://github.com/nschloe/meshio'

from meshio.io import read
from meshio.io import write

__all__ = [
        'h5m',
        'io',
        'gmsh',
        'vtk'
        ]
