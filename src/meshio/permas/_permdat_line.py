import re
from .._exceptions import ReadError
class pmkwd_line(object):
    """A valid line from permas-dat/dato/post file in the $STRUCTURE block"""
    # static variables
    ## re to split a permas keyword line
    splkwd_pattern =  '[A-Za-z0-9_]+(?:\s*=\s*)?[A-Za-z0-9_]+'
    splkwd         = re.compile(splkwd_pattern)
    def __init__(self,location,string):
        self.location = location
        self.kwds     = type(self).splkwd.findall(string)
        
class eldef(pmkwd_line):
    permas_to_meshio_type = {
    "PLOT1": "vertex",
    "PLOTL2": "line",
    "FLA2": "line",
    "FLA3": "line3",
    "PLOTL3": "line3",
    "BECOS": "line",
    "BECOC": "line",
    "BETAC": "line",
    "BECOP": "line",
    "BETOP": "line",
    "BEAM2": "line",
    "FSCPIPE2": "line",
    "LOADA4": "quad",
    "PLOTA4": "quad",
    "QUAD4": "quad",
    "QUAD4S": "quad",
    "QUAMS4": "quad",
    "SHELL4": "quad",
    "PLOTA8": "quad8",
    "LOADA8": "quad8",
    "QUAMS8": "quad8",
    "PLOTA9": "quad9",
    "LOADA9": "quad9",
    "QUAMS9": "quad9",
    "PLOTA3": "triangle",
    "SHELL3": "triangle",
    "TRIA3": "triangle",
    "TRIA3K": "triangle",
    "TRIA3S": "triangle",
    "TRIMS3": "triangle",
    "LOADA6": "triangle6",
    "TRIMS6": "triangle6",
    "HEXE8": "hexahedron",
    "HEXFO8": "hexahedron",
    "HEXE20": "hexahedron20",
    "HEXE27": "hexahedron27",
    "TET4": "tetra",
    "TET10": "tetra10",
    "PYRA5": "pyramid",
    "PENTA6": "wedge",
    "PENTA15": "wedge15",
}
    """Begining line of a list of permas element definitions"""
    def _read_cells(self,f):#,point_gids):
        cell_type = self.kwds[1].split('=')[1].strip()
        if 'TYPE' not in self.kwds[1]:
            raise ReadError('*E* PERMAS Element type not defined')
        elif cell_type not in self.permas_to_meshio_type.keys():
            raise ReadError(f"*E* PERMAS Element type not available: {cell_type}")
        
