import re
import numpy as np
from .._exceptions import ReadError
class pmkwd_line(object):
    """A valid line from permas-dat/dato/post file in the $STRUCTURE block"""
    
    # static variables
    ## re to split a permas keyword line
    splkwd_pattern =  '[A-Za-z0-9_]+(?:\s*=\s*)?[A-Za-z0-9_]+'
    splkwd         = re.compile(splkwd_pattern)

    ## re to parse a data line with only integers and floating point numbers
    num_pattern  =  '[e0-9+-.]+'
    num          =   re.compile(num_pattern,re.IGNORECASE)

    ## re to get first char of the line
    fchar_pattern  =  '^[\s\t]*(.)'
    fchar          =   re.compile(fchar_pattern)

    ## re to recoginize empty lines
    emptline_pattern = '^[\s\t]*\n$'
    emptline = re.compile(emptline_pattern)

    # constructor
    def __init__(self,location,string):
        self.location = location
        self.kwds     = type(self).splkwd.findall(string)

class nddef(pmkwd_line):
    
    """Begining line of a list of PERMAS node definitions"""
    def _read_nodes(self,f,nidx0,points,point_gids):
        f.seek(self.location)
        nidx = nidx0
        while True:
            dline = f.readline()
            if '$' in type(self).fchar.findall(dline):
                if nidx == nidx0:
                    raise ReadError('*E* PERMAS: No data line found under keyword $%s'% self.kwds[0])
                break
            elif type(self).emptline.findall(dline) or '!' in type(self).fchar.findall(dline)  : continue
            elif dline:
                entries = type(self).num.findall(dline)
                if len(entries) < 4:
                    raise ReadError('*E* PERMAS: Wrong coordinate on node %s'% entries[0])
                elif int(entries[0]) in point_gids.keys():
                    raise ReadError('*E* PERMAS: Duplicated definition on node %s'% entries[0])
                point_gids[int(entries[0])] = nidx
                points.append(np.array(entries[1:4]).astype(np.float))
                nidx += 1
        return nidx,points,point_gids

        
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
    permas_nelnd = {
    "PLOT1": 1,
    "PLOTL2": 2,
    "FLA2": 2,
    "FLA3": 2,
    "PLOTL3": 3,
    "BECOS": 2,
    "BECOC": 2,
    "BETAC": 2,
    "BECOP": 2,
    "BETOP": 2,
    "BEAM2": 2,
    "FSCPIPE2": 2,
    "LOADA4": 4,
    "PLOTA4": 4,
    "QUAD4": 4,
    "QUAD4S": 4,
    "QUAMS4": 4,
    "SHELL4": 4,
    "PLOTA8": 8,
    "LOADA8": 8,
    "QUAMS8": 8,
    "PLOTA9": 9,
    "LOADA9": 9,
    "QUAMS9": 9,
    "PLOTA3": 3,
    "SHELL3": 3,
    "TRIA3": 3,
    "TRIA3K": 3,
    "TRIA3S": 3,
    "TRIMS3": 3,
    "LOADA6": 6,
    "TRIMS6": 6,
    "HEXE8": 8,
    "HEXFO8": 8,
    "HEXE20": 20,
    "HEXE27": 27,
    "TET4": 4,
    "TET10": 10,
    "PYRA5": 5,
    "PENTA6": 6,
    "PENTA15": 15,
}
    """Begining line of a list of PERMAS element definitions"""
    def _read_cells(self,f,point_gids):
        """Reading cells with internal label(not from PERMAS)"""
        f.seek(self.location)
        petype = self.kwds[1].split('=')[1].strip()
        etype = type(self).permas_to_meshio_type[petype]
        if 'TYPE' not in self.kwds[1]:
            raise ReadError('*E* PERMAS: Element type not defined')
        elif petype not in type(self).permas_to_meshio_type.keys():
            raise ReadError(f"*E* PERMAS: Element type not available: {etype}")
        elems,elnds = [],[]
        # starting to read data lines
        eidx = []
        while True:
            dline = f.readline() 
            if '$' in type(self).fchar.findall(dline):
                if not eidx:
                    raise ReadError('*E* PERMAS: No data line found under keyword $%s'% (self.kwds[0] + ' ' +self.kwds[1]))
                break
            elif type(self).emptline.findall(dline) or '!' in type(self).fchar.findall(dline)  : continue
            elif dline:
                entries = type(self).num.findall(dline)
                nond    = type(self).permas_nelnd[petype]
                if len(entries) < nond+1:
                    raise ReadError('*E* PERMAS: Wrong number of nodal points defined for Element %s'% entries[0])
                pcell_def = np.array(entries[1:nond+1]).astype(np.int)
                eidx.append([point_gids[key] for key in pcell_def])
        return etype,np.array(eidx)

    
