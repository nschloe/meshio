"""
Input for PERMAS dat files.
"""
import numpy as np
import re
from . import _permdat_line as pdl
from .._exceptions import ReadError
from .._mesh import CellBlock, Mesh

# re to get keyword that follows dollar sign
dlkwd_pattern = '^[\s\t]*\$([A-Z,a-z]*) *.*'
dlkwd         = re.compile(dlkwd_pattern)
estrc_pattern = '.*\$END\s+STRUCTURE'
estrc         = re.compile(estrc_pattern,re.IGNORECASE)
fchar_pattern  =  '^[\s\t]*(.)'
fchar          =   re.compile(fchar_pattern)


def read_buffer(f):
    # Initialize the optional data fields
    cells = []
    nsets = {}
    elsets = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    # initialize counters and dictionaries of blocks
    ## element blocks
    ecnt  = 0
    eblk  = {}
    ## element set blocks
    escnt = 0
    esblk = {}
    ## node blocks (not sure if only one allowed)
    nidx  = 0
    ncnt  = 0
    nblk  = {}
    ## node set blocks
    nscnt = 0
    nsblk = {}

    # initialize outputs
    points = []
    point_gids = {}
    
    # start scanning lines
    while True:
        line = f.readline()
        keyword = ''
        # EOF or end of structure definition
        if not line or estrc.match(line):
            if not nblk:
                raise ReadError('*E* PERMAS: No nodal point found till the end of the structure definition')
            break
        # Comments
        elif '!' in fchar.findall(line):
            continue
        # if a keyword in current line?
        elif '$' in fchar.findall(line):
            keyword = dlkwd.findall(line)[0].upper()

        # find all key word positions
        if keyword.startswith("COOR"):
            nblk[ncnt] = pdl.nddef(f.tell(),line)
            ncnt += 1
        elif keyword.startswith("ELEMENT"):
            eblk[ecnt] = pdl.eldef(f.tell(),line)
            ecnt += 1
        elif keyword.startswith("NSET"):
            nsblk[nscnt] = pdl.setdef(f.tell(),line)
            nscnt += 1
        elif keyword.startswith("ESET"):
            esblk[escnt] = pdl.setdef(f.tell(),line)
            escnt += 1
        else:pass
    
    # read points
    for key in nblk.keys():
        nidx,points,point_gids = nblk[key]._read_nodes(f,nidx,points,point_gids) 

    # read cells
    for key in eblk.keys():
        etype,eidx = eblk[key]._read_cells(f,point_gids)
        cells.append(CellBlock(etype, eidx))

    # read node set
    for key in nsblk.keys():
        ids,name = nsblk[key]._read_set(f)
        if name in nsets.keys(): raise ReadError('*E* PERMAS: Duplicated eset name')
        else: nsets[name] = ids
        
    # read element set
    for key in esblk.keys():
        ids,name = esblk[key]._read_set(f)
        if name in elsets.keys(): raise ReadError('*E* PERMAS: Duplicated nset name')
        else: elsets[name] = ids

    return Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=nsets,
        cell_sets=elsets,
    )
