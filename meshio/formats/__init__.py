from . import (
    _gmsh, _xdmf, _abaqus, _ansys, _cgns, _dolfin, _exodus, _flac3d, _h5m, _mdpa, _med, _medit, _nastran, _obj, _off, _permas, _ply, _stl, _svg, _tetgen, _vtk, _vtu,
)
__all__ = [
"_gmsh", "_xdmf", "_abaqus", "_ansys", "_cgns", "_dolfin", "_exodus", "_flac3d", "_h5m", "_mdpa", "_med", "_medit", "_nastran", "_obj", "_off", "_permas", "_ply", "_stl", "_svg", "_tetgen", "_vtk", "_vtu",
]

for name in __all__:
    locals()[name].register()
