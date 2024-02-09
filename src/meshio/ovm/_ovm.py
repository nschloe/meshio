"""
I/O for OpenVolumeMesh <https://openvolumemesh.org/> native .ovm file format.

Original author: Martin Heistermann <martin.heistermann@inf.unibe.ch>

This I/O module is a bit more complex than for many other file formats,
as the OVM format does not describe volumetric cells as tuple of vertices
for specific polyhedral types, but only describes general polyhedra in terms
of "half-faces" (oriented faces), which in turn are described by half-edges,
while edges finally consist of vertex indices.

There are certain impedance mismatches we have to handle:

    - In OVM the n-1-dimensional entities that make up an n-cell are always
      explicitly represented, a roundtrip conversion meshio->ovm->meshio
      might end up with additional cells

    - Order of vertices of a cell is not uniquely represented - we might end
      up with a different order after a roundtrip conversion.
      For edges and faces we try to preserve ordering inside of edges and faces
      at a slight runtime and memory cost (we could perform more efficient caching
      by always storing a known canonical orientation).

      For volumetric cells this could also be achieved by selecting a convention,
      however currently this is not guaranteed.

"""

from .._exceptions import WriteError
from .._files import open_file
from .._helpers import register_format

from .ovm_ascii import ovm_ascii_read, ovm_ascii_write
from .ovm_mesh import OpenVolumeMesh


def read(filename):
    with open_file(filename) as f:
        return ovm_ascii_read(f)


def write(filename, mesh, float_fmt=".15e", binary=False):
    if binary:
        raise WriteError("OVM binary format currently not implemented in meshio")
    ovm = OpenVolumeMesh.from_meshio(mesh)
    with open_file(filename, "w") as fh:
        ovm_ascii_write(ovm, fh, float_fmt=float_fmt)


register_format("ovm", [".ovm"], read, {"ovm": write})
