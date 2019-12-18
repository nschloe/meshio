<p align="center">
  <a href="https://github.com/nschloe/meshio"><img alt="meshio" src="https://nschloe.github.io/meshio/logo-with-text.svg" width="60%"></a>
  <p align="center">I/O for mesh files.</p>
</p>

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/meshio/master.svg?style=flat-square)](https://circleci.com/gh/nschloe/meshio/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/meshio.svg?style=flat-square)](https://codecov.io/gh/nschloe/meshio)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![PyPi Version](https://img.shields.io/pypi/v/meshio.svg?style=flat-square)](https://pypi.org/project/meshio)
[![Anaconda Cloud](https://anaconda.org/conda-forge/meshio/badges/version.svg?=style=flat-square)](https://anaconda.org/conda-forge/meshio/)
[![Debian CI](https://badges.debian.net/badges/debian/testing/python3-meshio/version.svg?style=flat-square)](https://tracker.debian.org/pkg/python-meshio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1173115.svg?style=flat-square)](https://doi.org/10.5281/zenodo.1173115)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshio.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/meshio)
[![PyPi downloads](https://img.shields.io/pypi/dm/meshio.svg?style=flat-square)](https://pypistats.org/packages/meshio)

There are various mesh formats available for representing unstructured meshes.
meshio can read and write all of the following and smoothly converts between them:

 * [Abaqus](http://abaqus.software.polimi.it/v6.14/index.html)
 * [ANSYS msh](https://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf)
 * [CGNS](https://cgns.github.io/)
 * [DOLFIN XML](https://manpages.ubuntu.com/manpages/disco/man1/dolfin-convert.1.html)
 * [Exodus](https://cubit.sandia.gov/public/13.2/help_manual/WebHelp/finite_element_model/exodus/block_specification.htm)
 * [FLAC3D](https://www.itascacg.com/software/flac3d)
 * [H5M](https://www.mcs.anl.gov/~fathom/moab-docs/h5mmain.html)
 * [Kratos/MDPA](https://github.com/KratosMultiphysics/Kratos/wiki/Input-data)
 * [Medit](https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html)
 * [MED/Salome](https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html)
 * [Nastran](https://help.autodesk.com/view/NSTRN/2019/ENU/?guid=GUID-42B54ACB-FBE3-47CA-B8FE-475E7AD91A00) (bulk data)
 * [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed#mesh-representation-of-segmented-object-surfaces)
 * [Gmsh](http://gmsh.info/doc/texinfo/gmsh.html#File-formats) (versions 2 and 4)
 * [OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
 * [OFF](https://segeval.cs.princeton.edu/public/off_format.html)
 * [PERMAS](https://www.intes.de)
 * [PLY](https://en.wikipedia.org/wiki/PLY_(file_format))
 * [STL](https://en.wikipedia.org/wiki/STL_(file_format))
 * [TetGen .node/.ele](https://wias-berlin.de/software/tetgen/fformats.html)
 * [SVG](https://www.w3.org/TR/SVG/) (2D only, output only)
 * [UGRID](http://www.simcenter.msstate.edu/software/downloads/doc/ug_io/3d_grid_file_type_ugrid.html)
 * [VTK](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
 * [VTU](https://www.vtk.org/Wiki/VTK_XML_Formats) ([not raw binary data](https://github.com/nschloe/meshio/wiki/VTU-with-raw-binary-data))
 * [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) ([TIN](https://en.wikipedia.org/wiki/Triangulated_irregular_network))
 * [XDMF](http://www.xdmf.org/index.php/XDMF_Model_and_Format)

Install with
```
pip3 install meshio[all] --user
```
and simply call
```
meshio-convert input.msh output.vtu
```
with any of the supported formats.

In Python, simply do
```python
import meshio

mesh = meshio.read(
    filename,  # string, os.PathLike, or a buffer/open file
    file_format="stl"  # optional if filename is a path; inferred from extension
)
# mesh.points, mesh.cells, ...

# mesh.vtk.read() is also possible
```
to read a mesh. To write, do
```python
points = numpy.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    ])
cells = {
    "triangle": numpy.array([
        [0, 1, 2]
        ])
    }
meshio.write_points_cells(
    "foo.vtk",
    points,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    # cell_data=cell_data,
    # field_data=field_data
    )
```
or explicitly create a mesh object for writing
```python
mesh = meshio.Mesh(points, cells)
meshio.write(
    "foo.vtk",  # str, os.PathLike, or buffer/ open file
    mesh,
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

# mesh.vtk.write() is also possible
```
For both input and output, you can optionally specify the exact `file_format`
(in case you would like to enforce ASCII over binary VTK, for example).

Reading and writing can also be handled directly by the `Mesh` object:
```python
m = meshio.Mesh.read(filename, "vtk")  # same arguments as meshio.read
m.write("foo.vtk")  # same arguments as meshio.write, besides `mesh`
```

#### Time series

The [XDMF format](http://www.xdmf.org/index.php/XDMF_Model_and_Format) supports time
series with a shared mesh. You can write times series data using meshio with
```python
with meshio.xdmf.TimeSeriesWriter(filename) as writer:
    writer.write_points_cells(points, cells)
    for t in [0.0, 0.1, 0.21]:
        writer.write_data(t, point_data={"phi": data})
```
and read it with
```python
with meshio.xdmf.TimeSeriesReader(filename) as reader:
    points, cells = reader.read_points_cells()
    for k in range(reader.num_steps):
        t, point_data, cell_data = reader.read_data(k)
```

### Performance comparison

The comparisons here are for a tetrahedral mesh with about 400k points and 2M
tetrahedra.  The red lines mark the size of the mesh in memory.

#### File sizes

<img alt="file size" src="https://nschloe.github.io/meshio/filesizes.svg" width="60%">

#### I/O speed

<img alt="performance" src="https://nschloe.github.io/meshio/performance.svg" width="90%">

#### Maximum memory usage

<img alt="memory usage" src="https://nschloe.github.io/meshio/memory.svg" width="90%">


### Installation

meshio is [available from the Python Package Index](https://pypi.org/project/meshio/),
so simply do
```
pip3 install meshio --user
```
to install.

Additional dependencies (`netcdf4`, `h5py` and `lxml`) are required for some of the
output formats and can be pulled in by
```
pip install meshio[all] --user
```

You can also install meshio from [anaconda](https://anaconda.org/conda-forge/meshio):
```
conda install -c conda-forge meshio
```

### Testing

To run the meshio unit tests, check out this repository and type
```
pytest
```

### License

meshio is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
