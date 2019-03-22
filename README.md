# meshio

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/meshio/master.svg)](https://circleci.com/gh/nschloe/meshio)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/meshio.svg)](https://codecov.io/gh/nschloe/meshio)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/meshio.svg)](https://pypi.org/project/meshio)
[![Debian CI](https://badges.debian.net/badges/debian/testing/python3-meshio/version.svg)](https://tracker.debian.org/pkg/python-meshio)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1173115.svg)](https://doi.org/10.5281/zenodo.1173115)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshio.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/meshio)
[![PyPi downloads](https://img.shields.io/pypi/dd/meshio.svg)](https://pypistats.org/packages/meshio)

<p align="center">
  <img src="https://nschloe.github.io/meshio/meshio_logo.svg" width="20%">
</p>

There are various mesh formats available for representing unstructured meshes.
meshio can read and write all of the following and smoothly converts between them:

 * [Abaqus](http://abaqus.software.polimi.it/v6.14/index.html)
 * [ANSYS msh](http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf)
 * [DOLFIN XML](http://manpages.ubuntu.com/manpages/wily/man1/dolfin-convert.1.html)
 * [Exodus](https://cubit.sandia.gov/public/13.2/help_manual/WebHelp/finite_element_model/exodus/block_specification.htm)
 * [H5M](https://www.mcs.anl.gov/~fathom/moab-docs/h5mmain.html)
 * [Kratos/MDPA](https://github.com/KratosMultiphysics/Kratos/wiki/Input-data)
 * [Medit](https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html)
 * [MED/Salome](https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html)
 * [Gmsh](http://gmsh.info/doc/texinfo/gmsh.html#File-formats) (versions 2 and 4)
 * [OFF](http://segeval.cs.princeton.edu/public/off_format.html)
 * [PERMAS](http://www.intes.de)
 * [STL](https://en.wikipedia.org/wiki/STL_(file_format))
 * [SVG](https://www.w3.org/TR/SVG/) (2D only, output only)
 * [VTK](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
 * [VTU](https://www.vtk.org/Wiki/VTK_XML_Formats)
 * [XDMF](http://www.xdmf.org/index.php/XDMF_Model_and_Format)

Simply call
```
meshio-convert input.msh output.vtu
```
with any of the supported formats.

In Python, simply call
```python
import meshio

mesh = meshio.read(filename)
# mesh.points, mesh.cells, ...
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
meshio.write("foo.vtk", mesh)
```
For both input and output, you can optionally specify the exact `file_format`
(in case you would like to enforce binary over ASCII VTK, for example).

#### Time series

The [XDMF format](http://www.xdmf.org/index.php/XDMF_Model_and_Format) supports time
series with a shared mesh. You can write times series data using meshio with
```python
writer = meshio.XdmfTimeSeriesWriter(filename)
writer.write_points_cells(points, cells)
for t in [0.0, 0.1, 0.21]:
    writer.write_data(t, point_data={"phi": data})
```
and read it with
```python
reader = meshio.XdmfTimeSeriesReader(filename)
points, cells = reader.read_points_cells()
for k in range(reader.num_steps):
    t, point_data, cell_data = reader.read_data(k)
```

### Installation

meshio is [available from the Python Package
Index](https://pypi.org/project/meshio/), so simply type
```
pip install -U meshio
```
to install or upgrade.


### Testing

To run the meshio unit tests, check out this repository and type
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. tag and upload to PyPi:
    ```
    make publish
    ```

### License

meshio is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
