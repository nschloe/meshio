# meshio

[![Build Status](https://travis-ci.org/nschloe/meshio.svg?branch=master)](https://travis-ci.org/nschloe/meshio)
[![codecov.io](https://codecov.io/github/nschloe/meshio/branch/master/graphs/badge.svg)](https://codecov.io/github/nschloe/meshio/branch/master)
[![PyPi Version](https://img.shields.io/pypi/v/meshio.svg)](https://pypi.python.org/pypi/meshio)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshio.svg?style=social&label=Star&maxAge=2592000)](https://github.com/nschloe/meshio)

![](https://nschloe.github.io/meshio/pp.png)

There are various mesh formats available for representing unstructured meshes,
e.g.,

 * [ANSYS msh](http://www.afs.enea.it/fluent/Public/Fluent-Doc/PDF/chp03.pdf)
 * [DOLFIN XML](http://manpages.ubuntu.com/manpages/wily/man1/dolfin-convert.1.html)
 * [Exodus](https://cubit.sandia.gov/public/13.2/help_manual/WebHelp/finite_element_model/exodus/block_specification.htm)
 * [H5M](http://press3.mcs.anl.gov/sigma/moab/h5m-file-format)
 * [Medit](https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html)
 * [Gmsh](http://gmsh.info/doc/texinfo/gmsh.html#File-formats)
 * [OFF](http://segeval.cs.princeton.edu/public/off_format.html)
 * [PERMAS](http://www.intes.de)
 * [VTK](http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
 * [VTU](http://www.vtk.org/Wiki/VTK_XML_Formats)
 * [XDMF/XDMF3](http://www.xdmf.org/)

meshio can read and write all of these formats and smoothly converts between
them. Simply call
```
meshio-convert input.msh output.vtu
```
with any of the supported formats.

In Python, simply call
```python
points, cells, point_data, cell_data, field_data = \
    meshio.read(args.infile)
```
to read a mesh. To write, do
```python
meshio.write(
    args.outfile,
    points,
    cells,
    point_data=point_data,
    cell_data=cell_data,
    field_data=field_data
    )
```
For both input and output, you can optionally specify the exact `file_format`
(in case you would like to enforce binary over ASCII VTK, for example).

### Installation

meshio is [available from the Python Package
Index](https://pypi.python.org/pypi/meshio/), so simply type
```
pip install -U meshio
```
to install or upgrade.

### Usage

Just
```
import meshio
```
and make use of all the goodies the module provides.


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
