# MeshIO

[![Build Status](https://travis-ci.org/nschloe/meshio.svg?branch=master)](https://travis-ci.org/nschloe/meshio)
[![Code Health](https://landscape.io/github/nschloe/meshio/master/landscape.png)](https://landscape.io/github/nschloe/meshio/master)
[![codecov.io](https://codecov.io/github/nschloe/meshio/coverage.svg?branch=master)](https://codecov.io/github/nschloe/meshio?branch=master)
[![PyPi Version](https://img.shields.io/pypi/v/meshio.svg)](https://pypi.python.org/pypi/meshio)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/meshio.svg?style=social&label=Star&maxAge=2592000)](https://github.com/nschloe/meshio)

![](https://nschloe.github.io/meshio/pp.png)

There are various mesh formats available for representing unstructured meshes,
e.g.,

 * [DOLFIN XML](http://manpages.ubuntu.com/manpages/wily/man1/dolfin-convert.1.html)
 * [Exodus](https://cubit.sandia.gov/public/13.2/help_manual/WebHelp/finite_element_model/exodus/block_specification.htm)
 * [H5M](https://trac.mcs.anl.gov/projects/ITAPS/wiki/MOAB/h5m)
 * [MSH](http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats)
 * [PERMAS](http://www.intes.de)
 * [VTK](http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
 * [VTU](http://www.vtk.org/Wiki/VTK_XML_Formats)
 * [XDMF](http://www.xdmf.org/)

MeshIO can read and write all of these formats and smoothly converts between
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

#### Python Package Index

MeshIO is [available from the Python Package
Index](https://pypi.python.org/pypi/meshio/), so simply type
```
pip install meshio
```
to install or
```
pip install meshio -U
```
to upgrade.

#### Manual installation

Download MeshIO from [PyPi](https://pypi.python.org/pypi/meshio/)
or [GitHub](https://github.com/nschloe/meshio) and
install it with
```
python setup.py install
```

### Requirements

MeshIO depends on

 * [h5py](http://www.h5py.org/),
 * [NumPy](http://www.numpy.org/), and
 * [VTK](http://www.vtk.org/Wiki/VTK/Examples/Python).

### Usage

Just
```
import meshio
```
and make use of all the goodies the module provides.


### Testing

To run the MeshIO unit tests, check out this repository and type
```
nosetests
```
or
```
nose2 -s test
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. create a Git tag,
    ```
    git tag v0.3.1
    git push --tags
    ```
    and

3. upload to PyPi:
    ```
    make upload
    ```


### License

MeshIO is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
