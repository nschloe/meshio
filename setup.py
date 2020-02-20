import itertools
import os

from setuptools import find_packages, setup

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "meshio", "__about__.py"), "rb") as f:
    exec(f.read(), about)


extras = {
    "exodus": ["netCDF4"],
    "hdf5": ["h5py"],  # CGNS, H5M, MED, XDMF formats
}
extras["all"] = list(set(itertools.chain.from_iterable(extras.values())))


setup(
    name="meshio",
    version=about["__version__"],
    author=about["__original_author__"],
    author_email=about["__original_author_email__"],
    packages=find_packages(),
    description="I/O for many mesh formats",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nschloe/meshio",
    project_urls={
        "Code": "https://github.com/nschloe/meshio",
        "Issue tracker": "https://github.com/nschloe/meshio/issues",
    },
    license=about["__license__"],
    platforms="any",
    install_requires=["numpy"],
    # For pathlib >= 3.5, f-strings 3.6:
    python_requires=">=3.5",
    extras_require=extras,
    classifiers=[
        about["__status__"],
        about["__license__"],
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    data_files=[("paraview-plugins", ["tools/paraview-meshio-plugin.py"])],
    entry_points={
        "console_scripts": [
            "meshio-ascii = meshio._cli:ascii",
            "meshio-binary = meshio._cli:binary",
            "meshio-compress = meshio._cli:compress",
            "meshio-convert = meshio._cli:convert",
            "meshio-decompress = meshio._cli:decompress",
            "meshio-info = meshio._cli:info",
        ]
    },
)
