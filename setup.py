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
    "hdf5": ["h5py"],  # MED, MOAB, XDMF formats
}
extras["all"] = list(set(itertools.chain.from_iterable(extras.values())))


setup(
    name="meshio",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=find_packages(),
    description="I/O for various mesh formats",
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
    python_requires=">=3.6",
    extras_require=extras,
    classifiers=[
        about["__status__"],
        about["__license__"],
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    data_files=[("paraview-plugins", ["tools/paraview-meshio-plugin.py"])],
    entry_points={
        "console_scripts": [
            "meshio-convert = meshio._cli:convert",
            "meshio-info = meshio._cli:info",
        ]
    },
)
