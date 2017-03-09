# -*- coding: utf-8 -*-
#
import os
from distutils.core import setup
import codecs

# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, 'meshio', '__about__.py')) as f:
    exec(f.read(), about)


def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content


setup(
    name='meshio',
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=['meshio'],
    description='I/O for various mesh formats',
    long_description=read('README.rst'),
    url='https://github.com/nschloe/meshio',
    download_url='https://pypi.python.org/pypi/meshio',
    license=about['__license__'],
    platforms='any',
    install_requires=[
        'numpy',
        'pipdated',
        ],
    extras_require={
        'dolfin': ['lxml'],  # Dolfin's legacy format
        'h5m': ['h5py'],  # MOAB's format
        },
    scripts=[
      'tools/meshio-convert',
      ],
    classifiers=[
        about['__status__'],
        about['__license__'],
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
        ]
    )
