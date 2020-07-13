import sys
from pathlib import Path

import helpers
import pytest

import meshio

OBJ_PATH = helpers.MESHES_DIR / "obj" / "elephav.obj"


def test_read_str():
    meshio.read(str(OBJ_PATH))


def test_read_pathlike():
    meshio.read(OBJ_PATH)


def test_read_buffer():
    with open(str(OBJ_PATH)) as f:
        meshio.read(f, "obj")


@pytest.fixture
def mesh():
    return meshio.read(OBJ_PATH)


def test_write_str(mesh, tmpdir):
    tmp_path = str(tmpdir.join("tmp.obj"))
    meshio.write(tmp_path, mesh)
    assert Path(tmp_path).is_file()


@pytest.mark.skipif(sys.version_info < (3, 6), reason="Fails with 3.5")
def test_write_pathlike(mesh, tmpdir):
    tmp_path = Path(tmpdir.join("tmp.obj"))
    meshio.write(tmp_path, mesh)
    assert Path(tmp_path).is_file()


def test_write_buffer(mesh, tmpdir):
    tmp_path = str(tmpdir.join("tmp.obj"))
    with open(tmp_path, "w") as f:
        meshio.write(f, mesh, "obj")
    assert Path(tmp_path).is_file()
