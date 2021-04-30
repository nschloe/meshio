from pathlib import Path

import pytest

import meshio

OBJ_PATH = Path(__file__).resolve().parent / "meshes" / "ply" / "bun_zipper_res4.ply"


def test_read_str():
    meshio.read(str(OBJ_PATH))


def test_read_pathlike():
    meshio.read(OBJ_PATH)


@pytest.mark.skip
def test_read_buffer():
    with open(str(OBJ_PATH)) as f:
        meshio.read(f, "ply")


@pytest.fixture
def mesh():
    return meshio.read(OBJ_PATH)


def test_write_str(mesh, tmpdir):
    tmp_path = str(tmpdir.join("tmp.ply"))
    meshio.write(tmp_path, mesh)
    assert Path(tmp_path).is_file()


def test_write_pathlike(mesh, tmpdir):
    tmp_path = Path(tmpdir.join("tmp.ply"))
    meshio.write(tmp_path, mesh)
    assert Path(tmp_path).is_file()


@pytest.mark.skip
def test_write_buffer(mesh, tmpdir):
    tmp_path = str(tmpdir.join("tmp.ply"))
    with open(tmp_path, "w") as f:
        meshio.write(f, mesh, "ply")
    assert Path(tmp_path).is_file()
