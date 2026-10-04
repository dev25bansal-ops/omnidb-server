"""F13 - the read path must distinguish an unusable index from a bug in our code.

`open_read_only_store` caught bare `Exception` around `open_vector_index` and fell
back to brute force with no output. Two consequences: a TypeError raised inside the
index layer was laundered into a silent downgrade, and the operator saw a store that
looked healthy while it answered from a different backend than configured.

The catch is now narrowed to `VectorIndexError`, the index layer own signal, and the
fallback says why on stderr - which is what ARCH-04 already did one layer down.
"""
from __future__ import annotations

import sys
import types

import pytest
from conftest import make_store

import omnidb_server.server as srv
from omnidb_server.vector_index import (
    BruteForceIndex,
    HnswlibIndex,
    VectorIndexError,
)


def _read_cfg(home):
    from omnidb_server.config import Config
    return Config(home=home, embedder="hashing", vector_index="brute")


@pytest.fixture()
def home(tmp_path):
    """A real home, so open_read_only_store gets past its db-exists check."""
    st = make_store(tmp_path)
    st.remember("a row so the index has something to open", kind="semantic")
    st.close()
    return tmp_path


def test_a_programming_error_propagates(home, monkeypatch):
    """THE F13 REGRESSION. The old bare except Exception swallowed this and
    handed back a working-looking store. A bug must not become a fallback."""

    def boom(*a, **k):
        raise TypeError("open_vector_index() got an unexpected keyword argument")

    monkeypatch.setattr("omnidb_server.vector_index.open_vector_index", boom)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        srv.open_read_only_store(_read_cfg(home))


def test_an_unusable_index_still_falls_back_and_says_why(home, monkeypatch, capsys):
    """The ARCH-04 behaviour is preserved: the fallback stays, but silent is not."""

    def unusable(*a, **k):
        raise VectorIndexError("this home was built with the hnsw backend")

    monkeypatch.setattr("omnidb_server.vector_index.open_vector_index", unusable)
    store = srv.open_read_only_store(_read_cfg(home))
    try:
        assert isinstance(store.index, BruteForceIndex)
    finally:
        store.close()
    err = capsys.readouterr().err
    assert "falling back to brute force" in err
    assert "built with the hnsw backend" in err, "the reason must survive"


def test_a_healthy_open_says_nothing(home, capsys):
    """No failure, no stderr - the warning must not become noise."""
    store = srv.open_read_only_store(_read_cfg(home))
    try:
        pass
    finally:
        store.close()
    assert "falling back" not in capsys.readouterr().err


# --- F31: real corruption must still fall back, not crash ---------------------

class _FakeHnswIndex:
    """Minimal stand-in. The real hnswlib raises RuntimeError on a corrupt file;
    faking it is the only way to test this without the optional dependency."""
    def __init__(self, **kwargs):
        pass

    def load_index(self, path, max_elements=None):
        raise RuntimeError("Wrong file format or corrupted index")

    def init_index(self, **kwargs):
        pass

    def set_ef(self, n):
        pass

@pytest.fixture()
def fake_hnswlib(monkeypatch):
    mod = types.ModuleType("hnswlib")
    mod.Index = _FakeHnswIndex
    monkeypatch.setitem(sys.modules, "hnswlib", mod)
    return mod

def test_a_corrupt_sidecar_becomes_a_vector_index_error(tmp_path, fake_hnswlib):
    """F31 REGRESSION. hnswlib raises RuntimeError on a corrupt file, which is NOT
    a VectorIndexError, so the F13 narrowing leaked it and the open crashed
    instead of falling back - the opposite of what ARCH-04 asked for."""
    side = tmp_path / "index.hnsw"
    side.write_bytes(b"not an index at all")

    with pytest.raises(VectorIndexError):
        HnswlibIndex(side, dim=512)

def test_a_corrupt_tombstone_sidecar_becomes_a_vector_index_error(
        tmp_path, fake_hnswlib):
    """The other corruption path: json.loads raises ValueError on a bad sidecar."""
    side = tmp_path / "i2.hnsw"
    side.write_bytes(b"x")
    (tmp_path / "i2.hnsw.deleted.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(VectorIndexError):
        HnswlibIndex(side, dim=512)