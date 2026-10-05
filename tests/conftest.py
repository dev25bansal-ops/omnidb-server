import os
import time

import pytest

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.db import get_meta
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import open_vector_index


def make_store(tmp_path, now=None, half_life=168.0, embedder="hashing",
               embedder_impl=None, **extra_cfg):
    conn = dbmod.connect(tmp_path)
    # Honour the home's stored backend rather than forcing brute. ADR-08 / W3-3
    # keeps a real guard against reopening a home on a backend it was not built
    # with, and a fixture that pinned "brute" would trip it on any home a test
    # had switched to hnsw.
    stored = get_meta(conn, "vec_kind")
    kind = stored if stored in ("hnsw", "brute") else "brute"
    cfg = Config(home=tmp_path, embedder=embedder, vector_index=kind,
                 half_life_hours=half_life, **extra_cfg)
    if embedder_impl is None:
        embedder_impl = HashingEmbedder()
    # Through open_vector_index, not a hand-built backend: the store's index is
    # a per-tenant router, and a fixture that constructed the backend directly
    # would test a shape the server never constructs.
    count_hint = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    index = open_vector_index(tmp_path, kind, embedder_impl.dim, count_hint)
    return MemoryStore(conn, embedder_impl, index, cfg,
                       now_fn=now if now is not None else time.time)


@pytest.fixture(autouse=True, scope="session")
def scrub_omnidb_environment():
    original = {name: os.environ.pop(name) for name in tuple(os.environ) if name.startswith("OMNIDB_")}
    yield
    os.environ.update(original)


@pytest.fixture()
def store(tmp_path):
    st = make_store(tmp_path)
    yield st
    st.close()


def make_index(tmp_path, kind="brute", dim=512, count_hint=0):
    """The home's index as production builds it.

    ADR-08 / W3-3 makes the index a per-tenant router, so a fixture that hands
    `MemoryStore` a bare `BruteForceIndex` would exercise a shape the server
    never constructs — and every store call site that passes a tenant would fail
    with a TypeError instead of testing the behaviour it means to test.
    """
    from omnidb_server.vector_index import open_vector_index as _open

    return _open(tmp_path, kind, dim, count_hint)


def make_store_at(tmp_path, kind="brute", dim=512, **cfg_kwargs):
    """A MemoryStore whose index came from the real open path."""
    cfg = Config(home=tmp_path, embedder="hashing", vector_index=kind,
                 **cfg_kwargs)
    conn = dbmod.connect(tmp_path)
    embedder = HashingEmbedder()
    count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    return MemoryStore(conn, embedder, make_index(tmp_path, kind, dim, count), cfg)
