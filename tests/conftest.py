import os
import time

import pytest

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import BruteForceIndex


def make_store(tmp_path, now=None, half_life=168.0, **extra_cfg):
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute",
                 half_life_hours=half_life, **extra_cfg)
    conn = dbmod.connect(tmp_path)
    embedder = HashingEmbedder()
    index = BruteForceIndex(tmp_path / "vectors.npz", embedder.dim)
    return MemoryStore(conn, embedder, index, cfg,
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
