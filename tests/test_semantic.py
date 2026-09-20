"""Opt-in semantic-recall validation with real embeddings (fastembed).

Downloads a model on first use, so it only runs when OMNIDB_TEST_SEMANTIC=1.
The query is a paraphrase sharing almost no surface tokens with the target,
which the hashing embedder cannot match — this gate proves real semantics.
"""

import os

import pytest

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import FastEmbedEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import BruteForceIndex


@pytest.mark.skipif(
    os.environ.get("OMNIDB_TEST_SEMANTIC") != "1",
    reason="downloads an embedding model; opt in with OMNIDB_TEST_SEMANTIC=1",
)
def test_fastembed_paraphrase_recall(tmp_path):
    embedder = FastEmbedEmbedder()  # BAAI/bge-small-en-v1.5, 384-dim
    cfg = Config(home=tmp_path, embedder="fastembed", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    store = MemoryStore(conn, embedder, BruteForceIndex(tmp_path / "vectors.npz", embedder.dim), cfg)

    store.remember("Python developers often use virtual environments to isolate project dependencies")
    store.remember("The chef prepared a delicious pasta dish garnished with fresh basil")
    store.remember("Quarterly revenue increased by twelve percent year over year")

    hits = store.recall("how do I keep my project dependencies separate from each other?", k=3)
    assert "virtual environments" in hits[0]["content"]

    # embedder identity pinning: reopening with a different embedder must fail
    from conftest import make_store
    with pytest.raises(Exception, match="embedder"):
        make_store(tmp_path)
    store.close()
