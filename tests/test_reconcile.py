"""Startup reconciliation self-heals crash windows between DB and index."""

import pytest
from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import BruteForceIndex


def _reopen_with_index(tmp_path, index):
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    return MemoryStore(conn, HashingEmbedder(), index, cfg)


def test_rebuilds_index_after_sidecar_loss(tmp_path):
    st = make_store(tmp_path)
    mid = st.remember("reconciliation probe entry")["id"]
    st.conn.close()  # simulate crash: DB row committed, sidecar never written
    st2 = _reopen_with_index(tmp_path, BruteForceIndex(tmp_path / "vectors.npz", 512))
    # AUDIT-I14: FTS-only recall must not hide an empty or stale vector index.
    assert st2.index.indexed_ids() == {mid}
    dense = st2.index.query(st2.embedder.encode(["reconciliation probe entry"])[0], k=1)
    assert dense[0][0] == mid
    assert dense[0][1] == pytest.approx(1.0, abs=1e-5)
    hits = st2.recall("reconciliation probe", k=3)
    assert hits and hits[0]["id"] == mid  # reconcile re-embedded it
    st2.close()


def test_drops_index_entries_for_dead_rows(tmp_path):
    st = make_store(tmp_path)
    keep = st.remember("survivor entry")["id"]
    dead = st.remember("doomed entry")["id"]
    # crash window: row soft-deleted directly in DB, index not updated
    st.conn.execute("UPDATE memories SET deleted_ms = 1 WHERE id = ?", (dead,))
    st.conn.commit()
    st.conn.close()
    st2 = _reopen_with_index(tmp_path, BruteForceIndex.load(tmp_path / "vectors.npz", 512))
    hits = st2.recall("doomed entry", k=5)
    assert all(h["id"] != dead for h in hits)
    assert any(h["id"] == keep for h in st2.recall("survivor entry", k=5))
    st2.close()


def test_hnsw_reconciliation(tmp_path):
    pytest.importorskip("hnswlib")
    from omnidb_server.vector_index import HnswlibIndex

    st = make_store(tmp_path)
    mid = st.remember("hnsw reconcile probe")["id"]
    st.conn.close()
    st2 = _reopen_with_index(tmp_path, HnswlibIndex(tmp_path / "index.hnsw", 512))
    assert st2.index.kind == "hnsw"
    assert st2.index.indexed_ids() == {mid}
    dense = st2.index.query(st2.embedder.encode(["hnsw reconcile probe"])[0], k=1)
    assert dense[0][0] == mid
    assert dense[0][1] == pytest.approx(1.0, abs=1e-5)
    hits = st2.recall("hnsw reconcile probe", k=3)
    assert hits and hits[0]["id"] == mid
    st2.close()
