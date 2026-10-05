"""Startup reconciliation self-heals crash windows between DB and index."""

import pytest
from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import open_vector_index


def _reopen_with_index(tmp_path, kind="brute"):
    """Reopen a home through the real open path.

    ADR-08 / W3-3 makes the index a per-tenant router, so a fixture that hands
    `MemoryStore` a bare backend would test a shape the server never constructs.
    """
    cfg = Config(home=tmp_path, embedder="hashing", vector_index=kind)
    conn = dbmod.connect(tmp_path)
    embedder = HashingEmbedder()
    count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    index = open_vector_index(tmp_path, kind, embedder.dim, count)
    return MemoryStore(conn, embedder, index, cfg)


def test_rebuilds_index_after_sidecar_loss(tmp_path):
    st = make_store(tmp_path)
    mid = st.remember("reconciliation probe entry")["id"]
    st.conn.close()  # simulate crash: DB row committed, sidecar never written
    st2 = _reopen_with_index(tmp_path)
    # AUDIT-I14: FTS-only recall must not hide an empty or stale vector index.
    assert st2.index.indexed_ids() == {mid}
    dense = st2.index.query(st2.embedder.encode(["reconciliation probe entry"])[0], 1, "default")
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
    st2 = _reopen_with_index(tmp_path)
    hits = st2.recall("doomed entry", k=5)
    assert all(h["id"] != dead for h in hits)
    assert any(h["id"] == keep for h in st2.recall("survivor entry", k=5))
    st2.close()


def test_hnsw_reconciliation(tmp_path):
    pytest.importorskip("hnswlib")

    st = make_store(tmp_path)
    mid = st.remember("hnsw reconcile probe")["id"]
    st.conn.close()
    st2 = _reopen_with_index(tmp_path, "hnsw")
    assert st2.index.kind == "hnsw"
    assert st2.index.indexed_ids() == {mid}
    dense = st2.index.query(st2.embedder.encode(["hnsw reconcile probe"])[0], 1, "default")
    assert dense[0][0] == mid
    assert dense[0][1] == pytest.approx(1.0, abs=1e-5)
    hits = st2.recall("hnsw reconcile probe", k=3)
    assert hits and hits[0]["id"] == mid
    st2.close()
