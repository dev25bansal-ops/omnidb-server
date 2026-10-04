import pytest

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore, StoreError
from omnidb_server.vector_index import BruteForceIndex


def test_wal_mode(store):
    mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_synchronous_normal_and_busy_timeout(store):
    assert store.conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def _fts_hit(store, term):
    """The default tenant's FTS table is `fts_default` (ADR-08 / W3-3)."""
    return store.conn.execute(
        'SELECT rowid FROM "fts_default" WHERE "fts_default" MATCH ?',
        (term,),
    ).fetchone()


def test_fts_explicit_sync(store):
    """ADR-08 / W3-3: FTS is synchronised by the write path, not triggers.

    The triple of external-content triggers (`memories_ai/ad/au`) was removed.
    This pins the replacement: a memory written through the store API is
    searchable through FTS, which is the only contract that matters to the
    retrieval path.
    """
    store.remember("zebra migrations across the serengeti")
    assert _fts_hit(store, "serengeti") is not None


def test_fts_content_update_keeps_postings_consistent(store):
    """A content edit replaces the FTS posting, not appends a second one."""
    mid = store.remember("initial wording")["id"]
    store.update_memory(mid, content="revised phrasing")
    assert _fts_hit(store, "initial") is None
    assert _fts_hit(store, "revised") is not None


def test_meta_roundtrip(store):
    dbmod.set_meta(store.conn, "probe", "v42")
    assert dbmod.get_meta(store.conn, "probe") == "v42"
    dbmod.set_meta(store.conn, "probe", "v43")
    assert dbmod.get_meta(conn=store.conn, key="probe") == "v43"
    assert dbmod.get_meta(store.conn, "missing", "dflt") == "dflt"


def test_embedder_mismatch_rejected(tmp_path):
    from conftest import make_store

    st = make_store(tmp_path)
    st.remember("hello world")
    st.close()

    class WrongDim:
        name = "hashing-v1"
        dim = 384

        def encode(self, texts):
            raise AssertionError("should not be called")

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    index = BruteForceIndex(tmp_path / "vectors.npz", 384)
    with pytest.raises(StoreError):
        MemoryStore(conn, WrongDim(), index, cfg)
    conn.close()


def test_hnsw_roundtrip_and_reopen(tmp_path):
    pytest.importorskip("hnswlib")
    from conftest import make_store

    from omnidb_server.vector_index import HnswlibIndex

    emb = HashingEmbedder()
    st = make_store(tmp_path)
    # swap in an hnsw-backed store on the same database
    idx = HnswlibIndex(tmp_path / "index.hnsw", emb.dim)
    st.index = idx
    mid = st.remember("kubernetes ingress controller setup")["id"]
    st.close()

    reopened = HnswlibIndex(tmp_path / "index.hnsw", emb.dim, count_hint=10)
    assert reopened.count >= 1
    st2 = make_store(tmp_path)
    st2.index = reopened
    hits = st2.recall("kubernetes ingress", k=3)
    assert hits and hits[0]["id"] == mid
    st2.forget(mid)
    assert st2.recall("kubernetes ingress", k=3) == []
    st2.close()
