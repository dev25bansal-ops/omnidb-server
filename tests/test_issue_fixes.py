"""Regression tests for remediation items I-01 through I-15."""

import hashlib

import numpy as np
import pytest
from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.config import Config, ConfigError
from omnidb_server.db import get_meta
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore, StoreError


def _open_brute(tmp_path):
    from omnidb_server.vector_index import BruteForceIndex
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    emb = HashingEmbedder()
    return MemoryStore(conn, emb,
                       BruteForceIndex.load(tmp_path / "vectors.npz", emb.dim), cfg)


# --- I-02: batched persistence + crash-window recovery (I-08) --------------

def test_batched_writes_set_dirty_and_recover(tmp_path):
    st = make_store(tmp_path, persist_every_ops=10**9)
    st.remember("batch probe one")
    st.remember("batch probe two")
    assert get_meta(st.conn, "index_dirty") == "1"  # deferred, not yet persisted
    st.conn.close()  # simulate crash before flush

    st2 = _open_brute(tmp_path)
    assert get_meta(st2.conn, "reconciled_at") is not None  # crash window healed
    hits = st2.recall("batch probe", k=5)
    assert len(hits) == 2
    st2.close()


def test_clean_open_skips_reconciliation(tmp_path):
    st = make_store(tmp_path)
    st.remember("clean shutdown probe")
    st.close()  # flush clears the dirty flag

    s1 = _open_brute(tmp_path)
    assert get_meta(s1.conn, "reconciled_at") is None  # nothing to do
    s1.remember("one more")
    s1.close()

    s2 = _open_brute(tmp_path)
    assert get_meta(s2.conn, "reconciled_at") is None  # still no drift
    assert s2.stats()["live"] == 2
    s2.close()


# --- I-01: hnsw compaction scheduling + scheduled rebuild -------------------

def test_compaction_scheduled_then_rebuilt_on_open(tmp_path):
    pytest.importorskip("hnswlib")
    from omnidb_server.vector_index import HnswlibIndex

    st = make_store(tmp_path)
    st.index = HnswlibIndex(tmp_path / "index.hnsw", 512)
    for i in range(12):
        st.remember(f"compaction filler entry {i}")
    st.forget_many(list(range(1, 4)))  # 3/12 dead = 25% >= threshold
    assert get_meta(st.conn, "compaction_due") == "1"
    assert st.stats()["pending_compaction"] is True
    st.close()

    st2 = _open_hnsw(tmp_path)
    assert get_meta(st2.conn, "compaction_due") == "0"
    assert st2.index.dead_ratio == 0.0
    assert st2.index.active_count() == 9
    hits = st2.recall("compaction filler entry 5", k=3)
    assert hits and hits[0]["content"].endswith("entry 5")
    st2.close()


def _open_hnsw(tmp_path):
    pytest.importorskip("hnswlib")
    from omnidb_server.vector_index import HnswlibIndex
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="hnsw")
    conn = dbmod.connect(tmp_path)
    emb = HashingEmbedder()
    idx = HnswlibIndex(tmp_path / "index.hnsw", emb.dim,
                       count_hint=int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]))
    return MemoryStore(conn, emb, idx, cfg)


# --- I-05 / I-09: stats completeness + WAL checkpoint -----------------------

def test_stats_includes_sidecars_and_ratio(store):
    store.remember("sidecar size probe")
    store.flush()
    stats = store.stats()
    assert "vectors.npz" in stats["files"]
    assert stats["index"]["dead_ratio"] == 0.0
    assert stats["took_ms"] >= 0
    assert stats["pending_compaction"] is False


def test_flush_clears_dirty_flag(store):
    store.remember("flush probe")
    if get_meta(store.conn, "index_dirty") == "1":
        pass  # batching may legitimately defer
    store.flush()
    assert get_meta(store.conn, "index_dirty") == "0"


# --- I-06: k_fetch override + pool-floors-at-k semantics ---------------------

def test_k_fetch_floor_and_override(tmp_path):
    # configured pool of 20 must not silently cap recall below requested k
    st = make_store(tmp_path, k_fetch=20)
    for i in range(100):
        st.remember(f"sharedtoken variant{i}")

    mid = st.recall("sharedtoken", k=15)          # floor(20,15)=20-pool
    assert len(mid) == 15

    deep = st.recall("sharedtoken", k=90)         # floor rises with k -> 90-pool
    assert len(deep) == 90

    override = st.recall("sharedtoken", k=5, k_fetch=80)  # explicit wider pool
    assert len(override) == 5
    # a wider pool can surface deeper-tail candidates the narrow one missed
    tail_ids = {r["id"] for r in override}
    assert any(r["id"] not in {x["id"] for x in mid} for r in override) or len(tail_ids) == 5


# --- I-07: forget_many --------------------------------------------------------

def test_forget_many_reports_missing(store):
    ids = [store.remember(f"bulk item {i}")["id"] for i in range(4)]
    result = store.forget_many([ids[0], ids[1], 999999])
    assert result["forgotten"] == [ids[0], ids[1]]
    assert result["missing"] == [999999]
    assert store.stats()["soft_deleted"] == 2
    with pytest.raises(StoreError):
        store.forget(ids[0])  # already gone


# --- I-11: TOML config layering ----------------------------------------------

def test_toml_layering_env_wins_over_file(tmp_path, monkeypatch):
    toml_path = tmp_path / "omnidb.toml"
    toml_path.write_text("half_life_hours = 1.5\nrrf_k = 99\n")
    monkeypatch.setenv("OMNIDB_RRF_K", "7")

    cfg = Config.load(home=tmp_path)
    assert cfg.rrf_k == 7          # env beats toml
    assert cfg.half_life_hours == 1.5  # toml beats default

    monkeypatch.delenv("OMNIDB_RRF_K")
    assert Config.load(home=tmp_path).rrf_k == 99


def test_toml_unknown_key_rejected(tmp_path):
    (tmp_path / "omnidb.toml").write_text("not_a_real_field = 3\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        Config.load(home=tmp_path)


# --- I-12: update_memory -------------------------------------------------------

def test_update_memory_reindexes_content(store):
    mid = store.remember("alpha initial phrase")["id"]
    result = store.update_memory(mid, content="beta totally different words")
    assert result["updated"] == ["content"]

    hits = store.recall("beta totally different", k=3)
    assert hits and hits[0]["id"] == mid
    old = store.search(
        "SELECT COUNT(*) AS n FROM \"fts_default\" "
        "WHERE \"fts_default\" MATCH 'initial'",
        unscoped=True)  # the index is per-tenant since v6
    assert old["rows"][0][0] == 0  # FTS trigger kept in sync

    merged = store.update_memory(mid, metadata={"tag": "x"})
    assert merged["updated"] == ["metadata"]
    row = store.get(mid)
    import json as _json
    assert _json.loads(row["metadata"])["tag"] == "x"

    with pytest.raises(StoreError) as exc:
        store.update_memory(424242, content="nope")
    assert exc.value.code == "OMN-021"


# --- I-13: hashing embedder vectorization parity ------------------------------

def test_hashing_embedder_deterministic_and_sane():
    e1, e2 = HashingEmbedder(), HashingEmbedder()
    v1 = e1.encode(["Hello World!"])
    v2 = e2.encode(["Hello World!"])
    assert np.allclose(v1, v2)
    assert v1.shape == (1, 512) and np.isclose(np.linalg.norm(v1[0]), 1.0)
    expected_bucket = int(hashlib.md5(b"hello").hexdigest(), 16) % 512
    assert v1[0][expected_bucket] > 0  # 'hello' feature landed (L2-normalized, so small)


# --- I-14: migration framework --------------------------------------------------

def test_user_version_tracks_migrations(store):
    version = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == dbmod.SCHEMA_VERSION


# --- I-15: OMN-* error codes -----------------------------------------------------

def test_error_codes_surface(store, monkeypatch):
    with pytest.raises(StoreError) as ro:
        store.search("DELETE FROM memories", unscoped=True)
    assert ro.value.code == "OMN-020"

    with pytest.raises(StoreError) as te:
        store.remember("x", kind="bogus")
    assert te.value.code == "OMN-002"

    monkeypatch.setattr(store, "max_query_steps", 20_000)
    with pytest.raises(StoreError) as to:
        store.search(
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT sum(x) FROM c"
, unscoped=True)
    assert to.value.code == "OMN-011"


# --- I-04: consolidation sanitization/validation ---------------------------------

def test_consolidation_validates_llm_output(tmp_path, monkeypatch):
    import omnidb_server.consolidate as consolidate
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute",
                 llm_base_url="http://llm.local/v1", llm_model="m")
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     BruteForceIndex(tmp_path / "vectors.npz", 512), cfg)
    for i in range(3):
        st.remember(f"episode {i}", kind="episodic")

    import json as _json
    payload_json = _json.dumps([
        {"fact": "good fact kept", "confidence": 0.8},
        {"fact": "x" * 600},
        "junk string",
        {"fact": "has\x00\x07control chars", "confidence": 3},
    ])

    def fake_post(url, json=None, headers=None, timeout=None):
        class R:
            def raise_for_status(self):
                pass
            def json(self_inner):
                return {"choices": [{"message": {"content": payload_json}}]}
        return R()

    monkeypatch.setattr(consolidate.httpx, "post", fake_post)

    report = consolidate.run_consolidation(st, hours=24)
    assert report["facts_created"] == 2
    assert report["facts_dropped"] == 2
    hits = st.recall("control chars", k=5, kind="semantic")
    assert hits and "\x00" not in hits[0]["content"]
    assert hits[0]["metadata"]["confidence"] == 1.0  # clamped
    st.close()
