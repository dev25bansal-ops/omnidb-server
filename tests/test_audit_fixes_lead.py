"""Regressions for the audit fixes landed by the team lead (docs/AUDIT.md).

Covers I01, I02, I03, I04, I05, I06, I11, I13, I16, I19, I21 with bounded
synthetic fixtures and lifecycle doubles only — no exploit payloads, no
pathological regexes, no exhaustion workloads.
"""

import json
import sqlite3
import time

import numpy as np
import pytest
from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.server import _acquire_home_lock, _release_home_lock, open_owned_store
from omnidb_server.store import MemoryStore, StoreError
from omnidb_server.validation import MAX_TRIGGER_PATTERNS
from omnidb_server.vector_index import BruteForceIndex


def _reopen(tmp_path):
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(cfg.home)
    emb = HashingEmbedder()
    idx = BruteForceIndex.load(cfg.home / "vectors.npz", emb.dim)
    return MemoryStore(conn, emb, idx, cfg)


# --- I01: encoder failure leaves content unchanged; pending revision heals ----

def test_update_encode_failure_is_atomic(tmp_path):
    st = make_store(tmp_path)
    mid = st.remember("original content")["id"]
    st.flush()

    class Boom(HashingEmbedder):
        def encode(self, texts):
            raise RuntimeError("encoder exploded")

    st.embedder = Boom()
    with pytest.raises(RuntimeError, match="encoder exploded"):
        st.update_memory(mid, content="replacement content")
    st.embedder = HashingEmbedder()
    assert st.get(mid)["content"] == "original content"
    st.close()


def test_pending_revision_recovered_on_reopen(tmp_path):
    st = make_store(tmp_path)
    mid = st.remember("v1 content")["id"]
    st.update_memory(mid, content="v2 content")
    st.close()

    # simulate a crash after the content commit but before the sidecar
    # persist: the durable pending marker must survive and drive recovery
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    conn.execute("INSERT INTO meta(key,value) VALUES ('index_dirty','1') "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value")
    conn.execute("INSERT INTO meta(key,value) VALUES ('pending_reembed',?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (json.dumps([mid]),))
    conn.commit()
    conn.close()
    (tmp_path / "vectors.npz").unlink()

    st2 = _reopen(tmp_path)
    assert st2.get(mid)["content"] == "v2 content"
    vec = st2.embedder.encode(["v2 content"])[0]
    hits = st2.index.query(vec, k=1)
    assert hits and hits[0][0] == mid
    assert dbmod.get_meta(st2.conn, "pending_reembed") == "[]"
    st2.close()


# --- I02: ownership precedes writable init; read-only diagnostics coexist ----

def test_reembed_refused_before_writable_init(tmp_path):
    """AUDIT-I02: re-embed must check the home lock BEFORE connecting
    (connecting runs migrations/WAL pragmas), so a rejected migration leaves
    no database behind."""
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.portability import reembed_home

    home = tmp_path / "home"
    home.mkdir()
    fd = _acquire_home_lock(home)
    assert not isinstance(fd, OSError)
    try:
        with pytest.raises(StoreError, match="another process is using"):
            reembed_home(home, HashingEmbedder(), assume_yes=True)
    finally:
        _release_home_lock(fd)
    assert not (home / "omnidb.db").exists(), \
        "rejected re-embed must not run writable initialization"


def test_second_owned_opener_refused_without_writes(tmp_path):
    fd = _acquire_home_lock(tmp_path)
    assert not isinstance(fd, OSError)
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    with pytest.raises(StoreError, match="another process is using"):
        with open_owned_store(cfg):
            pass  # must never be reached
    # refusal happened before any mutable initialization: no DB was created
    assert not (tmp_path / "omnidb.db").exists()
    _release_home_lock(fd)


# --- I03: interrupted re-embed marker forces a consistent rebuild ------------

def test_reembed_pending_marker_rebuilds_on_open(tmp_path):
    st = make_store(tmp_path)
    mid = st.remember("some content")["id"]
    st.flush()
    set_meta = dbmod.set_meta
    set_meta(st.conn, "reembed_pending", json.dumps({"embedder": "other", "dim": 8}))
    # corrupt the sidecar with a foreign vector so a stale read is detectable
    foreign = np.ones((1, st.embedder.dim), dtype=np.float32)
    st.index.add([9999], foreign)
    st.close()

    st2 = _reopen(tmp_path)
    assert dbmod.get_meta(st2.conn, "reembed_pending") is None
    assert 9999 not in st2.index.indexed_ids()
    assert st2.index.indexed_ids() == {mid}
    st2.close()


# --- I04: invalid edge endpoint leaves the home unchanged after reopen -------

def test_import_rejection_is_atomic(tmp_path):
    src = make_store(tmp_path / "src")
    a = src.remember("alpha memory")["id"]
    src.link(a, a, label="self")
    dump = tmp_path / "dump.jsonl"
    from omnidb_server.portability import export_jsonl
    export_jsonl(src, dump)
    src.close()
    lines = dump.read_text(encoding="utf-8").splitlines()
    # append an edge referencing a memory id that was never exported
    lines.append(json.dumps({"type": "edge", "src": a, "dst": 424242,
                             "label": "ghost"}))
    (tmp_path / "bad.jsonl").write_text("\n".join(lines), encoding="utf-8")

    dst = make_store(tmp_path / "dst")
    from omnidb_server.portability import import_jsonl
    with pytest.raises(StoreError, match="unknown memory id"):
        import_jsonl(dst, tmp_path / "bad.jsonl")
    assert dst.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    dst.close()

    # a fresh reopen of the rejected home is still empty
    dst2 = make_store(tmp_path / "dst")
    assert dst2.stats()["live"] == 0
    dst2.close()


# --- I05: the query budget is VM instructions, enforced and cleaned up -------

def test_tiny_instruction_budget_interrupts(store):
    for i in range(5):
        store.remember(f"budget probe {i}")
    store.max_query_steps = 1  # one callback interval of 1000 instructions
    # a recursive generator that cannot finish inside 1000 VM instructions
    with pytest.raises(StoreError) as exc:
        store.search(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n "
            "WHERE i < 100000) SELECT count(*) FROM n")
    assert exc.value.code == "OMN-011"
    # handlers are cleaned up on the interrupt path: a normal query works
    store.max_query_steps = 5_000_000
    result = store.search("SELECT id FROM memories LIMIT 2")
    assert len(result["rows"]) == 2


def test_search_result_byte_cap(store):
    store.remember("x" * 500)
    import omnidb_server.store as store_mod
    original = store_mod._QUERY_RESULT_BYTES_LIMIT
    store_mod._QUERY_RESULT_BYTES_LIMIT = 100
    try:
        result = store.search("SELECT content FROM memories")
    finally:
        store_mod._QUERY_RESULT_BYTES_LIMIT = original
    assert result["truncated"] is True


# --- I06: trigger admission and matcher bounds -------------------------------

def test_trigger_admission_limits(store):
    store.cfg.max_content_chars = 100
    with pytest.raises(StoreError):
        store.remember_many([{"content": "ok", "kind": "procedural",
                              "trigger_patterns": ["a"] * (MAX_TRIGGER_PATTERNS + 1)}])
    with pytest.raises(StoreError):
        store.remember("ok", kind="procedural",
                       trigger_patterns=["x" * 600])
    assert store.stats()["live"] == 0


def test_oversized_query_skips_trigger_matching(store):
    store.remember("skill body", kind="procedural",
                   trigger_patterns=["needle"])
    hits = store.recall("needle " + "z" * 5000, k=5)
    assert not any(h.get("metadata", {}).get("skill_trigger") for h in hits)
    hits = store.recall("needle", k=5)
    assert any(h.get("metadata", {}).get("skill_trigger") for h in hits)


# --- I11: per-chunk provenance and agent scope -------------------------------

def test_consolidation_provenance_and_scope(tmp_path, monkeypatch):
    import omnidb_server.consolidate as consolidate

    st = make_store(tmp_path, llm_base_url="http://llm.local/v1",
                    llm_model="m", consolidation_chunk_episodes=2)
    for i in range(4):
        st.remember(f"episode {i} detail", kind="episodic", agent_id="alice")
    for i in range(4):
        st.remember(f"bob episode {i}", kind="episodic", agent_id="bob")

    calls = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            # every chunk returns one fact so the failing chunk is observable
            return {"choices": [{"message": {"content": json.dumps(
                [{"fact": f"extracted fact {len(calls)} about detail",
                  "confidence": 0.9}])}}]}

    def fake_post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("chunk failed")
        return R()

    monkeypatch.setattr(consolidate.httpx, "post", fake_post)
    report = consolidate.run_consolidation(st, hours=24)
    assert report["chunks_failed"] >= 1

    # episodes were created in order: alice ids 1..4, bob ids 5..8; with a
    # chunk size of 2 the chunks are [1,2], [3,4], [5,6], [7,8] and the
    # SECOND call (chunk [3,4]) is the one that failed.
    facts = [r for r in st.conn.execute(
        "SELECT content, metadata, agent_id FROM memories WHERE kind='semantic'")]
    assert facts, "successful chunks still commit"
    allowed = [{1, 2}, {5, 6}, {7, 8}]
    for row in facts:
        meta = json.loads(row["metadata"])
        assert meta["origin"] == "consolidation"
        # AUDIT-I11: provenance is exactly one successful chunk's episodes
        assert set(meta["source_episodes"]) in allowed
        assert row["agent_id"] in ("alice", "bob")
    st.close()


# --- I13: recovery fetches/encodes in bounded batches ------------------------

def test_reconcile_batches_fetch(tmp_path, monkeypatch):
    import omnidb_server.store as store_mod

    st = make_store(tmp_path)
    for i in range(6):
        st.remember(f"batched {i}")
    st.flush()
    st.close()
    (tmp_path / "vectors.npz").unlink()

    monkeypatch.setattr(store_mod, "_RECONCILE_BATCH_CAP", 2)

    class RecordingEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.batches: list[int] = []

        def encode(self, texts):
            self.batches.append(len(texts))
            return super().encode(texts)

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(cfg.home)
    emb = RecordingEmbedder()
    idx = BruteForceIndex(tmp_path / "vectors.npz", emb.dim)  # sidecar lost
    st2 = MemoryStore(conn, emb, idx, cfg)  # recovery runs during open
    assert st2.index.indexed_ids() == set(range(1, 7))
    assert emb.batches and max(emb.batches) <= 2  # bounded encode batches
    st2.close()


# --- I16: idle persistence deadline is enforced ------------------------------

def test_maintenance_tick_flushes_when_idle(tmp_path, monkeypatch):
    st = make_store(tmp_path, persist_every_ops=10**9, persist_max_s=0.05)
    persists = {"n": 0}
    orig = st.index.persist

    def counting():
        persists["n"] += 1
        orig()

    st.index.persist = counting
    st.remember("idle dirty write")
    assert persists["n"] == 0  # below the op threshold, not yet elapsed

    st._last_persist_mono = time.monotonic() - 0.1
    st.maintenance_tick()
    assert persists["n"] == 1

    st.maintenance_tick()  # already clean: no second flush
    assert persists["n"] == 1
    st.close()


# --- I19: missing computation uses one membership set ------------------------

def test_forget_many_missing_and_duplicates(store):
    a = store.remember("forget a")["id"]
    b = store.remember("forget b")["id"]
    result = store.forget_many([a, a, b, 999])
    assert result["forgotten"] == [a, b]
    assert result["missing"] == [999]


# --- I21: the content policy is one contract across write paths --------------

def test_content_limit_applies_to_update_and_import(tmp_path):
    st = make_store(tmp_path, max_content_chars=16)
    mid = st.remember("short")["id"]
    with pytest.raises(StoreError) as exc:
        st.update_memory(mid, content="x" * 17)
    assert exc.value.code == "OMN-005"
    assert st.get(mid)["content"] == "short"
    st.close()

    dump = tmp_path / "long.jsonl"
    dump.write_text(json.dumps({"type": "memory", "id": 1, "kind": "semantic",
                                "content": "y" * 17}) + "\n", encoding="utf-8")
    dst = make_store(tmp_path / "dst", max_content_chars=16)
    from omnidb_server.portability import import_jsonl
    with pytest.raises(StoreError, match="max_content_chars"):
        import_jsonl(dst, dump)
    assert dst.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    dst.close()
