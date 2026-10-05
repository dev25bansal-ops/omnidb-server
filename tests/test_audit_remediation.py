"""Regression tests for the Sep 2026 audit remediation (BUG-01 .. BUG-09).

Each test pins a defect reproduced or confirmed during the audit; see
`../OmniDB Vault/06 Analysis/Issues and Required Fixes (Sep 2026).md`.
"""

import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest
from conftest import make_index, make_store

import omnidb_server.consolidate as consolidate
from omnidb_server import db as dbmod
from omnidb_server.adapters import OmniDBMemory, OmniDBStore
from omnidb_server.config import Config
from omnidb_server.store import StoreError

# --- BUG-01: consolidation marks only the episodes it consumed ----------------

def test_consolidation_marks_only_distilled_episodes(tmp_path, monkeypatch):
    # chunked extraction (E-12) raised the per-run cap to 500; pin the cap
    # behavior itself at 50 here so the marking boundary stays tested
    monkeypatch.setattr(consolidate, "_MAX_FACTS_PER_RUN", 50)
    st = make_store(tmp_path)
    for i in range(60):
        st.remember(f"episode fact number {i} about topic {i}", kind="episodic")

    report = consolidate.run_consolidation(st, hours=24)
    assert report["facts_created"] == 50  # the fact cap still holds
    assert len(report["episode_ids"]) == 50

    marked = st.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE kind = 'episodic' "
        "AND metadata LIKE '%consolidated%'"
    ).fetchone()[0]
    assert marked == 50, "episodes beyond the fact cap must stay retryable"

    # the undistilled tail is picked up by the next run, not lost
    report2 = consolidate.run_consolidation(st, hours=24)
    assert report2["facts_created"] == 10
    st.close()


# --- BUG-11: an oversized episode cannot starve the window --------------------

def test_fit_episode_budget_skips_oversized():
    episodes = [
        {"id": 1, "content": "x" * 200},  # cost 58, alone exceeds the toy budget
        {"id": 2, "content": "y" * 10},   # cost 10
        {"id": 3, "content": "z" * 10},   # cost 10
    ]
    kept, skipped = consolidate._fit_episode_budget(episodes, max_tokens=40)
    assert [e["id"] for e in kept] == [2, 3]
    assert skipped == 1


# --- BUG-03: a failed batch rolls back instead of half-committing later -------

def test_failed_batch_rolls_back_and_translates(store):
    store.remember("seed one", _explicit_id=1)
    with pytest.raises(StoreError) as exc:
        store.remember_many([
            {"content": "will be inserted A", "_explicit_id": 20},
            {"content": "will be inserted B", "_explicit_id": 21},
            {"content": "collides", "_explicit_id": 1},
        ])
    assert exc.value.code == "OMN-005"

    # a later unrelated write-path commit must not resurrect the partial batch
    store.recall("seed", k=1)
    n = store.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id IN (20, 21)"
    ).fetchone()[0]
    assert n == 0


# --- BUG-02: concurrent tool calls serialize on the store lock ----------------

def test_concurrent_search_and_writers_do_not_hang_or_crash(tmp_path):
    st = make_store(tmp_path)
    for i in range(20):
        st.remember(f"seed memory {i}", kind="semantic")

    stop = threading.Event()
    errors: list[Exception] = []

    def searcher():
        while not stop.is_set():
            try:
                st.search("SELECT id FROM memories LIMIT 5", unscoped=True)
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(exc)
                return

    def writer(name):
        while not stop.is_set():
            try:
                st.remember(f"{name} concurrent write", kind="semantic")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
                return

    # daemon=True so a deadlock regression fails the assertions below instead
    # of wedging the whole test run
    threads = [
        threading.Thread(target=searcher, daemon=True),
        threading.Thread(target=writer, args=("w1",), daemon=True),
        threading.Thread(target=writer, args=("w2",), daemon=True),
    ]
    for t in threads:
        t.start()
    time.sleep(2.0)
    stop.set()
    for t in threads:
        t.join(timeout=15)
    assert not errors, f"tool threads failed: {errors[:3]}"
    assert all(not t.is_alive() for t in threads), "tool threads hung"
    st.close()


# --- BUG-04: the documented OMNIDB_INDEX alias is honored ---------------------

def test_omnidb_index_alias_and_native_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIDB_INDEX", "brute")
    assert Config.load(home=tmp_path).vector_index == "brute"

    # the field-native variable wins when both are set
    monkeypatch.setenv("OMNIDB_VECTOR_INDEX", "hnsw")
    assert Config.load(home=tmp_path).vector_index == "hnsw"


# --- BUG-08: related() hides superseded rows -----------------------------------

def test_related_hides_superseded(store):
    hub = store.remember("hub node about dragons")["id"]
    old = store.remember("old dragon lore")["id"]
    new = store.remember("new dragon lore")["id"]
    store.link(hub, old)
    store.link(hub, new)
    store.supersede(old, new)

    got = {r["id"] for r in store.related(hub, max_hops=1)}
    assert got == {new}


# --- BUG-09: supersede refuses an already-superseded target --------------------

def test_supersede_rejects_cycle(store):
    a = store.remember("row a")["id"]
    b = store.remember("row b")["id"]
    store.supersede(a, b)
    with pytest.raises(StoreError, match="already superseded"):
        store.supersede(b, a)  # would close A->B->A and hide both rows
    assert store.get(a)["superseded_by"] == b


# --- BUG-07: migrations are atomic, and torn homes heal ------------------------

def test_torn_v2_migration_heals_on_open(tmp_path):
    st = make_store(tmp_path)
    st.remember("pre-tear memory")
    st.close()

    # simulate the state the pre-atomic runner produced on a mid-upgrade
    # crash: v2 columns present, user_version stuck at 1
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    st2 = make_store(tmp_path)  # must heal, not raise "duplicate column"
    version = st2.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == dbmod.SCHEMA_VERSION
    assert st2.get(1) is not None
    st2.close()


# --- PERF-01: spaced single writes no longer dump the sidecar per write --------

def test_persist_policy_spaced_writes(tmp_path):
    st = make_store(tmp_path, persist_every_ops=10**9)  # op-count trigger off
    flushes = {"n": 0}
    _orig_backend_persist = st.index.backend_for("default").persist

    def counting(*a, **kw):
        flushes["n"] += 1
        return _orig_backend_persist(*a, **kw)

    st.index.backend_for("default").persist = counting
    st._persist_count_attr = flushes
    for i in range(6):
        st.remember(f"spaced write {i}", kind="semantic")
        time.sleep(0.08)
    # ~0.4s of writes repeatedly crossed the old 0.25s time trigger and
    # flushed per write; the new pending-ops floor (default 8) must not
    assert flushes["n"] == 0

    # the durability ceiling still fires with very few pending ops
    st.cfg.persist_max_s = 0.05
    st._last_persist_at = st.now_fn() - 0.2
    st.remember("ceiling probe", kind="semantic")
    assert flushes["n"] == 1
    st.close()


def test_persist_interval_scales_with_flush_cost(tmp_path):
    st = make_store(tmp_path, persist_every_ops=10**9,
                    persist_min_ops=2, persist_max_s=10**9)
    st._last_flush_s = 1.0  # simulate a 1s full-index dump at scale
    st._last_persist_at = st.now_fn() - 1.0  # > persist_every_s (0.25)
    flushes = {"n": 0}
    _orig_backend_persist = st.index.backend_for("default").persist

    def counting(*a, **kw):
        flushes["n"] += 1
        return _orig_backend_persist(*a, **kw)

    st.index.backend_for("default").persist = counting
    st._persist_count_attr = flushes
    st.remember("cost probe a", kind="semantic")
    st.remember("cost probe b", kind="semantic")
    # interval = max(0.25, 4 * 1.0) = 4s -> elapsed 1s must not flush
    assert flushes["n"] == 0
    st._last_persist_at = st.now_fn() - 4.5
    st.remember("cost probe c", kind="semantic")
    assert flushes["n"] == 1
    st.close()


# --- PERF-02: brute-force buffer semantics --------------------------------------

def test_brute_buffer_roundtrip(tmp_path):
    """A backend unit test, so it constructs the backend directly.

    ADR-08 / W3-3 made the store's index a per-tenant router; this exercises
    `BruteForceIndex`'s own buffer semantics (active_count, update, remove,
    persist, reload), which the router would only delegate to.
    """
    import numpy as np

    from omnidb_server.vector_index import BruteForceIndex

    idx = BruteForceIndex(tmp_path / "tenant-default.npz", 8)
    rng = np.random.default_rng(7)
    vecs = rng.normal(size=(100, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx.add(list(range(100)), vecs)
    assert idx.active_count() == 100 and idx.count == 100
    assert idx.query(vecs[3], k=1)[0][0] == 3

    idx.update(3, vecs[50] * 2)  # unnormalized rewrite dominates the ip score
    assert idx.query(vecs[50], k=1)[0][0] == 3

    idx.remove([i for i in range(100) if i % 2 == 0])
    assert idx.active_count() == 50
    assert idx.indexed_ids() == set(range(1, 100, 2))

    idx.persist()
    reloaded = BruteForceIndex.load(tmp_path / "tenant-default.npz", 8)
    assert reloaded.active_count() == 50
    assert reloaded.query(vecs[3], k=1)[0][0] != 3  # 3 was removed


# --- PERF-04/05: query logging single-commit + retention pruning ----------------

def test_query_log_records_empty_recall(tmp_path):
    st = make_store(tmp_path, query_log=True)
    st.recall("nothing matches zzz", k=3)
    n = st.conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    assert n == 1
    st.close()


def test_query_log_retention_prunes(tmp_path):
    st = make_store(tmp_path, query_log=True, query_retention_rows=8)
    for i in range(4):
        st.remember(f"retention filler {i} alpha")
    for i in range(130):
        st.recall(f"alpha query {i}", k=1)
    # retention bounds growth to max_rows + one prune interval (the interval
    # is max(64, max_rows // 4)); it converges to max_rows at each prune
    interval = max(64, 8 // 4)
    n = st.conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
    assert n <= 8 + interval
    st.close()


# --- DEBT-01: purge covers superseded-only rows ---------------------------------

def test_purge_removes_superseded_rows(store):
    a = store.remember("superseded purge probe a")["id"]
    b = store.remember("superseded purge probe b")["id"]
    store.supersede(a, b)
    assert store.check()["superseded"] == 1
    result = store.purge()
    assert result["purged"] == 1
    assert store.get(a) is None
    assert store.get(b) is not None


# --- BUG-12: recall k / k_fetch are clamped --------------------------------------

def test_recall_k_and_k_fetch_clamped(store):
    for i in range(5):
        store.remember(f"clamp filler {i} unique")
    hits = store.recall("clamp filler", k=10**7)
    assert 0 < len(hits) <= 5
    hits = store.recall("clamp filler", k=3, k_fetch=10**9)
    assert len(hits) <= 3


# --- ARCH-05: the home lock is exclusive across processes and death-released ----

def test_home_lock_is_exclusive_across_processes(tmp_path):
    from omnidb_server.server import _acquire_home_lock, _release_home_lock

    child = subprocess.Popen(
        [sys.executable, "-c",
         "import pathlib, time;"
         "from omnidb_server.server import _acquire_home_lock;"
         "fd = _acquire_home_lock(pathlib.Path(r'%s'));"
         "assert not isinstance(fd, OSError), fd;"
         "print('locked', flush=True); time.sleep(10)" % tmp_path],
        stdout=subprocess.PIPE, text=True, cwd=os.getcwd(),
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        held = _acquire_home_lock(tmp_path)
        assert isinstance(held, OSError), "second lock must fail while held"
    finally:
        child.terminate()
        child.wait(timeout=10)

    # the OS releases the lock when the holder dies: acquire must succeed
    deadline = time.monotonic() + 5
    while True:
        fd = _acquire_home_lock(tmp_path)
        if not isinstance(fd, OSError):
            break
        assert time.monotonic() < deadline, "lock not released after holder death"
        time.sleep(0.1)
    _release_home_lock(fd)


# --- QA-02: non-serializable metadata fails typed --------------------------------

def test_non_serializable_metadata_raises_typed(store):
    with pytest.raises(StoreError) as exc:
        store.remember("bad metadata probe", metadata={"x": object()})
    assert exc.value.code == "OMN-002"


# --- SEC-04: content length and type are bounded ---------------------------------

def test_content_cap_and_type_raise_typed(store):
    store.cfg.max_content_chars = 50
    with pytest.raises(StoreError) as exc:
        store.remember("x" * 51)
    assert exc.value.code == "OMN-005"
    with pytest.raises(StoreError) as exc:
        store.remember_many([{"content": 42}])
    assert exc.value.code == "OMN-002"
    # under the cap works as before
    store.remember("fits fine")
    assert store.stats()["live"] == 1


# --- SEC-01: consolidation facts carry provenance and honor the floor -------------

def test_consolidation_marks_origin_and_honors_confidence_floor(tmp_path, monkeypatch):
    import json as _json

    cfg_kwargs = dict(llm_base_url="http://llm.local/v1", llm_model="m",
                      consolidation_min_confidence=0.5)
    from omnidb_server import db as dbmod
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute", **cfg_kwargs)
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     make_index(tmp_path, "brute", 512), cfg)
    for i in range(3):
        st.remember(f"episode {i}: prefers dark mode", kind="episodic")

    payload_json = _json.dumps([
        {"fact": "confident fact about dark mode", "confidence": 0.9},
        {"fact": "shaky fact about dark mode", "confidence": 0.2},
    ])

    class R:
        def raise_for_status(self):
            pass

        def json(self_inner):
            return {"choices": [{"message": {"content": payload_json}}]}

    monkeypatch.setattr(consolidate.httpx, "post", lambda *a, **k: R())
    report = consolidate.run_consolidation(st, hours=24)
    assert report["facts_created"] == 1   # the 0.2-confidence fact hit the floor
    assert report["facts_dropped"] == 1

    hits = st.recall("confident fact", k=3, kind="semantic")
    meta = hits[0]["metadata"]
    assert meta["origin"] == "consolidation"
    assert meta["confidence"] == 0.9
    assert meta["source_episodes"]
    st.close()


# --- PERF-08: agent-scoped recall over-fetches the dense leg ----------------------

def test_agent_overfetch_scales_dense_fetch(tmp_path):
    st = make_store(tmp_path, agent_overfetch=1.0)
    for i in range(10):
        st.remember(f"overfetch filler {i}", agent_id="a")
    seen = []

    # ADR-08 / W3-3: the router is slotted, so the spy goes on the backend it
    # delegates to. recall now passes the tenant through, which is the thing
    # this test is watching — the over-fetch factor for agent-scoped recalls.
    backend = st.index.backend_for("default")
    orig_query = backend.query

    def spy(vec, k):
        seen.append(k)
        return orig_query(vec, k)

    backend.query = spy
    st.recall("overfetch filler", k=5, agent_id="a")
    base = seen[-1]
    seen.clear()

    st.cfg.agent_overfetch = 3.0
    st.recall("overfetch filler", k=5, agent_id="a")
    assert seen[-1] == base * 3
    # unscoped recall is untouched by the multiplier
    seen.clear()
    st.recall("overfetch filler", k=5)
    assert seen[-1] == base
    st.close()


# --- PERF-07: ef_search reaches the hnsw index ------------------------------------

def test_ef_search_config_reaches_hnsw(tmp_path):
    pytest.importorskip("hnswlib")
    from omnidb_server.config import Config
    from omnidb_server.db import connect
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.vector_index import open_vector_index

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="hnsw",
                 ef_search=128)
    conn = connect(cfg.home)
    emb = HashingEmbedder()
    idx = open_vector_index(cfg.home, "hnsw", emb.dim, 0, ef_search=cfg.ef_search)
    # ADR-08 / W3-3: the store's index is a per-tenant router, so the hnswlib
    # object lives one hop down, on that tenant's backend.
    assert idx.backend_for("default").index.ef == 128
    conn.close()


# --- BUG-06: LangGraph put is an upsert ----------------------------------------

def test_langgraph_put_is_upsert(store):
    st = OmniDBStore(store)
    st.put(("users",), "pref", {"text": "likes vanilla"})
    st.put(("users",), "pref", {"text": "likes chocolate"})

    assert st.get(("users",), "pref")["value"] == {"text": "likes chocolate"}
    live = st.store.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE metadata LIKE '%\"key\": \"pref\"%' "
        "AND deleted_ms IS NULL"
    ).fetchone()[0]
    assert live == 1, "re-putting a key must not accumulate duplicate rows"


# --- BUG-05: adapter reads and writes are agent-scoped --------------------------

def test_llamaindex_clear_is_scoped_to_its_agent(store):
    mem = OmniDBMemory(store)
    foreign = store.remember("semantic fact from another agent",
                             agent_id="other-agent")

    mem.put("first snippet")
    mem.put("second snippet")
    mem.clear()

    assert mem.get_all() == []
    kept = store.get(foreign["id"])
    assert kept is not None and kept["deleted_ms"] is None


def test_langgraph_scope_isolates_agents(store):
    st = OmniDBStore(store)
    foreign = store.remember("note about zebra quotas", agent_id="other-agent")

    st.put(("ns",), "k", {"text": "adapter-owned value"})
    item = st.get(("ns",), "k")
    assert item is not None

    # the adapter's search stays inside its agent scope
    hits = st.search(("ns",), "zebra quotas", limit=5)
    assert all(h["key"] != "note about zebra quotas" for h in hits)
    assert store.get(foreign["id"]) is not None
