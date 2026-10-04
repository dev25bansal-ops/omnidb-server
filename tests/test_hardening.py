
import pytest
from conftest import make_store

import omnidb_server.consolidate as consolidate
from omnidb_server.store import StoreError


def test_with_delete_blocked_and_data_intact(store):
    store.remember("precious memory one")
    store.remember("precious memory two")
    with pytest.raises(StoreError):
        store.search("WITH t AS (SELECT 1) DELETE FROM memories", unscoped=True)
    with pytest.raises(StoreError):
        store.search("WITH t AS (SELECT 1) UPDATE memories SET content = 'x'", unscoped=True)
    with pytest.raises(StoreError):
        store.search("PRAGMA journal_mode=DELETE", unscoped=True)
    out = store.search("SELECT COUNT(*) AS n FROM memories", unscoped=True)
    assert out["rows"][0][0] == 2


def test_runaway_cte_aborted_by_step_budget(store, monkeypatch):
    monkeypatch.setattr(store, "max_query_steps", 50_000)
    with pytest.raises(StoreError, match="aborted"):
        store.search(
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
            "SELECT sum(x) FROM c"
, unscoped=True)


def test_related_cycle_safe(store):
    a = store.remember("a")["id"]
    b = store.remember("b", links=[{"to": a, "label": "calls"}])["id"]
    # mutual edge creates a cycle; unbounded recursion would explode here
    store.link(a, b, label="calls")
    got = store.related(a, max_hops=50)
    ids = [r["id"] for r in got]
    assert ids.count(b) <= 1


def test_bad_link_does_not_persist_memory(store):
    before = store.stats()["live"]
    with pytest.raises(StoreError):
        store.remember("orphan entry", links=[{"to": 999999}])
    assert store.stats()["live"] == before
    with pytest.raises(StoreError):
        store.remember("malformed", links=[{"nope": 1}])


def test_link_to_soft_deleted_target_rejected(store):
    target = store.remember("soon forgotten")["id"]
    store.forget(target)
    src = store.remember("still here")["id"]
    with pytest.raises(StoreError):
        store.link(src, target)


def test_recency_clock_skew_no_overflow(tmp_path):
    clock = {"t": 1_700_000_000.0}
    st = make_store(tmp_path, now=lambda: clock["t"])
    st.remember("skewed timeline entry")
    clock["t"] -= 10_000 * 3600  # wall clock jumped backwards
    hits = st.recall("skewed timeline", k=3)
    assert hits and hits[0]["recency"] == 1.0
    st.close()


def test_non_ascii_recall(store):
    store.remember("café résumé with naïve details")
    hits = store.recall("café résumé", k=3)
    assert hits and "résumé" in hits[0]["content"]


def test_llm_failure_does_not_consume_episodes(tmp_path, monkeypatch):
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    from omnidb_server.vector_index import BruteForceIndex

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute",
                 llm_base_url="http://llm.local/v1", llm_model="m")
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     BruteForceIndex(tmp_path / "vectors.npz", 512), cfg)

    def boom(*a, **k):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(consolidate.httpx, "post", boom)

    for i in range(3):
        st.remember(f"episode {i}", kind="episodic")

    report = consolidate.run_consolidation(st, hours=24)
    assert report["llm_used"] is False
    assert report["facts_created"] == 0
    assert "llm_error" in report

    flagged = st.search(
        "SELECT metadata FROM memories WHERE kind = 'episodic' AND metadata LIKE '%consolidated%'"
, unscoped=True)
    assert flagged["rows"] == []  # episodes left retryable
    st.close()
