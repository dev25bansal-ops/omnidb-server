import pytest
from conftest import make_index

from omnidb_server.store import StoreError


def _positions(results):
    return {r["id"]: i for i, r in enumerate(results)}


def test_remember_recall_roundtrip(store):
    stored = store.remember("python decorators guide with examples")
    hits = store.recall("python decorators guide", k=3)
    assert hits, "recall returned nothing"
    assert hits[0]["id"] == stored["id"]
    assert hits[0]["content"].startswith("python decorators")
    for field in ("rrf", "recency", "frequency", "score"):
        assert field in hits[0]


def test_fusion_orders_relevant_above_irrelevant(store):
    a = store.remember("python decorators guide")["id"]
    b = store.remember("jazz improvisation chords")["id"]
    c = store.remember("history of jazz chords evolution")["id"]
    results = store.recall("jazz chords", k=5)
    pos = _positions(results)
    assert b in pos and c in pos
    if a in pos:
        assert pos[b] < pos[a]
        assert pos[c] < pos[a]


def test_recency_decays_with_age(tmp_path):
    import conftest

    clock = {"t": 1_700_000_000.0}
    st = conftest.make_store(tmp_path, now=lambda: clock["t"])
    older = st.remember("alpha beta gamma")["id"]
    clock["t"] += 500 * 3600
    newer = st.remember("alpha beta gamma")["id"]
    results = st.recall("alpha beta gamma", k=5)
    by_id = {r["id"]: r for r in results}
    assert by_id[newer]["score"] > by_id[older]["score"]
    assert by_id[older]["recency"] < by_id[newer]["recency"]
    st.close()


def test_frequency_boosts_accessed_memory(tmp_path):
    import conftest

    st = conftest.make_store(tmp_path)
    first = st.remember("unique zebra habitat facts")["id"]
    second = st.remember("unique zebra habitat facts")["id"]

    st.recall("zebra habitat", k=1)  # touches only the top hit (lower id on tie)

    results = st.recall("zebra habitat", k=5)
    counts = {r["id"]: r["access_count"] for r in results}
    assert counts[first] >= 1
    pos = _positions(results)
    if counts[first] > counts[second]:
        assert pos[first] < pos[second]
    st.close()


def test_kind_filter(store):
    epi = store.remember("quarterly budget review notes", kind="episodic")["id"]
    sem = store.remember("budget planning best practices", kind="semantic")["id"]
    only_semantic = store.recall("budget", k=5, kind="semantic")
    ids = [r["id"] for r in only_semantic]
    assert sem in ids
    assert epi not in ids


def test_forget_soft_deletes(store):
    mid = store.remember("secret launch credentials")[  "id"]
    assert any(r["id"] == mid for r in store.recall("launch credentials", k=5))
    store.forget(mid)
    assert not any(r["id"] == mid for r in store.recall("launch credentials", k=5))
    with pytest.raises(StoreError):
        store.forget(mid)  # already deleted
    row = store.search(f"SELECT deleted_ms FROM memories WHERE id = {mid}", unscoped=True)
    assert row["rows"][0][0] is not None


def test_search_readonly_guard(store):
    with pytest.raises(StoreError):
        store.search("DELETE FROM memories", unscoped=True)
    with pytest.raises(StoreError):
        store.search("INSERT INTO memories(content) VALUES ('x')", unscoped=True)
    out = store.search("SELECT COUNT(*) AS n FROM memories", unscoped=True)
    assert out["columns"] == ["n"]


def test_search_truncation(tmp_path):
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute",
                 max_search_rows=3)
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     make_index(tmp_path, "brute", 512), cfg)
    out = st.search(
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt WHERE x<10) "
        "SELECT x FROM cnt"
, unscoped=True)
    assert len(out["rows"]) == 3
    assert out["truncated"] is True
    st.close()
