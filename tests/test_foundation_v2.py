"""Foundation v2: agent scoping, skill triggers, supersede, check/purge."""

import sqlite3

import pytest

from omnidb_server import db as dbmod
from omnidb_server.store import StoreError


def _open_brute(tmp_path):
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.vector_index import BruteForceIndex

    cfg = _cfg(tmp_path)
    conn = dbmod.connect(tmp_path)
    emb = HashingEmbedder()
    return _store(conn, emb, BruteForceIndex.load(tmp_path / "vectors.npz", emb.dim), cfg)


def _cfg(tmp_path, **extra):
    from omnidb_server.config import Config

    return Config(home=tmp_path, embedder="hashing", vector_index="brute", **extra)


def _store(conn, embedder, index, cfg):
    from omnidb_server.store import MemoryStore

    return MemoryStore(conn, embedder, index, cfg)


# --- migration v2 -------------------------------------------------------------

def test_fresh_db_at_v2(store):
    # Schema is now at v3 (queries table added); the v2 columns are still
    # part of the expected schema so we verify the migration didn't drop them.
    version = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == dbmod.SCHEMA_VERSION
    assert dbmod.SCHEMA_VERSION >= 3
    cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(memories)")}
    assert {"agent_id", "trigger_patterns"} <= cols
    q_cols = {r["name"] for r in store.conn.execute("PRAGMA table_info(queries)")}
    assert {"kind", "query", "result_count", "took_ms", "created_ms"} <= q_cols


def test_v1_db_upgrades_without_data_loss(tmp_path):
    # hand-build a v1 database (pre-agent columns) with data in it
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    conn.executescript(dbmod.MIGRATIONS[0][1])
    conn.execute("INSERT INTO memories(kind, content, metadata, created_ms) "
                 "VALUES ('semantic', 'vintage v1 row', '{}', 12345)")
    conn.commit()
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    st = _open_brute(tmp_path)
    # Migration runs all the way to the current head.
    assert st.conn.execute("PRAGMA user_version").fetchone()[0] == dbmod.SCHEMA_VERSION
    hits = st.recall("vintage v1 row", k=3)
    assert len(hits) == 1 and hits[0]["content"] == "vintage v1 row"
    assert hits[0]["metadata"].get("skill_trigger") is None
    st.close()


# --- agent scoping ------------------------------------------------------------

def test_remember_agent_id_roundtrip_and_isolation(store):
    a = store.remember("deploy runbook for payments", agent_id="alpha")["id"]
    b = store.remember("deploy runbook for billing", agent_id="beta")["id"]
    assert store.get(a)["agent_id"] == "alpha"

    alpha_hits = store.recall("deploy runbook", k=10, agent_id="alpha")
    beta_hits = store.recall("deploy runbook", k=10, agent_id="beta")
    assert [h["id"] for h in alpha_hits] == [a]
    assert [h["id"] for h in beta_hits] == [b]

    unscoped = store.recall("deploy runbook", k=10)
    assert {h["id"] for h in unscoped} == {a, b}


def test_agent_stats_top20_and_live_only(store):
    for i in range(3):
        store.remember(f"shared note {i}", agent_id="worker-a")
    gone = store.remember("transient note", agent_id="worker-b")["id"]
    store.forget(gone)
    store.remember("orphan note")  # no agent_id: must not appear
    agents = store.stats()["agents"]
    assert agents == {"worker-a": 3}


# --- skill triggers -----------------------------------------------------------

def test_trigger_pattern_procedural_appears_in_recall(store):
    store.remember("kubernetes troubleshooting notes")
    skill = store.remember(
        "run kubectl get events first",
        kind="procedural",
        trigger_patterns=["^deploy\\b", "kubectl"],
    )["id"]

    hits = store.recall("how do I use kubectl here?", k=5)
    by_id = {h["id"]: h for h in hits}
    assert skill in by_id
    assert by_id[skill]["metadata"]["skill_trigger"] is True

    # no trigger match -> the procedural row only shows up if fusion finds it
    misses = store.recall("gardening tips for spring", k=5)
    assert all(not h["metadata"].get("skill_trigger") for h in misses)


def test_trigger_respects_agent_scope_and_selective(store):
    store.remember("restart the service", kind="procedural",
                   trigger_patterns=["restart"], agent_id="alpha")
    other = store.remember("restart the service (other agent)", kind="procedural",
                           trigger_patterns=["restart"], agent_id="beta")

    hits = store.recall("please restart things", k=5, agent_id="alpha")
    assert all(h["metadata"].get("skill_trigger") for h in hits)
    assert all(h["id"] != other for h in hits)

    # soft-deleted skills never fire
    store.forget_many([h["id"] for h in hits])
    assert store.recall("please restart things", k=5, agent_id="alpha") == []


def test_supersede_hides_old_double_raises(store):
    old = store.remember("draft policy text")["id"]
    new = store.remember("final policy text")["id"]

    result = store.supersede(old, new)
    assert result == {"superseded": old, "by": new}

    assert not any(h["id"] == old for h in store.recall("policy text", k=10))
    assert any(h["id"] == new for h in store.recall("policy text", k=10))
    assert store.stats()["live"] == 1

    with pytest.raises(StoreError) as dup:
        store.supersede(old, new)
    assert dup.value.code == "OMN-005"


def test_supersede_validation_errors(store):
    a = store.remember("row a")["id"]
    store.remember("row b")
    dead = store.remember("row to delete")["id"]
    store.forget(dead)

    with pytest.raises(StoreError) as same:
        store.supersede(a, a)
    assert same.value.code == "OMN-002"
    with pytest.raises(StoreError) as missing:
        store.supersede(a, 999999)
    assert missing.value.code == "OMN-021"
    with pytest.raises(StoreError) as deleted:
        store.supersede(dead, a)
    assert deleted.value.code == "OMN-021"


# --- check() --------------------------------------------------------------------

def test_check_reports_sane_keys(store):
    a = store.remember("check probe one")["id"]
    b = store.remember("check probe two")["id"]
    store.link(a, b, label="related")
    report = store.check()
    assert set(report) >= {"fts_integrity", "index_drift", "orphan_edges",
                           "wal_bytes", "soft_deleted"}
    assert report["fts_integrity"] == "ok"
    assert report["index_drift"] == 0
    assert report["orphan_edges"] == 0
    assert isinstance(report["wal_bytes"], int)


def test_check_counts_soft_deleted_and_drift(store):
    ids = [store.remember(f"drift probe {i}")["id"] for i in range(3)]
    store.flush()
    store.forget(ids[0])
    report = store.check()
    assert report["soft_deleted"] == 1
    assert report["index_drift"] == 0  # forget removed it from the index too
    # simulate index drift without touching rows
    store.index.remove([ids[1]])
    assert store.check()["index_drift"] == 1


# --- purge ----------------------------------------------------------------------

def test_purge_hard_deletes_and_cascades_edges(store):
    keep = store.remember("survivor of cleanup")["id"]
    gone_a = store.remember("quixotic target alpha")["id"]
    gone_b = store.remember("quixotic target beta")["id"]
    live_edge = store.remember("edge anchor")["id"]
    store.link(gone_a, gone_b, label="related")
    store.link(keep, live_edge, label="related")
    store.forget_many([gone_a, gone_b])
    assert store.stats()["edges"] == 2

    result = store.purge()
    assert result == {"purged": 2}
    assert store.stats()["soft_deleted"] == 0
    assert store.stats()["edges"] == 1  # cascade removed the dead pair's edge
    assert store.get(gone_a) is None
    assert store.get(keep) is not None
    assert store.search(
        "SELECT COUNT(*) AS n FROM memory_fts WHERE memory_fts MATCH 'quixotic'"
, unscoped=True)["rows"][0][0] == 0  # FTS ad-trigger kept the index synced

    store.purge(do_vacuum=True)
    assert store.stats()["live"] == 2


# --- importer hook ---------------------------------------------------------------

def test_explicit_id_insert_preserves_created_and_count(store):
    mid = store.remember(
        "imported historical row",
        created_ms=1_000,
        access_count=7,
        _explicit_id=4242,
    )["id"]
    assert mid == 4242
    row = store.get(mid)
    assert row["created_ms"] == 1_000
    assert row["access_count"] == 7
    hits = store.recall("imported historical row", k=3)
    assert any(h["id"] == 4242 and h["frequency"] > 0.0 for h in hits)

    # duplicate explicit ids roll back and raise typed errors (BUG-03), they
    # no longer leak a raw IntegrityError or half-commit on a later op
    with pytest.raises(StoreError) as exc:
        store.remember("duplicate id", _explicit_id=4242)
    assert exc.value.code == "OMN-005"
    store.recall("imported historical row", k=1)
    assert store.get(4242)["content"] == "imported historical row"
