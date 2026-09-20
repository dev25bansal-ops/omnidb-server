"""Portability: JSONL export/import roundtrips preserve ids, fields and edges."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.portability import backup_home, export_jsonl, import_jsonl
from omnidb_server.store import StoreError


def test_backup_snapshot_restores_and_reconciles_vectors(tmp_path):
    """AUDIT-I03 acceptance: every snapshot carries index_dirty=1, so a
    restore reconciles vectors from the rows even when the sidecar copy is
    torn or belongs to a different generation."""
    src = make_store(tmp_path / "src")
    a = src.remember("vector tuning guide")["id"]
    b = src.remember("hnsw ef search notes")["id"]
    src.flush()
    snap_root = tmp_path / "backups"
    report = backup_home(src, snap_root)
    src.close()

    snap = Path(report["dest"])
    conn = sqlite3.connect(snap / "omnidb.db")
    try:
        assert conn.execute(
            "SELECT value FROM meta WHERE key = 'index_dirty'").fetchone()[0] == "1"
    finally:
        conn.close()

    # simulate the torn/stale sidecar copy the backup docstring warns about
    (snap / "vectors.npz").unlink()

    restored = tmp_path / "restored"
    shutil.copytree(snap, restored)
    st = make_store(restored)
    try:
        assert st.stats()["live"] == 2
        vec = st.embedder.encode(["vector tuning guide"])[0]
        assert {h[0] for h in st.index.query(vec, k=2)} == {a, b}
        assert dbmod.get_meta(st.conn, "index_dirty") == "0"  # healed on open
    finally:
        st.close()


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_roundtrip_preserves_ids_fields_edges_and_recall(tmp_path):
    src = make_store(tmp_path / "src")
    ids = [
        src.remember("vector index tuning guide", metadata={"project": "apollo"})["id"],
        src.remember("rotate api keys quarterly", kind="procedural",
                     trigger_patterns=["^rotate\\b", "api key"])["id"],
        src.remember("standup notes monday", kind="episodic", agent_id="alpha",
                     created_ms=1_000, access_count=6)["id"],
    ]
    src.link(ids[0], ids[1], label="related")
    src.link(ids[1], ids[2], label="context")
    dump = tmp_path / "dump.jsonl"

    report = export_jsonl(src, dump)
    src.close()

    assert report["memories"] == 3
    assert report["edges"] == 2
    records = _records(dump)
    mem_records = [r for r in records if r["type"] == "memory"]
    edge_records = [r for r in records if r["type"] == "edge"]
    assert {r["id"] for r in mem_records} == set(ids)
    assert {(r["src"], r["dst"], r["label"]) for r in edge_records} == {
        (ids[0], ids[1], "related"),
        (ids[1], ids[2], "context"),
    }
    by_id = {r["id"]: r for r in mem_records}
    assert by_id[ids[0]]["metadata"] == {"project": "apollo"}
    assert by_id[ids[2]]["created_ms"] == 1_000
    assert by_id[ids[2]]["access_count"] == 6
    assert by_id[ids[2]]["agent_id"] == "alpha"
    assert by_id[ids[1]]["trigger_patterns"] == ["^rotate\\b", "api key"]

    dst = make_store(tmp_path / "dst")
    imported = import_jsonl(dst, dump)
    assert imported["memories"] == 3
    assert imported["edges"] == 2

    for mid in ids:
        row = dst.get(mid)
        assert row is not None
        assert row["kind"] == by_id[mid]["kind"]
        assert row["content"] == by_id[mid]["content"]
    assert dst.stats()["edges"] == 2
    hits = dst.recall("rotate api keys", k=5)
    assert any(h["id"] == ids[1] and h["metadata"].get("skill_trigger") for h in hits)

    src = make_store(tmp_path / "src")
    query = "vector index tuning"
    assert {h["id"] for h in src.recall(query, k=5)} == {h["id"] for h in dst.recall(query, k=5)}
    assert src.related(ids[1], max_hops=2) == dst.related(ids[1], max_hops=2)
    src.close()
    dst.close()


def test_export_skips_dead_rows_and_their_edges(tmp_path):
    src = make_store(tmp_path / "src")
    a = src.remember("kept alpha")["id"]
    b = src.remember("forgotten beta")["id"]
    c = src.remember("superseded gamma")["id"]
    d = src.remember("replacement delta")["id"]
    src.link(a, b, label="related")
    src.link(a, d, label="related")
    src.forget(b)
    src.supersede(c, d)
    dump = tmp_path / "dump.jsonl"

    report = export_jsonl(src, dump)
    src.close()

    assert report["memories"] == 2  # b soft-deleted, c superseded
    assert report["edges"] == 1  # the a->b edge dies with b
    assert {r["id"] for r in _records(dump) if r["type"] == "memory"} == {a, d}

    dst = make_store(tmp_path / "dst")
    import_jsonl(dst, dump)
    assert dst.stats()["live"] == 2
    assert dst.stats()["edges"] == 1
    # AUDIT-I14/I04: a gapped export preserves original IDs and edge endpoints.
    assert {r[0] for r in dst.conn.execute("SELECT id FROM memories")} == {a, d}
    assert dst.get(a)["content"] == "kept alpha"
    assert dst.get(d)["content"] == "replacement delta"
    assert dst.get(b) is None
    assert dst.get(c) is None
    assert {tuple(r) for r in dst.conn.execute("SELECT src, dst, label FROM edges")} == {
        (a, d, "related"),
    }
    memory_records = [r for r in _records(dump) if r["type"] == "memory"]
    assert not any(r["content"] == "forgotten beta" for r in memory_records)
    dst.close()


def test_import_twice_refuses_non_empty_home(tmp_path):
    src = make_store(tmp_path / "src")
    src.remember("only row")
    dump = tmp_path / "dump.jsonl"
    export_jsonl(src, dump)
    src.close()

    dst = make_store(tmp_path / "dst")
    import_jsonl(dst, dump)
    with pytest.raises(StoreError) as refused:
        import_jsonl(dst, dump)
    assert refused.value.code == "OMN-005"
    assert dst.stats()["live"] == 1  # the second attempt wrote nothing
    dst.close()


def test_malformed_export_validated_before_any_write(tmp_path):
    dump = tmp_path / "bad.jsonl"
    dump.write_text(
        '{"type":"memory","id":1,"content":"fine"}\n'
        '{"type":"memory","id":1,"content":"duplicate id"}\n',
        encoding="utf-8",
    )
    dst = make_store(tmp_path / "dst")
    with pytest.raises(StoreError) as bad:
        import_jsonl(dst, dump)
    assert bad.value.code == "OMN-005"
    assert dst.stats()["live"] == 0
    dst.close()
