"""Skill bundles (F-01) and query-log aggregation (F-02)."""

import json
import time
import zipfile

import conftest
import pytest

from omnidb_server.skills import export_skill, import_skill, list_skills
from omnidb_server.store import E_CONSTRAINT, E_TYPE, StoreError


def _read_bundle(path):
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        lines = [json.loads(line) for line in zf.read("memories.jsonl").decode("utf-8").splitlines() if line.strip()]
    return manifest, lines


def test_export_import_roundtrip_preserves_triggers(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        st.remember("deploy checklist: build, test, tag, push", kind="procedural",
                    metadata={"skill": "release"}, trigger_patterns=["deploy", "release"])
        st.remember("release cadence is every other Tuesday", metadata={"skill": "release"})
        st.remember("unrelated episodic noise")
        bundle = tmp_path / "release.zip"
        exported = export_skill(st, "release", bundle)
        assert exported == {"name": "release", "memories": 2, "path": str(bundle)}

        manifest, lines = _read_bundle(bundle)
        assert manifest["name"] == "release"
        assert manifest["triggers"] == ["deploy", "release"]
        assert {line["kind"] for line in lines} == {"procedural", "semantic"}
        assert {line["metadata"]["skill"] for line in lines} == {"release"}

        other = conftest.make_store(tmp_path / "imported")
        try:
            result = import_skill(other, bundle)
            assert result == {"name": "release", "memories": 2, "triggers": 2}
            hits = other.recall("deploy checklist", k=5)
            assert hits and hits[0]["metadata"]["skill_trigger"] is True
            assert hits[0]["metadata"]["skill"] == "release"
            rows = other.conn.execute(
                "SELECT kind, metadata, trigger_patterns FROM memories "
                "WHERE kind = 'procedural'"
            ).fetchall()
            assert len(rows) == 1
            assert json.loads(rows[0]["trigger_patterns"]) == ["deploy", "release"]
        finally:
            other.close()
    finally:
        st.close()


def test_import_refuses_duplicate_skill_and_overwrite_replaces(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        st.remember("v1 procedure: run the smoke suite", kind="procedural",
                    metadata={"skill": "qa"}, trigger_patterns=["smoke"])
        bundle = tmp_path / "qa.zip"
        export_skill(st, "qa", bundle)
        with pytest.raises(StoreError) as exc:
            import_skill(st, bundle)
        assert exc.value.code == E_CONSTRAINT

        result = import_skill(st, bundle, overwrite=True)
        assert result["name"] == "qa" and result["memories"] == 1
        listed = list_skills(st)
        assert listed == [{"name": "qa", "memories": 1}]
        live = st.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_ms IS NULL"
        ).fetchone()[0]
        assert live == 1
    finally:
        st.close()


def test_list_skills_groups_by_skill_and_ignores_untagged(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        st.remember("a one", metadata={"skill": "deploy"})
        st.remember("a two", metadata={"skill": "deploy"})
        st.remember("b one", metadata={"skill": "review"})
        st.remember("no skill at all")
        st.remember("deleted skill row", metadata={"skill": "deploy"})
        st.conn.execute("UPDATE memories SET deleted_ms = 123 WHERE content = 'deleted skill row'")
        st.conn.commit()
        st.remember("superseded skill row", metadata={"skill": "review"})
        superseded_id = st.conn.execute(
            "SELECT id FROM memories WHERE content = 'superseded skill row'"
        ).fetchone()[0]
        st.conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ?", (superseded_id, superseded_id)
        )
        st.conn.commit()
        listed = list_skills(st)
        assert listed == [{"name": "deploy", "memories": 2}, {"name": "review", "memories": 1}]
        st.remember("c one", metadata={"skill": "zebra"}, access_count=0)
        st.remember("c two", metadata={"skill": "zebra"})
        st.remember("c three", metadata={"skill": "zebra"})
        assert list_skills(st)[0] == {"name": "zebra", "memories": 3}
    finally:
        st.close()


def test_skill_reads_tolerate_metadata_that_is_not_an_object(tmp_path):
    """metadata is a JSON column, so a row can hold valid JSON that is not a
    dict - a list, a string, a number. list_skills already skips those; the
    _skill_rows read behind export/import called .get() on the parsed value
    unguarded, so a LIKE-matching non-object row raised AttributeError and took
    the whole export down instead of skipping one row."""
    st = conftest.make_store(tmp_path)
    try:
        st.remember("a real one", metadata={"skill": "deploy"})
        # valid JSON, not an object, and LIKE-matching the canonical
        # '"skill": "deploy"' form so the prefilter selects it
        st.conn.execute(
            'INSERT INTO memories (content, kind, metadata, tenant_id, created_ms) '
            "VALUES ('junk list', 'semantic', ?, 'default', 1)",
            (json.dumps(["skill", "deploy"]),),
        )
        st.conn.execute(
            'INSERT INTO memories (content, kind, metadata, tenant_id, created_ms) '
            "VALUES ('junk string', 'semantic', ?, 'default', 1)",
            ('"skill": "deploy"',),
        )
        st.conn.commit()

        # the two reads that share this shape must not raise
        assert list_skills(st) == [{"name": "deploy", "memories": 1}]
        bundle = tmp_path / "deploy.zip"
        result = export_skill(st, "deploy", bundle)
        assert result["name"] == "deploy" and result["memories"] == 1
    finally:
        st.close()


def test_import_rejects_bundle_missing_manifest(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        bundle = tmp_path / "broken.zip"
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("memories.jsonl", '{"content": "orphan", "kind": "semantic", "metadata": {}}\n')
        with pytest.raises(StoreError) as exc:
            import_skill(st, bundle)
        assert exc.value.code == E_TYPE
        assert st.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    finally:
        st.close()


def test_import_rejects_bad_json_and_unknown_manifest_keys(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        bad_json = tmp_path / "badjson.zip"
        with zipfile.ZipFile(bad_json, "w") as zf:
            zf.writestr("manifest.json", "{not json")
            zf.writestr("memories.jsonl", "")
        with pytest.raises(StoreError) as exc:
            import_skill(st, bad_json)
        assert exc.value.code == E_TYPE

        unknown_keys = tmp_path / "unknownkeys.zip"
        with zipfile.ZipFile(unknown_keys, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"name": "x", "bogus": 1}))
            zf.writestr("memories.jsonl", "")
        with pytest.raises(StoreError) as exc2:
            import_skill(st, unknown_keys)
        assert exc2.value.code == E_TYPE

        not_a_zip = tmp_path / "notazip.zip"
        not_a_zip.write_text("definitely not a zip")
        with pytest.raises(StoreError) as exc3:
            import_skill(st, not_a_zip)
        assert exc3.value.code == E_TYPE
    finally:
        st.close()


def test_import_missing_file_is_not_found(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        with pytest.raises(StoreError) as exc:
            import_skill(st, tmp_path / "absent.zip")
        assert exc.value.code == "OMN-021"
    finally:
        st.close()


def test_query_stats_empty_table(tmp_path):
    st = conftest.make_store(tmp_path, query_log=True)
    try:
        stats = st.query_stats(hours=24.0)
        assert stats == {"window_hours": 24.0, "by_kind": {}, "top_queries": []}
        assert st.query_stats(hours=0.0)["top_queries"] == []
    finally:
        st.close()


def test_query_stats_aggregates_by_kind_and_top_queries(tmp_path):
    # freeze the store clock in the past so rows logged with the real wall
    # clock stay at or above the stats cutoff even for a zero-hour window
    # (avoids racing a coarse clock tick between logging and query_stats)
    st = conftest.make_store(tmp_path, query_log=True, now=lambda: time.time() - 60.0)
    try:
        st.remember("python decorators guide with examples")
        st.recall("python decorators", k=3)
        st.recall("python decorators", k=5)
        st.search("SELECT COUNT(*) AS n FROM memories", unscoped=True)
        stats = st.query_stats(hours=24.0)
        assert stats["window_hours"] == 24.0
        assert set(stats["by_kind"]) == {"recall", "search"}
        recall = stats["by_kind"]["recall"]
        assert recall["count"] == 2
        assert recall["max_ms"] >= recall["avg_ms"] >= 0
        assert stats["by_kind"]["search"]["count"] == 1
        queries = [entry["query"] for entry in stats["top_queries"]]
        assert queries[0] == "python decorators"
        assert all(entry["count"] >= 1 for entry in stats["top_queries"])
        # a zero-hour window still surfaces rows logged at now_ms
        narrow = st.query_stats(hours=0.0)
        assert set(narrow["by_kind"]) == {"recall", "search"}
    finally:
        st.close()
