"""CLI operations: check, purge, export, import driven through cli.main()."""

import json

import pytest
from conftest import make_store

from omnidb_server import cli


def test_cli_check_reports_integrity_json(tmp_path, capsys):
    store = make_store(tmp_path)
    a = store.remember("cli check probe alpha")["id"]
    b = store.remember("cli check probe beta")["id"]
    store.link(a, b, label="related")
    store.close()

    assert cli.main(["check", "--home", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) >= {"fts_integrity", "index_drift", "orphan_edges",
                           "wal_bytes", "soft_deleted"}
    assert report["fts_integrity"] == "ok"
    assert report["index_drift"] == 0
    assert report["orphan_edges"] == 0


def test_cli_purge_with_and_without_vacuum(tmp_path, capsys):
    store = make_store(tmp_path)
    gone = store.remember("purge me softly")["id"]
    store.remember("survivor row")
    store.forget(gone)
    store.close()

    assert cli.main(["purge", "--home", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"purged": 1}
    reopened = make_store(tmp_path)
    assert reopened.stats()["live"] == 1
    assert reopened.stats()["soft_deleted"] == 0
    reopened.close()

    store = make_store(tmp_path)
    store.forget(_last_live_id(store))
    store.close()
    assert cli.main(["purge", "--vacuum", "--home", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"purged": 1}


def _last_live_id(store):
    return int(store.conn.execute(
        "SELECT MAX(id) FROM memories WHERE deleted_ms IS NULL"
    ).fetchone()[0])


def test_cli_export_import_roundtrip_across_homes(tmp_path, capsys):
    src_home = tmp_path / "src"
    dst_home = tmp_path / "dst"
    dump = tmp_path / "ops.jsonl"

    store = make_store(src_home)
    a = store.remember("kubernetes ingress runbook")["id"]
    b = store.remember("kubectl rollout undo steps", kind="procedural",
                       trigger_patterns=["kubectl"])["id"]
    store.link(a, b, label="related")
    store.close()

    assert cli.main(["export", str(dump), "--home", str(src_home)]) == 0
    exported = json.loads(capsys.readouterr().out)
    assert exported["memories"] == 2
    assert exported["edges"] == 1
    assert exported["path"] == str(dump)
    assert exported["filters"]["kind"] is None  # default filter echoed
    full_lines = dump.read_text(encoding="utf-8").splitlines()
    full_kinds = [json.loads(line)["type"] for line in full_lines]
    assert full_kinds.count("memory") == 2 and full_kinds.count("edge") == 1

    # J-07: --kind slices the export
    import json as _json
    assert cli.main(["export", str(dump), "--home", str(src_home),
                     "--kind", "procedural"]) == 0
    filtered = dump.read_text(encoding="utf-8").splitlines()
    assert all(_json.loads(line)["kind"] == "procedural" for line in filtered), filtered
    assert len(filtered) == 1, filtered

    # J-05: atomic export leaves no .tmp siblings behind
    leftover = [p.name for p in tmp_path.glob("*.tmp")]
    assert not leftover, leftover

    dst_home.mkdir(parents=True)
    # J-07: re-export the unfiltered set for the import (capsys accumulates
    # across cli.main calls, so consume the prior output first)
    capsys.readouterr()
    assert cli.main(["export", str(dump), "--home", str(src_home)]) == 0
    import_capsys = capsys.readouterr()
    assert cli.main(["import", str(dump), "--home", str(dst_home)]) == 0
    imported = json.loads(import_capsys.out)
    assert imported["memories"] == 2 and imported["edges"] == 1

    reopened = make_store(dst_home)
    assert {h["id"] for h in reopened.recall("kubectl rollout", k=5)} >= {b}
    rows = reopened.search(f"SELECT src, dst FROM edges WHERE src = {a} AND dst = {b}", unscoped=True)
    assert rows["rows"] == [[a, b]]
    reopened.close()


def test_cli_import_refuses_non_empty_home_and_bad_file(tmp_path, capsys):
    src_home = tmp_path / "src"
    dump = tmp_path / "data.jsonl"
    store = make_store(src_home)
    store.remember("seed row for export")
    store.close()
    assert cli.main(["export", str(dump), "--home", str(src_home)]) == 0
    capsys.readouterr()

    dst_home = tmp_path / "dst"
    dst_home.mkdir()
    assert cli.main(["import", str(dump), "--home", str(dst_home)]) == 0
    with pytest.raises(SystemExit) as usage:
        cli.main(["import", "--home", str(dst_home)])
    assert usage.value.code == 2

    rc = cli.main(["import", str(dump), "--home", str(dst_home)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "omnidb-server: error:" in err and "[OMN-005]" in err
