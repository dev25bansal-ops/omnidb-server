"""N-03/N-04 CLI: doctor, init, skill subcommands, query-stats driven via cli.main()."""

import json
import sqlite3

from conftest import make_store

from omnidb_server import cli


def test_init_creates_home_toml_and_sample(tmp_path, capsys):
    home = tmp_path / "fresh"
    assert cli.main(["init", "--home", str(home)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["home"] == str(home)
    assert out["toml"] == str(home / "omnidb.toml")
    assert isinstance(out["sample_memory_id"], int)

    assert (home / "omnidb.db").is_file()
    toml_text = (home / "omnidb.toml").read_text(encoding="utf-8")
    for key in ("home", "embedder", "query_log", "half_life_hours"):
        assert f"# {key} =" in toml_text
    reopened = make_store(home)
    try:
        assert reopened.stats()["live"] == 1
    finally:
        reopened.close()


def test_init_refuses_non_empty_home(tmp_path, capsys):
    home = tmp_path / "busy"
    assert cli.main(["init", "--home", str(home)]) == 0
    capsys.readouterr()

    rc = cli.main(["init", "--home", str(home)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "omnidb-server: error:" in err and "[OMN-005]" in err


def test_doctor_reports_checks_and_ok(tmp_path, capsys):
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    checks = report["checks"]
    for name in ("python", "platform", "sqlite", "fts5", "hnswlib", "fastembed",
                 "embedder", "index", "recovery", "home", "disk", "lock"):
        assert name in checks
        assert isinstance(checks[name]["ok"], bool)
        assert isinstance(checks[name]["detail"], str)
    assert checks["fts5"]["ok"] is True
    assert "hashing" in checks["embedder"]["detail"]
    assert checks["home"]["ok"] is True  # missing home is informational, not a failure
    assert checks["lock"]["detail"].startswith("absent")


def test_doctor_surfaces_pending_recovery_markers(tmp_path, capsys):
    """AUDIT follow-up: the recovery check reports pending work (informational,
    never a failure) so operators see what the next open will heal."""
    assert cli.main(["remember", "recovery probe row", "--home", str(tmp_path)]) == 0
    capsys.readouterr()  # drain the remember output before reading doctor's
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    conn.execute("INSERT INTO meta(key, value) VALUES ('pending_reembed', '[3]') "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
    conn.commit()
    conn.close()
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 0  # still informational
    report = json.loads(capsys.readouterr().out)
    recovery = report["checks"]["recovery"]
    assert recovery["ok"] is True
    assert "pending_reembed" in recovery["detail"]
    assert "1 content revisions" in recovery["detail"]


def test_doctor_lock_probe_reports_free_and_held(tmp_path, capsys):
    # a bare .lock file is not a conflict signal any more (the lock is an OS
    # advisory lock that releases on holder death); with no database yet the
    # check stays informational
    (tmp_path / ".lock").write_text("99999" + chr(10), encoding="utf-8")
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 0  # informational command
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["lock"]["detail"].startswith("absent")
    assert report["checks"]["index"]["detail"].startswith("no database yet")

    # with a database present, a free lock reads ok even though the file persists
    from omnidb_server import db as dbmod

    conn = dbmod.connect(tmp_path)
    conn.close()
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checks"]["lock"]["ok"] is True
    assert "free" in report["checks"]["lock"]["detail"]

    # a held lock (taken in-process; region locks conflict across handles and
    # open file descriptions) still flags, and doctor returns the health status
    from omnidb_server.server import _acquire_home_lock, _release_home_lock

    fd = _acquire_home_lock(tmp_path)
    try:
        assert not isinstance(fd, OSError)
        assert cli.main(["doctor", "--home", str(tmp_path)]) == 4
        report = json.loads(capsys.readouterr().out)
        assert report["ok"] is False
        assert "held by a running server" in report["checks"]["lock"]["detail"]
    finally:
        _release_home_lock(fd)


def test_skill_export_import_list_roundtrip(tmp_path, capsys):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    bundle = tmp_path / "release.zip"

    store = make_store(src)
    store.remember("deploy checklist: build, test, tag, push", kind="procedural",
                   metadata={"skill": "release"}, trigger_patterns=["deploy"])
    store.remember("release cadence is every other Tuesday", metadata={"skill": "release"})
    store.close()

    assert cli.main(["skill", "export", "release", str(bundle), "--home", str(src)]) == 0
    assert json.loads(capsys.readouterr().out)["memories"] == 2
    assert bundle.is_file()

    assert cli.main(["skill", "list", "--home", str(dst)]) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert cli.main(["skill", "import", str(bundle), "--home", str(dst)]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported == {"name": "release", "memories": 2, "triggers": 1}

    assert cli.main(["skill", "list", "--home", str(dst)]) == 0
    assert json.loads(capsys.readouterr().out) == [{"name": "release", "memories": 2}]

    reopened = make_store(dst)
    try:
        hits = reopened.recall("deploy checklist", k=5)
        assert hits and hits[0]["metadata"]["skill_trigger"] is True
    finally:
        reopened.close()


def test_skill_import_refuses_duplicate_and_overwrite_works(tmp_path, capsys):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    bundle = tmp_path / "s.zip"
    store = make_store(src)
    store.remember("rotate the api keys quarterly", kind="procedural",
                   metadata={"skill": "ops"}, trigger_patterns=["rotate"])
    store.close()

    assert cli.main(["skill", "export", "ops", str(bundle), "--home", str(src)]) == 0
    capsys.readouterr()
    assert cli.main(["skill", "import", str(bundle), "--home", str(dst)]) == 0
    capsys.readouterr()

    rc = cli.main(["skill", "import", str(bundle), "--home", str(dst)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "[OMN-005]" in err

    assert cli.main(["skill", "import", str(bundle), "--overwrite", "--home", str(dst)]) == 0
    assert json.loads(capsys.readouterr().out)["memories"] == 1
    reopened = make_store(dst)
    try:
        assert reopened.stats()["live"] == 1
    finally:
        reopened.close()


def test_skill_export_unknown_name_yields_empty_bundle(tmp_path, capsys):
    src = tmp_path / "src"
    bundle = tmp_path / "empty.zip"
    store = make_store(src)
    store.remember("untagged row")
    store.close()
    assert cli.main(["skill", "export", "nope", str(bundle), "--home", str(src)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == {"name": "nope", "memories": 0, "path": str(bundle)}


def test_query_stats_prints_json_with_default_window(tmp_path, capsys):
    # query_log is opt-in per home; seed rows through a logging store first
    store = make_store(tmp_path, query_log=True)
    store.remember("kubernetes ingress runbook")
    store.close()

    assert cli.main(["query-stats", "--home", str(tmp_path)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats == {"window_hours": 24.0, "tenant": "default", "total": 0,
                         "by_kind": {}, "top_queries": []}

    assert cli.main(["query-stats", "--hours", "0", "--home", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["window_hours"] == 0.0
