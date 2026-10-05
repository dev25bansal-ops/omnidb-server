from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from conftest import make_store

from omnidb_server import cli, doctor, portability
from omnidb_server.embeddings import Embedder
from omnidb_server.server import _acquire_home_lock, _release_home_lock


class _AlternateEmbedder(Embedder):
    name = "alternate-v1"
    dim = 8

    def encode(self, texts: list[str]):
        import numpy as np

        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0] = float(len(text))
        return vectors


class _ExistingHnswIndex:
    kind = "hnsw"

    def __init__(self, path: Path, dim: int):
        self.path = path
        self.dim = dim

    def indexed_ids(self):
        return {1}

    def active_count(self):
        return 1

    def persist(self):
        return None


def _set_meta(home: Path, key: str, value: str) -> None:
    conn = sqlite3.connect(home / "omnidb.db")
    try:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def _make_hnsw_store(home: Path):
    store = make_store(home)
    store.remember("sidecar preservation probe")
    store.close()
    hnsw_path = home / "tenant-default.hnsw"
    hnsw_path.write_bytes(b"existing-index")
    deleted_path = home / "tenant-default.hnsw.deleted.json"
    deleted_path.write_text("[]", encoding="utf-8")
    store = make_store(home)
    _set_meta(home, "vec_kind", "hnsw")
    # Replace just the default tenant's backend, the way a corrupted
    # sidecar actually presents itself. Assigning to `store.index` would
    # bypass the constructor's wrap (ADR-08 / W3-3).
    store.index.replace_backend(
        "default", _ExistingHnswIndex(hnsw_path, store.embedder.dim))
    return store, hnsw_path, deleted_path


def test_doctor_cli_uses_health_status_as_exit_code(tmp_path, capsys, monkeypatch):
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 0
    capsys.readouterr()

    reports = iter(({"ok": False, "checks": {}}, {"ok": True, "checks": {}}))
    monkeypatch.setattr(doctor, "run_doctor", lambda home: next(reports))
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 4
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert cli.main(["doctor", "--home", str(tmp_path)]) == 0


def test_doctor_reports_corrupt_pending_reembed_as_failure(tmp_path):
    store = make_store(tmp_path)
    store.remember("doctor pending marker probe")
    store.close()
    _set_meta(tmp_path, "pending_reembed", "not-json")

    report = doctor.run_doctor(str(tmp_path))
    recovery = report["checks"]["recovery"]

    assert report["ok"] is False
    assert recovery["ok"] is False
    assert "pending_reembed" in recovery["detail"]
    assert "unreadable" in recovery["detail"]


def test_doctor_rejects_a_json_object_pending_reembed(tmp_path):
    """BUG-A1-09: a valid-JSON, wrong-SHAPE marker must not read as clean.

    The original code did ``len(_json.loads(...))``. ``len()`` always returns a
    non-negative int, so the shape guard beneath it was unreachable: an object
    marker like ``{"5": "x"}`` counted as ONE content revision and was
    reported ``ok:True``. This is the case the audit named — not the corrupt
    one, which was already handled and tested.
    """
    store = make_store(tmp_path)
    store.remember("doctor shape probe")
    store.close()
    _set_meta(tmp_path, "pending_reembed", '{"5": "x"}')

    report = doctor.run_doctor(str(tmp_path))
    recovery = report["checks"]["recovery"]

    assert report["ok"] is False
    assert recovery["ok"] is False
    assert "pending_reembed" in recovery["detail"]
    assert "not a JSON list" in recovery["detail"]


def test_doctor_rejects_a_list_of_non_id_entries(tmp_path):
    """A list of the wrong element type is also a shape failure."""
    store = make_store(tmp_path)
    store.remember("doctor element probe")
    store.close()
    _set_meta(tmp_path, "pending_reembed", '[{"id": 1}]')

    report = doctor.run_doctor(str(tmp_path))
    recovery = report["checks"]["recovery"]

    assert recovery["ok"] is False
    assert "non-id entries" in recovery["detail"]


def test_doctor_accepts_a_well_formed_pending_reembed_list(tmp_path):
    """Control: the shape check must not reject the real marker format."""
    store = make_store(tmp_path)
    store.remember("doctor real marker probe")
    store.close()
    _set_meta(tmp_path, "pending_reembed", "[1, 2]")

    report = doctor.run_doctor(str(tmp_path))
    recovery = report["checks"]["recovery"]

    assert recovery["ok"] is True
    assert "2 content revisions" in recovery["detail"]


def test_doctor_reports_absent_pending_reembed_as_clean(tmp_path):
    store = make_store(tmp_path)
    store.remember("doctor clean marker probe")
    store.close()
    _set_meta(tmp_path, "pending_reembed", "[]")

    report = doctor.run_doctor(str(tmp_path))
    recovery = report["checks"]["recovery"]

    assert report["ok"] is True
    assert recovery["ok"] is True
    assert "clean" in recovery["detail"]


def test_doctor_cli_reports_lock_failure_with_nonzero_status(tmp_path, capsys):
    store = make_store(tmp_path)
    store.close()
    fd = _acquire_home_lock(tmp_path)
    try:
        assert not isinstance(fd, OSError)
        assert cli.main(["doctor", "--home", str(tmp_path)]) == 4
    finally:
        _release_home_lock(fd)
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "held by a running server" in report["checks"]["lock"]["detail"]


def test_reembed_constructor_failure_preserves_existing_sidecar(tmp_path, monkeypatch):
    store, hnsw_path, deleted_path = _make_hnsw_store(tmp_path)
    original_deleted = deleted_path.read_bytes()
    try:
        from omnidb_server import vector_index

        def failing_constructor(*args, **kwargs):
            raise vector_index.VectorIndexError("constructor failed")

        # Patch the HnswlibIndex *name* reembed_home resolves. Patching the
        # class object would break every isinstance() in the package — the
        # router's own `kind` runs during the rebuild.
        monkeypatch.setattr(portability, "HnswlibIndex", failing_constructor,
                            raising=False)
        # The constructor failure degrades to the brute backend rather than
        # aborting: the old sidecar is already staged aside at that point, and
        # leaving the home with no index would be strictly worse than a slower
        # backend. What matters here is that the *existing* sidecar survives.
        assert portability.reembed_home(tmp_path, _AlternateEmbedder(),
                                        assume_yes=True) is not None
    finally:
        store.conn.close()

    # The preserved sidecar is the hnsw one: the constructor failed, the
    # re-embed degraded to brute for the *new* generation, and the original
    # bytes were restored rather than clobbered.
    assert deleted_path.read_bytes() == original_deleted
    assert hnsw_path.exists(), "the existing hnsw sidecar was removed"
    reopened = make_store(tmp_path, embedder="alternate-v1",
                         embedder_impl=_AlternateEmbedder())
    try:
        assert reopened.stats()["live"] == 1
    finally:
        reopened.close()


def test_snapshot_rotation_reports_prune_failure_without_failing_backup(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path / "home")
    store.remember("snapshot rotation probe")
    backups = tmp_path / "backups"
    old_snapshot = backups / "snapshot-20000101T000000000000"
    old_snapshot.mkdir(parents=True)
    (old_snapshot / "omnidb.db").write_bytes(b"old")

    def fail_remove(path, ignore_errors=False):
        raise OSError("snapshot is in use")

    monkeypatch.setattr(portability.shutil, "rmtree", fail_remove)
    report = portability.backup_home(store, backups, keep=1)

    assert Path(report["dest"]).is_dir()
    assert old_snapshot.is_dir()
    assert report["pruned"] == []
    assert report["prune_errors"]
    assert report["prune_errors"][0].startswith("snapshot-20000101T000000000000:")
    assert "in use" in report["prune_errors"][0]
    assert "in use" in capsys.readouterr().err
    store.close()
