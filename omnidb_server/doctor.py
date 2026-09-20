"""Environment health report behind `omnidb-server doctor` (informational, never exits non-zero)."""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import sys
from importlib.util import find_spec
from pathlib import Path

from .config import Config
from .embeddings import HashingEmbedder

_MIN_FREE_BYTES = 64 * 1024 * 1024


def _resolve_home(home: str | None) -> Path:
    if home:
        return Path(home)
    env = os.environ.get("OMNIDB_HOME")
    return Path(env) if env else Path.home() / ".omnidb"


def _optional_dep(module: str, role: str) -> dict:
    if find_spec(module) is not None:
        return {"ok": True, "detail": f"installed ({role})"}
    return {"ok": True, "detail": f"not installed (optional; {role})"}


def _fts5_check() -> dict:
    try:
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(c)")
        finally:
            probe.close()
    except sqlite3.Error as exc:
        return {"ok": False, "detail": f"unavailable: {exc}"}
    return {"ok": True, "detail": "available"}


def _index_check(target: Path) -> dict:
    db_path = target / "omnidb.db"
    if not db_path.is_file():
        return {"ok": True, "detail": "no database yet (run `omnidb-server init`)"}
    try:
        # mode=ro never creates or migrates the file; a pure peek at meta
        ro = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            row = ro.execute("SELECT value FROM meta WHERE key = 'vec_kind'").fetchone()
        finally:
            ro.close()
    except sqlite3.Error as exc:
        return {"ok": False, "detail": f"cannot read {db_path.name}: {exc}"}
    kind = row[0] if row else None
    return {"ok": True, "detail": f"kind={kind or 'unset'} (from meta)"}


def _recovery_check(target: Path) -> dict:
    """AUDIT follow-up: surface pending recovery work (the markers behind
    AUDIT-I01/I03/I16) so operators can see what the next open will heal.
    Read-only peek; pending work is informational, never a doctor failure."""
    db_path = target / "omnidb.db"
    if not db_path.is_file():
        return {"ok": True, "detail": "no database yet"}
    try:
        ro = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            rows = dict(ro.execute("SELECT key, value FROM meta"))
        finally:
            ro.close()
    except sqlite3.Error as exc:
        return {"ok": False, "detail": f"cannot read meta: {exc}"}
    pending: list[str] = []
    if rows.get("index_dirty") == "1":
        pending.append("index_dirty (sidecar lagging rows)")
    try:
        import json as _json
        n_pending = len(_json.loads(rows.get("pending_reembed", "[]")))
    except ValueError:
        n_pending = -1
    if n_pending > 0:
        pending.append(f"pending_reembed ({n_pending} content revisions)")
    if rows.get("reembed_pending") is not None:
        pending.append("reembed_pending (interrupted embedder migration)")
    if rows.get("compaction_due") == "1":
        pending.append("compaction_due (scheduled index rebuild)")
    if not pending:
        return {"ok": True, "detail": "clean (no pending recovery work)"}
    return {"ok": True, "detail": "; ".join(pending) + " — next open reconciles from rows"}


def _home_check(target: Path) -> dict:
    if not target.exists():
        return {"ok": True, "detail": "missing (create with `omnidb-server init`)"}
    if not os.access(target, os.W_OK):
        return {"ok": False, "detail": f"not writable: {target}"}
    return {"ok": True, "detail": f"exists and writable: {target}"}


def _disk_check(target: Path) -> dict:
    probe_dir = target if target.exists() else next(
        (p for p in target.parents if p.exists()), Path.cwd()
    )
    try:
        free = shutil.disk_usage(probe_dir).free
    except OSError as exc:
        return {"ok": False, "detail": f"cannot measure: {exc}"}
    return {"ok": free >= _MIN_FREE_BYTES, "detail": f"{free} bytes free on {probe_dir}"}


def _lock_check(target: Path) -> dict:
    """Probe the single-writer lock by trying to take it (ARCH-05): the .lock
    file persists after a clean or dirty exit by design — only holding the
    OS lock means a server is running."""
    if not (target / "omnidb.db").is_file():
        return {"ok": True, "detail": "absent (no database yet)"}
    lock = target / ".lock"
    try:
        fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        return {"ok": False, "detail": f"cannot open {lock}: {exc}"}
    try:
        os.write(fd, b"0")
        os.lseek(fd, 0, os.SEEK_SET)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        return {"ok": True, "detail": "free (no server holds the home lock)"}
    except OSError:
        return {"ok": False, "detail": f"{lock} is held by a running server"}
    finally:
        os.close(fd)


def run_doctor(home: str | None) -> dict:
    """Gather environment checks as {"ok": bool, "checks": {name: {"ok", "detail"}}}.

    Pure: no prints, no writes, no model downloads. The embedder probe builds a
    HashingEmbedder directly (offline-safe) and only reports the configured
    setting; resolving cfg.embedder=fastembed for real would trigger a download.
    """
    checks: dict[str, dict] = {
        "python": {"ok": True, "detail": platform.python_version()},
        "platform": {"ok": True, "detail": platform.platform()},
        "sqlite": {"ok": True, "detail": f"sqlite3 {sqlite3.sqlite_version}"},
        "fts5": _fts5_check(),
        "hnswlib": _optional_dep("hnswlib", "vector index backend; brute-force fallback exists"),
        "fastembed": _optional_dep("fastembed", "semantic embedder; hashing fallback exists"),
    }

    target = _resolve_home(home)
    try:
        cfg = Config.load(home)
        target = Path(cfg.home)
    except Exception as exc:  # a broken config must not crash an informational report
        checks["embedder"] = {"ok": False, "detail": f"config error: {exc}"}
    else:
        probe = HashingEmbedder()
        checks["embedder"] = {
            "ok": True,
            "detail": (f"cfg.embedder={cfg.embedder!r}; offline resolution {probe.name} "
                       f"(dim {probe.dim}); fastembed model download not attempted"),
        }

    checks["index"] = _index_check(target)
    checks["recovery"] = _recovery_check(target)
    checks["home"] = _home_check(target)
    checks["disk"] = _disk_check(target)
    checks["lock"] = _lock_check(target)
    return {"ok": all(check["ok"] for check in checks.values()), "checks": checks}
