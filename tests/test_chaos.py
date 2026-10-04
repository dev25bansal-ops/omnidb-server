"""Crash-window chaos harness (analysis §6.2): kill a writer mid-insert stream
and prove the reopen path self-heals — no torn rows, reconciled index, recall
returns well-formed hits."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from omnidb_server.config import Config
from omnidb_server.server import build_stack

REPO = Path(__file__).resolve().parents[1]
ITERATIONS = 8
INSERTS_PER_CHILD = 400
MIN_DELAY_S, MAX_DELAY_S = 0.04, 0.22
READY_TIMEOUT_S = 45.0
KILL_WAIT_TIMEOUT_S = 15.0
POLL_INTERVAL_S = 0.005

_CHILD_SCRIPT = """
import sys
from pathlib import Path

from omnidb_server.config import Config
from omnidb_server.server import build_stack

home = sys.argv[1]
store = build_stack(Config.from_env(home))
Path(home, "ready").write_text("ok", encoding="utf-8")
for i in range(%d):
    store.remember(f"chaos probe record {i} payload token{i}")
    print(f"committed {i + 1}", flush=True)
store.close()
print("DONE", flush=True)
""" % INSERTS_PER_CHILD


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMNIDB_EMBEDDER"] = "hashing"
    env["OMNIDB_INDEX"] = "brute"
    return env


def _wait_for_ready(sentinel: Path, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if sentinel.exists():
            return
        if proc.poll() is not None:
            pytest.fail(
                "chaos child exited before signalling ready "
                f"(rc={proc.returncode}): {log_path.read_text(encoding='utf-8')[-2000:]}"
            )
        time.sleep(POLL_INTERVAL_S)
    pytest.fail("chaos child never signalled ready")


def _run_child_until_kill(home: Path, iteration: int) -> Path:
    log_path = home.parent / "child.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_SCRIPT, str(home)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_child_env(),
            cwd=str(REPO),
        )
        try:
            # killing before build_stack returns would test interpreter startup,
            # not the DB/index crash window
            _wait_for_ready(home / "ready", proc, log_path)
            random.seed(iteration)
            time.sleep(random.uniform(MIN_DELAY_S, MAX_DELAY_S))
        finally:
            proc.kill()
            proc.wait(timeout=KILL_WAIT_TIMEOUT_S)
    return log_path


@pytest.mark.parametrize("iteration", range(ITERATIONS))
def test_crash_window_chaos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, iteration: int):
    monkeypatch.setenv("OMNIDB_EMBEDDER", "hashing")
    monkeypatch.setenv("OMNIDB_INDEX", "brute")

    home = tmp_path / "home"
    _run_child_until_kill(home, iteration)

    reopened = build_stack(Config.from_env(home))
    try:
        first_stats = reopened.stats()
        reconciled_on_first_reopen = first_stats["reconciled_at"] is not None
        assert reopened.check()["fts_integrity"] == "ok"
        result = reopened.search(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT id) AS d FROM memories"
, unscoped=True)
        committed, distinct_ids = result["rows"][0]
        assert committed == distinct_ids
    finally:
        reopened.close()

    settled = build_stack(Config.from_env(home))
    try:
        stats = settled.stats()
        assert stats["live"] == committed
        assert stats["index"]["count"] == stats["live"] or reconciled_on_first_reopen
        for hit in settled.recall("chaos probe record", k=5):
            assert {"id", "content", "score"} <= set(hit)
            assert isinstance(hit["score"], (int, float))
    finally:
        settled.close()
