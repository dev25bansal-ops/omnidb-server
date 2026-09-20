"""Single-writer contention: the home lock must fail a second serve FAST and
cleanly (clear stderr, non-zero exit), while short-lived CLI readers
(`stats`) stay welcome while the server runs.

The lock is a real OS-level advisory lock on every platform (flock / msvcrt),
so the second-serve spawn runs on Windows too (ARCH-05).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from omnidb_server import cli

REPO = Path(__file__).resolve().parents[1]
READY_TIMEOUT_S = 15.0
SECOND_SERVE_TIMEOUT_S = 10.0
POLL_INTERVAL_S = 0.02


def _serve_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMNIDB_EMBEDDER"] = "hashing"
    # OMNIDB_VECTOR_INDEX is the key Config actually reads; OMNIDB_INDEX is
    # set too so the documented spelling matches what older ops guides use
    env["OMNIDB_VECTOR_INDEX"] = "brute"
    env["OMNIDB_INDEX"] = "brute"
    return env


def _stop_serve(proc: subprocess.Popen, home: Path) -> None:
    try:
        if proc.stdin is not None:
            proc.stdin.close()  # stdio server exits on EOF, runs its finally
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()
        proc.wait(timeout=10)
    finally:
        try:  # Windows has no atexit unlink; don't strand the sidecar
            (home / ".lock").unlink()
        except OSError:
            pass


def _wait_for_lock(home: Path, proc: subprocess.Popen, log_path: Path) -> None:
    # AUDIT-I02: the home lock is now acquired BEFORE the writable stack is
    # built, so lock-file existence no longer implies the server finished
    # initialization. The startup banner is the readiness signal: it is
    # printed only after the lock is held and the store is open.
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if (home / ".lock").exists() and "omnidb-server: serving home=" in \
                log_path.read_text(encoding="utf-8", errors="replace"):
            return
        if proc.poll() is not None:
            pytest.fail(
                f"serve exited before acquiring the home lock (rc={proc.returncode}): "
                f"{log_path.read_text(encoding='utf-8')[-2000:]}"
            )
        time.sleep(POLL_INTERVAL_S)
    pytest.fail(f"serve never created {home / '.lock'} or printed its banner")


def _spawn_serve(home: Path, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "omnidb_server", "serve", "--home", str(home)],
        stdin=subprocess.PIPE,  # held open so the stdio loop keeps running
        stdout=log,
        stderr=subprocess.STDOUT,
        env=_serve_env(),
        cwd=str(REPO),
    )


def test_reader_cli_succeeds_and_lock_held_while_serve_runs(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    log_path = tmp_path / "serve.log"
    proc = _spawn_serve(home, log_path)
    try:
        _wait_for_lock(home, proc, log_path)

        # the one-line startup banner (E-07) must be on stderr for support
        log_text = log_path.read_text(encoding="utf-8")
        assert "omnidb-server: serving home=" in log_text
        assert "embedder=hashing" in log_text

        # CLI tools are short-lived readers; they take no serve lock and must
        # succeed against a live server (WAL gives them a consistent snapshot)
        monkeypatch.setenv("OMNIDB_EMBEDDER", "hashing")
        monkeypatch.setenv("OMNIDB_VECTOR_INDEX", "brute")
        assert cli.main(["stats", "--home", str(home)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "live" in payload and "by_kind" in payload

        assert (home / ".lock").exists(), "lock must be held for the life of serve"
    finally:
        _stop_serve(proc, home)


def test_second_serve_on_locked_home_fails_fast(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    log_path = tmp_path / "serve.log"
    proc = _spawn_serve(home, log_path)
    try:
        _wait_for_lock(home, proc, log_path)

        t0 = time.monotonic()
        try:
            second = subprocess.run(
                [sys.executable, "-m", "omnidb_server", "serve", "--home", str(home)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env=_serve_env(),
                cwd=str(REPO),
                timeout=SECOND_SERVE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("second serve hung instead of failing fast on the held lock")
        elapsed = time.monotonic() - t0

        stderr = second.stderr.decode(errors="replace")
        assert "another process" in stderr, (
            f"conflict must say so on stderr (rc={second.returncode}, stderr={stderr[-500:]})"
        )
        # serve()'s lock-conflict exit code (3) is propagated by cli.main()
        assert second.returncode != 0
        assert elapsed < SECOND_SERVE_TIMEOUT_S
    finally:
        _stop_serve(proc, home)
