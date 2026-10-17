"""Watchdog for the CE held-out study test worker.

The host is a laptop that sleeps on lid-close; each sleep suspends the GPU and
kills the worker (observed three times: one reboot, two clean stops with no
traceback and no reboot between them). This script is meant to be run by a
Windows scheduled task every few minutes. It:

  1. Exits immediately if a ce_study_runner.py python process is alive.
  2. If the worker is absent but the run lock still exists, resumes the test
     phase from the last completed cell (the runner's --resume is safe: it
     re-scores only keys not already present and refuses a stale lock whose
     owner PID is still alive).
  3. Logs every decision to the run directory so the timeline is reconstructable.

It never starts a worker while one is alive, so repeated runs cannot
double-launch. It also refuses to act if the lock's owner PID is still alive
(the runner would reject the resume anyway, but we avoid spawning it needlessly).
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RUN = Path(r"D:\OMNIDB\omnidb-server")
RUN_ID = "ce-study-v1-20261009"
RUN_DIR = RUN / "bench" / "ce-study-runs" / RUN_ID
LOCK = RUN_DIR / "runner.lock"
LOG = RUN_DIR / "watchdog.log"
TEST_LOG = RUN_DIR / "test.log"


def _pid_alive(pid: int) -> bool:
    """True if `pid` is a live process (Windows via OpenProcess, else os.kill)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _worker_pids() -> list[int]:
    """PIDs of python processes whose command line mentions ce_study_runner.py."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*ce_study_runner.py*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            text=True, stderr=subprocess.DEVNULL, timeout=30)
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def _log(msg: str) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {msg}\n")


def _check_runaway_memory() -> None:
    """Kill any process ballooning virtual memory and starving CUDA allocations."""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process -Name '*OneDrive*' -ErrorAction SilentlyContinue | "
             "Where-Object { $_.PagedMemorySize64 -gt 2GB } | "
             "Select-Object -ExpandProperty Id"],
            text=True, stderr=subprocess.DEVNULL, timeout=15)
        for pid_str in out.split():
            if pid_str.strip().isdigit():
                pid = int(pid_str.strip())
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     f"Stop-Process -Id {pid} -Force"],
                    timeout=15)
                _log(f"killed runaway OneDrive process {pid} exceeding 2GB memory")
    except Exception:
        pass


TOTAL_TEST_ROWS = 224895  # 5 rerankers * 44979 test queries


def _is_test_complete() -> bool:
    results_path = RUN_DIR / "test-results.jsonl"
    if results_path.exists():
        return True
    scores_path = RUN_DIR / "test-scores.jsonl"
    if not scores_path.exists():
        return False
    try:
        # Check row count
        count = sum(1 for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip())
        return count >= TOTAL_TEST_ROWS
    except Exception:
        return False


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _check_runaway_memory()

    # User stop flag
    if (RUN_DIR / "stop.flag").exists():
        _log("stop.flag present; will not resume")
        return 0

    pids = _worker_pids()
    alive = [p for p in pids if _pid_alive(p)]
    if alive:
        _log(f"worker alive (pid {','.join(map(str, alive))}), no-op")
        return 0

    if _is_test_complete():
        _log("test phase complete; nothing to resume")
        return 0

    owner = None
    if LOCK.exists():
        try:
            owner = int(json.loads(LOCK.read_text(encoding="utf-8")).get("pid"))
        except Exception:
            owner = None
        if owner is not None and _pid_alive(owner):
            _log(f"no worker, but lock owner pid {owner} is alive; not resuming")
            return 0

    # Worker is absent and test phase is not complete -> resume
    _log("worker absent and test incomplete; resuming test phase")
    cmd = [sys.executable, str(HERE / "ce_study_runner.py"), "test",
           "--run-id", RUN_ID, "--batch-size", "32", "--resume"]
    flags = 0
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        flags = 0x00000008 | 0x00000200
    with TEST_LOG.open("a", encoding="utf-8") as out:
        p = subprocess.Popen(cmd, cwd=str(RUN), stdout=out,
                             stderr=subprocess.STDOUT, creationflags=flags)
    _log(f"resume launched (pid {p.pid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
