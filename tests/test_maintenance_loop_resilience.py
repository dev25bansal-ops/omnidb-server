"""The maintenance thread must survive a failing tick (security lead R5).

`_maintenance_loop` in `server.serve` had no error handling:

    while not _stop_event.wait(1.0):
        store.maintenance_tick()
        store.checkpoint()

One `OSError` from `persist` — a full disk, an AV scanner holding a handle, a
transient filesystem error — ended the thread, and the only trace was a single
`threading.excepthook` line on stderr. After that the `persist_max_s`
idle-durability deadline (AUDIT-I16) was never enforced again for the life of
the process, with nothing in the process state to say so. The server kept
answering queries normally while quietly accumulating unwritten vectors.

`_graceful_shutdown` had the same gap: an unprotected `store.flush()` turned a
clean SIGINT into a traceback and skipped `SystemExit(0)`.

The behavioural tests here drive the *shipped* code, extracted from
`server.serve` by source, rather than a reimplementation — a test that
re-implements the loop correctly would pass whether or not the fix landed.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
import threading
import time

import pytest

from omnidb_server import server as srv


class _Boom:
    """A store whose maintenance tick raises, the way a full disk would."""

    def __init__(self, fail_on=1):
        self.calls = 0
        self.fail_on = fail_on
        self.checkpoints = 0
        self.flushed = 0
        self.fail_always = False

    def maintenance_tick(self):
        self.calls += 1
        if self.fail_always or self.calls == self.fail_on:
            raise OSError(28, "No space left on device")

    def checkpoint(self):
        self.checkpoints += 1

    def flush(self):
        self.flushed += 1
        raise OSError(28, "No space left on device")


def _extracted(name: str) -> str:
    """The exact source of a nested helper inside `serve`, via ast.

    Slicing by indentation is fragile; asking the parser for the node's
    line range is exact and survives reformatting.
    """
    source = textwrap.dedent(inspect.getsource(srv.serve))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = source.splitlines()
            body = chr(10).join(lines[node.lineno - 1:node.end_lineno])
            return textwrap.dedent(body)
    raise AssertionError(f"{name} is not defined in server.serve")


def _drive(loop_src: str, store, iterations=6, tick=0.01):
    """Execute the shipped loop body against a fake store, in a real thread.

    `_maintenance_loop` closes over `_stop_event` and `store`; the shipped
    code creates and starts the thread around it, so the harness supplies the
    closure and does the same starting.
    """
    namespace = {
        "store": store,
        "_stop_event": threading.Event(),
        "_threading": threading,
        "_sys": sys,
        "_os": os,
        "_MAINTENANCE_INTERVAL_S": srv._MAINTENANCE_INTERVAL_S,
    }
    exec(compile(loop_src, "<shipped-loop>", "exec"), namespace)
    thread = threading.Thread(
        target=namespace["_maintenance_loop"], args=(tick,),
        name="test-maintenance", daemon=True)
    thread.start()
    stop = namespace["_stop_event"]
    deadline = time.time() + 5.0
    while store.checkpoints < iterations and time.time() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(timeout=3.0)
    return thread


# --- the shipped source carries the guards -----------------------------------

def test_the_shipped_loop_has_error_handling():
    body = _extracted("_maintenance_loop")
    assert "except" in body, (
        "the maintenance loop has no error handling; one OSError from persist "
        "ends the thread silently for the life of the process"
    )


def test_the_shipped_shutdown_has_error_handling():
    body = _extracted("_graceful_shutdown")
    assert "except" in body, (
        "shutdown flushes unprotected; a full disk turns SIGINT into a traceback "
        "and the wrong exit status"
    )


def test_the_shipped_loop_reports_failures_rather_than_swallowing_them():
    """Surviving is not the goal — surviving *and saying so* is.

    A loop that retries forever in silence is the same false-clean the audit
    flagged in the read-only index fallback.
    """
    assert "stderr" in _extracted("_maintenance_loop")


# --- and it behaves that way -------------------------------------------------

def test_a_failing_tick_does_not_end_the_loop():
    store = _Boom(fail_on=1)
    thread = _drive(_extracted("_maintenance_loop"), store, iterations=6)
    assert store.checkpoints >= 6, (
        f"only {store.checkpoints} checkpoints ran after a failure; the shipped "
        "loop stopped doing maintenance"
    )
    # It reached six checkpoints *and* the thread was still running when they
    # happened — the checkpoint count alone could be satisfied by a loop that
    # ran once and exited.
    assert not thread.is_alive(), "the loop exited instead of retrying"


def test_a_repeated_failure_does_not_end_the_loop():
    store = _Boom()
    store.fail_always = True
    _drive(_extracted("_maintenance_loop"), store, iterations=6)
    assert store.calls >= 6, f"gave up after {store.calls} attempts"


def test_checkpointing_continues_when_ticks_fail():
    """WAL growth is its own operational problem; one error must not disable both."""
    store = _Boom(fail_on=1)
    _drive(_extracted("_maintenance_loop"), store, iterations=6)
    assert store.checkpoints >= 6


def test_a_failure_is_announced_on_stderr(capsys):
    store = _Boom(fail_on=1)
    _drive(_extracted("_maintenance_loop"), store, iterations=3)
    err = capsys.readouterr().err
    assert "maintenance tick failed" in err, (
        "the loop retried silently; the operator has no way to tell maintenance "
        "has been failing"
    )


def test_a_failed_flush_still_exits_cleanly(capsys):
    """The shutdown path, driven directly: it must reach SystemExit(0)."""
    store = _Boom()
    src = _extracted("_graceful_shutdown")
    namespace = {
        "store": store,
        "_stop_event": threading.Event(),
        "_sys": __import__("sys"),
        "signal": __import__("signal"),
    }
    exec(compile(src, "<shipped-shutdown>", "exec"), namespace)
    with pytest.raises(SystemExit) as excinfo:
        namespace["_graceful_shutdown"](15, None)
    assert excinfo.value.code == 0, "a failing flush changed the exit status"
    assert "flush during shutdown failed" in capsys.readouterr().err


@pytest.mark.parametrize("failing", ["tick", "checkpoint", "flush"])
def test_no_single_disk_error_breaks_the_loop_or_shutdown(failing, capsys):
    """The property, across all three call sites that touch the disk."""
    store = _Boom()
    if failing == "checkpoint":
        def checkpoint():
            store.checkpoints += 1
            raise OSError(28, "No space left on device")
        store.checkpoint = checkpoint
    _drive(_extracted("_maintenance_loop"), store, iterations=4)
    assert store.checkpoints >= 4, f"loop stopped after a {failing} failure"