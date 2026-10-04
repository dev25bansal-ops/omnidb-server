"""F14 - a failed latency run must be visible, not filtered out.

between_process_latency kept a crashed run in the raw list and then dropped it
before computing the spread, and render() printed a header hardcoded to REPEATS.
The result was a published table reading "5 fresh processes" above three p50
values, with no n, no error and no retry - so a reader could not tell that runs
had been lost. That is the mechanism behind F1.

The fix is that the spread carries its own n. These tests pin that the n reaches
the reader, that a clean cell is unaffected, and that an artifact written before
the new keys existed still renders.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

# the bench harnesses import each other by bare name, so bench/ must be on the
# path before the module under test is imported
BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))

import research_bench_v3 as rb


def _stub_results(mode_data):
    """One entry per dataset. A dataset with no embedder key skips the nDCG
    table entirely, so only the latency block is exercised.

    mode_data maps an engine name to that engine whole latency record."""
    out = {}
    for ds in rb.DATASETS:
        out[ds] = {"_ndocs": 100, f"_latency-{ds}": dict(mode_data)}
    return out


@pytest.fixture(autouse=True)
def _never_touch_the_published_artifact(tmp_path, monkeypatch):
    """render() writes to the module constant OUT_MD, which is the PUBLISHED
    bench/research-v3-results.md. Left alone, running these tests overwrites the
    real artifact with stub data. Point it at a temp file instead.
    """
    monkeypatch.setattr(rb, "OUT_MD", tmp_path / "rendered.md")


def _render(results):
    """render() emits the markdown to OUT_MD, not to stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        rb.render(results)
    return rb.OUT_MD.read_text(encoding="utf-8")


def _clean(n=5):
    p50s = [100.0 + i for i in range(n)]
    return {"runs": [{"p50_ms": p} for p in p50s],
            "spread": {"p50s": p50s, "n_ok": n, "n_requested": n,
                       "n_failed": 0, "errors": []}}


def _two_failed_of_five():
    err = "sqlite3.OperationalError: disk I/O error"
    runs = [{"error": err}, {"p50_ms": 97.1}, {"error": err},
            {"p50_ms": 110.0}, {"p50_ms": 390.2}]
    return {"runs": runs,
            "spread": {"p50s": [97.1, 110.0, 390.2], "n_ok": 3,
                       "n_requested": 5, "n_failed": 2, "errors": [err, err]}}


# --- the spread must carry its own n -------------------------------------


def test_between_process_latency_records_failures(monkeypatch, tmp_path):
    """The old code built the spread from the survivors and dropped the count."""
    calls = {"n": 0}

    class _R:
        def __init__(self, rc, out="", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(*a, **k):
        calls["n"] += 1
        if calls["n"] % 5 in (2, 4):
            return _R(1, "", "boom: sqlite3.OperationalError: disk I/O error")
        return _R(0, '{"p50_ms": 10.0, "p95_ms": 12.0, "mean_ms": 11.0}')

    monkeypatch.setattr(rb.subprocess, "run", fake_run)
    monkeypatch.setattr(rb.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    out = rb.between_process_latency("fiqa", "minilm", "repo")

    for mode in ("omnidb", "qdrant"):
        sp = out[mode]["spread"]
        assert sp["n_requested"] == 5
        assert sp["n_ok"] == 3, mode
        assert sp["n_failed"] == 2, mode
        assert len(sp["p50s"]) == 3, mode
        assert all("disk I/O error" in e for e in sp["errors"]), mode
        assert len(out[mode]["runs"]) == 5, "raw runs must not be discarded"


# --- and it must reach the reader ----------------------------------------


def test_render_states_how_many_runs_were_lost():
    out = _render(_stub_results({"omnidb": _clean(),
                                 "qdrant": _two_failed_of_five()}))
    assert "3/5 processes" in out
    assert "2 of 5 runs FAILED" in out
    assert "disk I/O error" in out
    assert "5 requested; failed runs are listed below" in out


def test_a_clean_cell_keeps_the_plain_header():
    """No failure, no new wording - the change must be invisible where it
    does not apply."""
    out = _render(_stub_results({"omnidb": _clean(), "qdrant": _clean()}))
    assert "5 fresh processes" in out
    assert "5/5 processes" in out
    assert "FAILED" not in out


def test_a_cell_with_no_successful_runs_is_said_so():
    dead = {"runs": [{"error": "boom"}] * 5, "spread": None}
    out = _render(_stub_results({"omnidb": _clean(), "qdrant": dead}))
    assert "NO successful runs" in out


def test_an_artifact_written_before_the_new_keys_still_renders():
    """The shipped research-v3-results.json has no n_ok/n_requested. It must
    render rather than raise, which is why the lookups have defaults."""
    legacy = {"runs": [{"p50_ms": 1.0}, {"p50_ms": 2.0}, {"p50_ms": 3.0}],
              "spread": {"p50s": [1.0, 2.0, 3.0]}}
    out = _render(_stub_results({"omnidb": legacy, "qdrant": legacy}))
    assert "3/5 processes" in out
    assert "FAILED" not in out


def test_the_real_archived_fiqa_cell_renders_its_two_lost_runs():
    """The shape actually shipped by the pre-F1 artifact: five entries in `runs`,
    two of them error records, and a spread holding only p50s. The old renderer
    printed 3/5 in the body while the header claimed "5 fresh processes" and
    listed nothing, because n_failed defaulted to 0.
    """
    archived = {
        "runs": [{"p50_ms": 97.1}, {"p50_ms": 110.0}, {"p50_ms": 390.1},
                 {"error": "Traceback...\nsqlite3.OperationalError: disk I/O error"},
                 {"error": "Traceback...\nhttp.client.IncompleteRead: IncompleteRead(11479004 bytes read)"}],
        "spread": {"p50s": [97.1, 110.0, 390.1]},
    }
    out = _render(_stub_results({"omnidb": _clean(), "qdrant": archived}))
    assert "3/5 processes" in out
    assert "failed runs are listed below" in out
    assert "disk I/O error" in out
    assert "IncompleteRead" in out


def test_a_failed_run_with_no_stderr_text_renders_instead_of_raising():
    """`"".splitlines()[-1]` is an IndexError, and a child killed by the job
    object limit can return empty stderr, so one malformed record must not take
    the whole report down."""
    for bad in ("", None, "   \n  "):
        cell = {"runs": [{"p50_ms": 1.0}] * 4 + [{"error": bad}], "spread": None}
        out = _render(_stub_results({"omnidb": cell}))
        assert "4/5 processes" in out, repr(bad)
        assert "1 of 5 runs FAILED" in out, repr(bad)


def test_lost_runs_are_never_announced_without_a_reason():
    """A spread can claim losses whose text did not survive. Printing
    "excluded from the numbers above:" followed by nothing is the same
    concealment in a new place, so a placeholder is required."""
    hollow = {"runs": [{"p50_ms": float(i)} for i in range(3)],
              "spread": {"p50s": [0.0, 1.0, 2.0], "n_ok": 3,
                         "n_requested": 5, "n_failed": 2}}
    out = _render(_stub_results({"omnidb": hollow}))
    assert "2 of 5 runs FAILED" in out
    assert "(error text not stored)" in out


def test_a_short_runs_list_is_loss_not_a_smaller_experiment():
    """The pre-F14 collector deleted failed runs before writing, so a cell holding
    three records is evidence that two were lost. Falling back to
    n_ok + n_failed would print 3/3 and make the loss invisible again - which is
    the regression these F33 tests exist to prevent."""
    truncated = {"runs": [{"p50_ms": 1.0}, {"p50_ms": 2.0}, {"p50_ms": 3.0}],
                 "spread": {"p50s": [1.0, 2.0, 3.0]}}
    prov = rb._run_provenance(truncated)
    assert prov["n_requested"] == rb.REPEATS == 5
    assert prov["n_failed"] == 0  # unknown, and must never be read as "nothing was lost"
    out = _render(_stub_results({"omnidb": truncated}))
    assert "3/5 processes" in out
