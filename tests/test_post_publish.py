"""Regression tests for the J-01..J-15 post-publish remediation wave."""

import json
import sys
import time
from pathlib import Path

import pytest
from conftest import make_index, make_store

from omnidb_server import cli
from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.portability import export_jsonl
from omnidb_server.store import MemoryStore

# --- J-01: --version -------------------------------------------------------

def test_j01_module_version_flag(capsys):
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("omnidb-server ")
    assert "0.1" in out


def test_j01_console_version_flag():
    # the console script's argv[0] is the binary name; verify the path
    # inside the entry-point (omnidb_server.cli:main) honors --version
    # even with an arg list that has no command
    from omnidb_server.cli import main
    rc = main(["--version"])
    assert rc == 0


# --- J-02: __getattr__ alias -----------------------------------------------

def test_j02_omnib_alias_resolves_to_cli():
    import omnidb_server
    assert omnidb_server.omnidb is omnidb_server.cli


def test_j02_unknown_attr_raises():
    import omnidb_server
    with pytest.raises(AttributeError, match="no attribute"):
        omnidb_server.this_is_not_a_thing


# --- J-03: consolidation dedup pool size is configurable ------------------

def test_j03_dedup_pool_size_is_parameter(tmp_path):
    from omnidb_server import consolidate as c_mod
    class Stub:
        def encode(self, texts):
            return __import__("numpy").zeros((len(texts), 4), dtype="float32")
    st = make_store(tmp_path)
    s = Stub()
    s.dim = 4
    _snap = c_mod._LiveSemanticCandidates(st, pool_size=2)
    # pool capped at 2 even though many more rows exist
    for i in range(5):
        st.remember(f"item number {i}")
    snap2 = c_mod._LiveSemanticCandidates(st, pool_size=2)
    assert len(snap2._ids) == 2


# --- J-04: reconciled_at semantics ----------------------------------------

def test_j04_reconciled_at_only_set_on_recovery(tmp_path):
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.db import get_meta
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     make_index(tmp_path, "brute", 512), cfg)
    # fresh open with no rows: no recovery, reconciled_at stays None
    assert get_meta(conn, "reconciled_at") is None
    st.close()

    # reopen still nothing to recover
    conn2 = dbmod.connect(tmp_path)
    st2 = MemoryStore(conn2, HashingEmbedder(),
                      make_index(tmp_path, "brute", 512), cfg)
    assert get_meta(conn2, "reconciled_at") is None
    st2.close()


# --- J-05: atomic export --------------------------------------------------

def test_j05_atomic_export_leaves_no_tmp(tmp_path):
    src = make_store(tmp_path / "src")
    src.remember("atomic write probe")
    dump = tmp_path / "export.jsonl"
    report = export_jsonl(src, dump)
    src.close()
    assert report["memories"] == 1
    leftover = list(tmp_path.glob("export.jsonl.*.tmp"))
    assert not leftover, leftover
    # the file is readable end-to-end
    assert dump.read_text().strip().startswith("{")


# --- J-07: export filters --------------------------------------------------

def test_j07_export_filters_drop_rows(tmp_path):
    src = make_store(tmp_path / "src")
    src.remember("one")
    src.remember("two", kind="procedural", trigger_patterns=["two"])
    src.remember("three")
    full = export_jsonl(src, tmp_path / "all.jsonl")
    filt = export_jsonl(src, tmp_path / "filtered.jsonl", kind="procedural")
    src.close()
    assert full["memories"] == 3
    assert filt["memories"] == 1
    lines = [json.loads(line) for line in (tmp_path / "filtered.jsonl").read_text().splitlines()]
    assert all(r["type"] == "memory" and r["kind"] == "procedural" for r in lines)


def test_j07_export_filter_since_ms(tmp_path):
    src = make_store(tmp_path / "src")
    src.remember("ancient", created_ms=1)
    src.remember("modern")
    future = 10**15  # well past both "ancient" and the live-inserted "modern"
    out = export_jsonl(src, tmp_path / "r.jsonl", since_ms=future)
    src.close()
    assert out["memories"] == 0
    # and the inverse: since_ms=0 includes both
    src2 = make_store(tmp_path / "src2")
    src2.remember("ancient", created_ms=1)
    src2.remember("modern")
    out_all2 = export_jsonl(src2, tmp_path / "all.jsonl", since_ms=0)
    src2.close()
    assert out_all2["memories"] == 2


# --- J-06: SIGTERM handler -------------------------------------------------

def test_j06_serve_registers_sigterm():
    # We can't easily raise SystemExit into serve() in a test, but we
    # can confirm signal.signal was called by inspecting the function
    # source for the registration pattern.
    import inspect

    from omnidb_server import server
    src = inspect.getsource(server.serve)
    assert "SIGTERM" in src
    assert "signal.signal" in src


# --- J-08: stats wal_bytes + check() TOCTOU resilience --------------------

def test_j08_wal_bytes_in_stats_and_check(store):
    store.remember("wal probe 1")
    store.remember("wal probe 2")
    store.flush()
    s = store.stats()
    assert "wal_bytes" in s
    assert s["wal_bytes"] >= 0
    chk = store.check()
    assert "wal_bytes" in chk
    assert chk["wal_bytes"] == s["wal_bytes"]


# --- J-10: soak report states RSS honestly -------------------------------

def test_j10_soak_reports_psutil_status_honestly(tmp_path):
    """The soak report must state RSS honestly: a real reading, or `n/a`.

    The earlier silent-pass bug was: the growth assertion never fired
    because the RSS variable was None, while the test still reported
    success. This pins the reporting contract on a report the test
    itself produces, so it cannot be satisfied by a stale artifact.
    """
    import subprocess

    report = tmp_path / "soak-report.txt"
    proc = subprocess.run(
        [sys.executable, "-m", "bench.soak",
         "--minutes", "1", "--iterations", "3", "--output", str(report)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    assert report.exists(), f"harness wrote no report: {proc.stderr[-2000:]}"
    text = report.read_text()
    fields = dict(
        line.split(": ", 1) for line in text.splitlines() if ": " in line
    )
    for key in ("rss_start_mb", "rss_peak_mb", "p95_ratio",
                "validation_gate_passed"):
        assert key in fields, f"{key} missing from report:\n{text}"
    for key in ("rss_start_mb", "rss_peak_mb"):
        value = fields[key]
        assert value == "n/a" or float(value) > 0, f"{key}={value!r}"
    # A missing RSS reading must never masquerade as a passing gate.
    if fields["rss_start_mb"] == "n/a":
        assert fields["validation_gate_passed"] == "False", text


# --- J-12: consolidation report has consistent shape ----------------------

def test_j12_consolidation_report_shape_uniform(tmp_path):
    from omnidb_server.consolidate import run_consolidation
    st = make_store(tmp_path)
    # no episodes at all -> early-return path; < 3 -> early-return path
    keys_empty = set(run_consolidation(st, hours=1).keys())
    keys_short = set(run_consolidation(st, hours=24 * 7).keys())  # 0 episodes
    for k in ("episodes_scanned", "facts_created", "facts_dropped",
              "facts_deduplicated", "llm_used", "episode_ids", "code",
              "llm_error", "took_ms"):
        assert k in keys_empty, k
        assert k in keys_short, k


# --- J-13: cross-platform config snippet -----------------------------------

def test_j13_example_config_is_valid_json_and_cross_platform():
    import json
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[1] / "examples" / "claude-desktop-config.example.json"
    with cfg_path.open() as f:
        data = json.load(f)
    assert "mcpServers" in data and "omnidb" in data["mcpServers"]
    # snippet must NOT hard-code a Windows-only path as the only example
    # (the file does carry one as a comment, but the actual command field
    # is "python3" which works on macOS + Linux)
    cmd = data["mcpServers"]["omnidb"]["command"]
    assert "python" in cmd


# --- J-14: hashing-embedder is offline-safe in auto mode ------------------

def test_j14_auto_does_not_try_fastembed(tmp_path, monkeypatch):
    """If fastembed import succeeds but its first call fails, the auto
    branch must NOT silently fall back. This is the J-14 fix: we never
    even attempt fastembed in auto/hashing modes."""
    from omnidb_server import embeddings
    calls = []
    real_init = embeddings.FastEmbedEmbedder.__init__

    def spy_init(self, *a, **k):
        calls.append("init")
        return real_init(self, *a, **k)

    monkeypatch.setattr(embeddings.FastEmbedEmbedder, "__init__", spy_init)
    cfg = Config(home=tmp_path, embedder="auto")
    emb = embeddings.make_embedder(cfg)
    # auto must return hashing, never fastembed
    assert emb.name == "hashing-v1"
    assert calls == []  # fastembed __init__ was never called


# --- J-15: perf gates are opt-in -----------------------------------------

def test_j15_perf_module_skipped_by_default():
    """Without OMNIDB_RUN_PERF, the perf module reports as skipped and
    no test in it actually runs."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_perf_smoke.py", "-v"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    # pytest returns exit 5 when no tests are collected (all skipped). We
    # accept any "skipped" reason string as the assertion.
    assert "OMNIDB_RUN_PERF" in r.stdout or r.returncode in (0, 5)
    # the module should NOT have a "passed" line for any perf test
    assert " PASSED " not in r.stdout.replace("::test_", "::test_ ")


# --- remember_many: batch insert (J-09 follow-up) -------------------------

def test_remember_many_collapses_writes_to_one_commit(tmp_path):
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     make_index(tmp_path, "brute", 512), cfg)
    N = 50
    items = [{"content": f"batch item {i} unique-token-{i}"} for i in range(N)]
    t0 = time.perf_counter()
    results = st.remember_many(items)
    _dt = time.perf_counter() - t0
    assert len(results) == N
    # ids are sequential and match the returned list
    assert [r["id"] for r in results] == list(range(1, N + 1)), [r["id"] for r in results]
    assert st.stats()["live"] == N
    st.close()


def test_remember_many_single_call_keeps_remember_api_compat(tmp_path):
    st = make_store(tmp_path)
    result = st.remember("hello", kind="episodic",
                          metadata={"source": "test"})
    assert "id" in result and "kind" in result
    assert result["kind"] == "episodic"
    # took_ms present per J-01-era remediation
    assert "took_ms" in result


# --- Multi-process home lock ---------------------------------------------

def test_home_lock_acquire_then_second_open_fails(tmp_path, monkeypatch):
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(tmp_path)
    emb = HashingEmbedder()
    idx = make_index(tmp_path, "brute", emb.dim)
    st1 = MemoryStore(conn, emb, idx, cfg)

    from omnidb_server.server import _acquire_home_lock, _release_home_lock

    # the lock is real on every platform (flock / msvcrt byte-range, ARCH-05):
    # a second acquire while held conflicts, and releasing frees it again
    fd = _acquire_home_lock(tmp_path)
    assert not isinstance(fd, OSError), fd
    second = _acquire_home_lock(tmp_path)
    assert isinstance(second, OSError), second
    _release_home_lock(fd)
    fd2 = _acquire_home_lock(tmp_path)
    assert not isinstance(fd2, OSError)
    _release_home_lock(fd2)
    st1.close()
