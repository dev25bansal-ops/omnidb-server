"""Performance smoke gates with generous margins.

Offline hashing embedder over the brute-force index on a fresh temp home;
thresholds sit roughly 10x inside measured local headroom so only gross
regressions trip. Set OMNIDB_SKIP_PERF=1 to skip the whole module.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        os.environ.get("OMNIDB_SKIP_PERF") == "1" or os.environ.get("OMNIDB_RUN_PERF") != "1",
        reason="perf smoke disabled; set OMNIDB_RUN_PERF=1 to run",
    ),
]

_N_WRITES = 1200
_WRITE_FLOOR = 250.0
_N_QUERIES = 40
_P50_LIMIT_MS = 25.0
_P95_LIMIT_MS = 80.0
_SEARCH_LIMIT_MS = 100.0

_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu".split()
)


def _content(i: int) -> str:
    words = [_VOCAB[(i * 7 + step * 3) % len(_VOCAB)] for step in range(9)]
    return f"perf note {i} about " + " ".join(words)


def _probe(i: int) -> str:
    return " ".join(_content(i).split()[:6])


def _percentile(sorted_ms: list[float], q: float) -> float:
    rank = min(int(q * (len(sorted_ms) - 1)), len(sorted_ms) - 1)
    return sorted_ms[rank]


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    import conftest

    store = conftest.make_store(tmp_path_factory.mktemp("perf"))
    t0 = time.perf_counter()
    ids = [store.remember(_content(i))["id"] for i in range(_N_WRITES)]
    write_s = time.perf_counter() - t0
    yield store, ids, write_s
    store.close()


def test_write_throughput(bench):
    _store, ids, write_s = bench
    wps = len(ids) / write_s
    print(f"\nwrite throughput: {wps:.0f} writes/s ({len(ids)} writes in {write_s:.2f}s)")
    assert wps >= _WRITE_FLOOR


def test_recall_latency(bench):
    store, ids, _write_s = bench
    latencies = []
    for q in range(_N_QUERIES):
        probe = _probe((q * 29) % len(ids))
        t0 = time.perf_counter()
        hits = store.recall(probe, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        assert hits, f"probe {probe!r} returned no hits"
    latencies.sort()
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    print(f"\nrecall latency ms: p50={p50:.2f} p95={p95:.2f} over {_N_QUERIES} queries")
    assert p50 < _P50_LIMIT_MS
    assert p95 < _P95_LIMIT_MS


def test_search_count(bench):
    store, _ids, _write_s = bench
    store.search("SELECT COUNT(*) FROM memories")
    worst = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        store.search("SELECT COUNT(*) FROM memories")
        worst = max(worst, (time.perf_counter() - t0) * 1000)
    print(f"\nsearch COUNT(*) worst of 3: {worst:.2f} ms")
    assert worst < _SEARCH_LIMIT_MS
