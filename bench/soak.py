"""AUDIT-I12: bounded-live stability workload, revision 2 (not a scale test).

Seed 110 live rows outside timing. Each iteration forgets the 11 oldest rows,
remembers 10 entries plus one probe, then recalls twice. Live rows return to
110 each iteration and never exceed 110; the soft-deleted SQL tail still grows.
One completed API call is one operation: 14 per iteration (forget_many counts
once). `cycles` is a compatibility alias for completed operations, NOT recalls;
`iterations` and `recall_samples` are reported separately. Setup, instrumentation,
and final flush are excluded from operation latency; wall throughput includes
resource sampling. Encoding and automatic flush times overlap operation times.
The separate flush total includes automatic flushes, explicit final flush, and
close-time flush, but excludes seeding/setup flushes. Growth experiment evidence
is pending; this harness does not establish full I12 performance acceptance.

The duration/iteration cap stops only between complete iterations. Samples of
WAL follow every API call; RSS follows every iteration (sampled peaks, not OS
high-water guarantees). Results bind the workload source hash and label short
validation separately. No short run establishes the historical 60-second gate.
Writes soak-results-v2.txt, leaving soak-results.txt historical evidence intact.
The unchanged diagnostic gates are last/first-third recall P95 <= 4, WAL < 80 MB,
and sampled RSS growth < 300 MB when measurable. A full stability run is at
least 60 seconds; shorter runs report only a validation gate result.

AUDIT-I12: `--workload growth` is a separate scale experiment — the live set
grows monotonically (no forgets) to `--growth-cap`, and every recall is tagged
with the live count so latency-vs-size is read directly from the report. The
growth workload has no pass/fail gate: it is scaling evidence, not a stability
claim, and writes growth-results.txt by default.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import statistics
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

_PROBE = "soak stability probe about quarterly latency dashboards"

_VOCAB = (
    "anchor breeze cobalt drift ember flint granite harbor ivory jasper "
    "kernel lagoon meadow nimbus opal prairie quartz ridge summit tundra".split()
)

_WAL_LIMIT_BYTES = 80 * 1024 * 1024
_RSS_GROWTH_LIMIT_BYTES = 300 * 1024 * 1024
_P95_RATIO_LIMIT = 4.0


def _cycle_content(i: int) -> str:
    words = [_VOCAB[(i * 11 + step * 5) % len(_VOCAB)] for step in range(10)]
    return f"soak entry {i} covering " + " ".join(words)


def _p95(values_ms: list[float]) -> float:
    ordered = sorted(values_ms)
    rank = min(int(0.95 * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[rank]


def _source_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *(root / "omnidb_server").rglob("*.py")]
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda path: path.relative_to(root).as_posix()):
        name = source.relative_to(root).as_posix().encode("utf-8")
        content = source.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _make_rss_reader():
    try:
        import psutil

        proc = psutil.Process()

        def read() -> int:
            return int(proc.memory_info().rss)

        return read
    except ImportError:
        pass

    if sys.platform == "win32":
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        *[(f"Field{i}", ctypes.c_size_t) for i in range(8)]]

        def read() -> int:
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            handle = kernel32.GetCurrentProcess()
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return 0
            return int(pmc.Field1)

        return read

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OmniDB soak: bounded-live stability or growth scaling")
    parser.add_argument("--minutes", type=float, default=2,
                        help="duration cap in minutes (0 < minutes <= 60)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="optional completed-iteration cap; 14 API operations each")
    parser.add_argument("--backend", choices=("brute", "hnsw"), default="brute")
    parser.add_argument("--workload", choices=("stability", "growth"), default="stability",
                        help="stability: bounded live set with the diagnostic gates; "
                             "growth: monotonic live-set scaling evidence, no gate")
    parser.add_argument("--growth-cap", type=int, default=4000,
                        help="growth workload: stop once live rows reach this count")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0 < args.minutes <= 60:
        parser.error("--minutes must be greater than zero and at most 60")
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.workload == "growth" and args.growth_cap <= 110:
        parser.error("--growth-cap must exceed the 110-row seed")
    if args.output is None:
        args.output = Path(__file__).with_name(
            "growth-results.txt" if args.workload == "growth" else "soak-results-v2.txt")
    if args.output.resolve() == Path(__file__).with_name("soak-results.txt").resolve():
        parser.error("the historical soak-results.txt must not be overwritten")

    source_identity = _source_identity()
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    from omnidb_server.vector_index import BruteForceIndex, HnswlibIndex

    flush_ms: list[float] = []
    embedding_ms: list[float] = []

    class TimedEmbedder(HashingEmbedder):
        def encode(self, texts):
            t0 = time.perf_counter()
            try:
                return super().encode(texts)
            finally:
                embedding_ms.append((time.perf_counter() - t0) * 1000)

    class TimedStore(MemoryStore):
        def flush(self):
            t0 = time.perf_counter()
            try:
                return super().flush()
            finally:
                flush_ms.append((time.perf_counter() - t0) * 1000)

    live_target = 110
    iterations = 0
    timings: dict[str, list[float]] = {"remember": [], "recall": [], "forget_many": []}
    content_sizes: list[int] = []
    rss_read = _make_rss_reader()
    with tempfile.TemporaryDirectory(prefix="omnidb-soak-") as directory:
        home = Path(directory)
        cfg = Config(home=home, embedder="hashing", vector_index=args.backend)
        emb = TimedEmbedder()
        index = (BruteForceIndex(home / "vectors.npz", emb.dim) if args.backend == "brute"
                 else HnswlibIndex(home / "index.hnsw", emb.dim))
        store = TimedStore(dbmod.connect(home), emb, index, cfg)
        try:
            ring = deque(store.remember(_cycle_content(i))["id"] for i in range(live_target))
            store.flush()
            flush_ms.clear()
            embedding_ms.clear()
            rss_first = rss_read() if rss_read else 0
            rss_peak = rss_first
            wal_max = 0
            live_min = live_max = live_target
            boundary_min = boundary_max = live_target
            wal_path = home / "omnidb.db-wal"

            def call(name, *positional, **keywords):
                nonlocal wal_max
                t0 = time.perf_counter()
                result = getattr(store, name)(*positional, **keywords)
                timings[name].append((time.perf_counter() - t0) * 1000)
                wal_max = max(wal_max, wal_path.stat().st_size if wal_path.exists() else 0)
                return result

            if args.workload == "growth":
                # AUDIT-I12: separate scale experiment. The live set grows
                # monotonically (no forgets) to --growth-cap; every recall is
                # tagged with the live count so latency-vs-size is read
                # directly. No pass/fail gate — scaling evidence only.
                recall_points: list[tuple[int, float]] = []
                started = time.perf_counter()
                live = live_target
                while live < args.growth_cap:
                    if args.iterations is not None and iterations >= args.iterations:
                        break
                    if time.perf_counter() - started >= args.minutes * 60:
                        break
                    for j in range(10):
                        content = _cycle_content(live + j)
                        call("remember", content)
                        content_sizes.append(len(content.encode("utf-8")))
                    live += 10
                    for _ in range(2):
                        call("recall", _PROBE, k=5)
                        recall_points.append((live, timings["recall"][-1]))
                    rss_peak = max(rss_peak, rss_read() if rss_read else 0)
                    iterations += 1
                elapsed_s = time.perf_counter() - started
                final_state = store.stats()
                store.flush()
                rss_peak = max(rss_peak, rss_read() if rss_read else 0)
                ops = sum(len(v) for v in timings.values())
                boundaries = [110, 500, 1000, 2000, 4000, 8000]
                lines = [
                    "workload_revision: growth-v1",
                    f"workload_sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
                    f"source_identity_sha256: {source_identity}",
                    f"python: {sys.version.split()[0]}",
                    f"platform: {sys.platform}",
                    "gate: none (scaling evidence, not a stability claim)",
                    f"backend: {args.backend}",
                    f"dimensions: {emb.dim}",
                    f"embedder: {emb.name}",
                    f"growth_cap: {args.growth_cap}",
                    f"duration_s: {elapsed_s:.3f}",
                    f"iterations: {iterations}",
                    f"ops: {ops}",
                    f"ops_per_second: {ops / elapsed_s:.2f}",
                    f"live_final: {final_state['live']}",
                    f"deleted_final: {final_state['soft_deleted']}",
                    f"flush_count: {len(flush_ms)}",
                    f"flush_total_ms: {sum(flush_ms):.3f}",
                    f"embedding_calls: {len(embedding_ms)}",
                    f"embedding_total_ms: {sum(embedding_ms):.3f}",
                    f"rss_start_mb: {rss_first / 1024 ** 2:.3f}" if rss_first else "rss_start_mb: n/a",
                    f"rss_peak_mb: {rss_peak / 1024 ** 2:.3f}" if rss_peak else "rss_peak_mb: n/a",
                    f"wal_peak_bytes: {wal_max}",
                    f"wal_max_mb: {wal_max / 1024 ** 2:.3f}",
                ]
                for lo, hi in zip(boundaries, boundaries[1:]):
                    samples = [ms for n, ms in recall_points if lo <= n < hi]
                    if samples:
                        lines.append(f"recall_p50_ms_{lo}_to_{hi}: {statistics.median(samples):.3f}")
                        lines.append(f"recall_p95_ms_{lo}_to_{hi}: {_p95(samples):.3f}")
                        lines.append(f"recall_count_{lo}_to_{hi}: {len(samples)}")
                args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
                print(f"wrote {args.output}")
                print("\n".join(lines))
                return 0

            started = time.perf_counter()
            while (iterations == 0 or time.perf_counter() - started < args.minutes * 60):
                if args.iterations is not None and iterations >= args.iterations:
                    break
                oldest = [ring.popleft() for _ in range(11)]
                forgotten = call("forget_many", oldest)["forgotten"]
                if set(forgotten) != set(oldest):
                    raise RuntimeError("bounded-live workload did not forget its oldest rows")
                live_min = min(live_min, len(ring))
                contents = [_PROBE] + [_cycle_content(live_target + iterations * 10 + j) for j in range(10)]
                for content in contents:
                    ring.append(call("remember", content)["id"])
                    content_sizes.append(len(content.encode("utf-8")))
                live_max = max(live_max, len(ring))
                for _ in range(2):
                    call("recall", _PROBE, k=5)
                state = store.stats()
                boundary_min = min(boundary_min, state["live"])
                boundary_max = max(boundary_max, state["live"])
                if state["live"] != live_target:
                    raise RuntimeError(f"live count drifted: {state['live']} != {live_target}")
                rss_peak = max(rss_peak, rss_read() if rss_read else 0)
                iterations += 1
            elapsed_s = time.perf_counter() - started
            final_state = store.stats()
            store.flush()
            rss_peak = max(rss_peak, rss_read() if rss_read else 0)
        finally:
            store.close()

    latencies = timings["recall"]
    third = max(1, len(latencies) // 3)
    p95_first, p95_last = _p95(latencies[:third]), _p95(latencies[-third:])
    ops = sum(len(values) for values in timings.values())
    stable = p95_last <= _P95_RATIO_LIMIT * p95_first and wal_max < _WAL_LIMIT_BYTES
    if rss_first:
        stable = stable and rss_peak - rss_first < _RSS_GROWTH_LIMIT_BYTES
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = [
        "workload_revision: bounded-live-v2",
        f"workload_sha256: {source_hash}",
        # AUDIT-I12: bind the measurement to the runtime under test (no git
        # in this tree, so the digest covers the harness plus every shipped
        # module) rather than claiming a revision that does not exist.
        f"source_identity_sha256: {source_identity}",
        f"python: {sys.version.split()[0]}",
        f"platform: {sys.platform}",
        f"run_kind: {'short-validation' if elapsed_s < 60 else 'stability'}",
        "historical_gate: not re-evaluated; v1 workload is not comparable",
        f"backend: {args.backend}",
        f"dimensions: {emb.dim}",
        f"embedder: {emb.name}",
        "operation_unit: completed API call (forget_many counts once)",
        f"duration_s: {elapsed_s:.3f}",
        f"iterations: {iterations}",
        f"cycles: {ops}",
        f"ops: {ops}",
        f"ops_per_second: {ops / elapsed_s:.2f}",
        f"recall_samples: {len(latencies)}",
        f"content_bytes_min: {min(content_sizes)}",
        f"content_bytes_max: {max(content_sizes)}",
        f"content_bytes_mean: {statistics.fmean(content_sizes):.2f}",
        f"live_target: {live_target}",
        f"live_min: {live_min}",
        f"live_max: {live_max}",
        f"iteration_end_live_min: {boundary_min}",
        f"iteration_end_live_max: {boundary_max}",
        f"live_final: {final_state['live']}",
        f"deleted_final: {final_state['soft_deleted']}",
        f"rows_total_final: {final_state['live'] + final_state['soft_deleted']}",
        f"flush_count: {len(flush_ms)}",
        f"flush_total_ms: {sum(flush_ms):.3f}",
        f"embedding_calls: {len(embedding_ms)}",
        f"embedding_total_ms: {sum(embedding_ms):.3f}",
        f"p95_first_third_ms: {p95_first:.3f}",
        f"p95_last_third_ms: {p95_last:.3f}",
        f"p95_ratio: {p95_last / p95_first:.3f}",
        f"rss_start_mb: {rss_first / 1024 ** 2:.3f}" if rss_first else "rss_start_mb: n/a",
        f"rss_peak_mb: {rss_peak / 1024 ** 2:.3f}" if rss_peak else "rss_peak_mb: n/a",
        f"wal_peak_bytes: {wal_max}",
        f"wal_max_mb: {wal_max / 1024 ** 2:.3f}",
        "resource_sampling: WAL after each API call; RSS after each iteration and final flush",
        f"validation_gate_passed: {stable}",
    ]
    for name, values in timings.items():
        lines.extend((f"{name}_count: {len(values)}",
                      f"{name}_p50_ms: {statistics.median(values):.3f}",
                      f"{name}_p95_ms: {_p95(values):.3f}"))
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print("\n".join(lines))
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
