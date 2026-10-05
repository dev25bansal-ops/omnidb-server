"""Scale axis for the tenancy claim (paper Section VI).

The first draft measured cost at one tenant and isolation at 25, which is not a
scale result — it is two point measurements. What a reader actually needs to
know before adopting per-tenant partitioning is:

  1. what does it cost as the *deployment* grows (more tenants)?
  2. what does it cost as the *corpus* grows (more rows per tenant)?
  3. does the write path stay linear, or does per-tenant bookkeeping make it
     superlinear — the failure mode a reviewer will look for first?

(3) is the one that could invalidate the design, so it is measured explicitly
rather than assumed: a batch that interleaves T tenants must not do work
proportional to rows x tenants.

Latency is reported as percentiles, not a mean, because a per-tenant lock or
lookup shows up as a tail, not an average.

Usage:  python -m bench.tenancy_scale
        python -m bench.tenancy_scale --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnidb_server import db as dbmod  # noqa: E402
from omnidb_server.config import Config  # noqa: E402
from omnidb_server.embeddings import HashingEmbedder  # noqa: E402
from omnidb_server.store import MemoryStore  # noqa: E402
from omnidb_server.vector_index import open_vector_index  # noqa: E402


def _store(home: Path) -> MemoryStore:
    cfg = Config(home=home, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(home)
    embedder = HashingEmbedder()
    count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    index = open_vector_index(home, "brute", embedder.dim, count)
    return MemoryStore(conn, embedder, index, cfg)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


def measure_single_tenant_writes(n_writes: int = 400) -> dict:
    """Latency distribution for one tenant, the common deployment case."""
    with tempfile.TemporaryDirectory() as d:
        st = _store(Path(d) / "home")
        try:
            st.remember("warmup", tenant="default")
            st.flush()
            lat: list[float] = []
            for i in range(n_writes):
                t0 = time.perf_counter()
                st.remember(f"single tenant write {i}", tenant="default")
                lat.append((time.perf_counter() - t0) * 1000)
            st.flush()
            return {
                "tenants": 1,
                "writes": n_writes,
                "mean_ms": statistics.fmean(lat),
                "p50_ms": _pct(lat, 0.50),
                "p95_ms": _pct(lat, 0.95),
                "p99_ms": _pct(lat, 0.99),
            }
        finally:
            st.close()


def measure_batch_scaling(tenants: int, batch: int = 200,
                          reps: int = 5, trials: int = 5) -> dict:
    """Cost of one interleaved batch as the tenant count grows.

    A batch deliberately spans every tenant, which is the worst case: the write
    path must group by tenant and issue one statement per distinct tenant, so the
    cost should grow with `tenants` and be *flat* in `batch` once grouped. If the
    per-write cost rises faster than the tenant count does, grouping is not
    happening and the design does not scale.
    """
    with tempfile.TemporaryDirectory() as d:
        st = _store(Path(d) / "home")
        try:
            for t in range(tenants):
                st.remember(f"seed {t}", tenant=f"t{t:03d}")
            st.flush()

            # Repeated identically: the machine's own run-to-run spread on
            # this host is 9-23% of the mean, so a single trial cannot support
            # a claim finer than that. Trials are reported with their spread so
            # the paper can state which differences are real.
            trial_means: list[float] = []
            lat: list[float] = []
            for t in range(trials):
                trial: list[float] = []
                for r in range(reps):
                    items = [
                        {"content": f"batch {t}-{r} item {i} for tenant {i % tenants}",
                         "tenant": f"t{i % tenants:03d}"}
                        for i in range(batch)
                    ]
                    t0 = time.perf_counter()
                    st.remember_many(items)
                    trial.append((time.perf_counter() - t0) * 1000)
                trial_means.append(statistics.fmean(trial))
                lat.extend(trial)
            st.flush()
            mean = statistics.fmean(trial_means)
            return {
                "tenants": tenants,
                "batch": batch,
                "reps": reps,
                "trials": trials,
                "mean_ms": mean,
                "trial_means_ms": [round(v, 2) for v in trial_means],
                # relative spread across identical trials: the resolution limit
                "noise_pct": (100.0 * (max(trial_means) - min(trial_means)) / mean
                               if mean else 0.0),
                "p50_ms": _pct(lat, 0.50),
                "p95_ms": _pct(lat, 0.95),
                "us_per_write": mean * 1000 / batch,
            }
        finally:
            st.close()


def measure_corpus_scaling(rows_per_tenant: int,
                           tenants: int = 4) -> dict:
    """Write cost as each tenant's own corpus grows."""
    with tempfile.TemporaryDirectory() as d:
        st = _store(Path(d) / "home")
        try:
            for t in range(tenants):
                for i in range(10):
                    st.remember(f"warm {t}-{i}", tenant=f"t{t}")
            st.flush()
            t0 = time.perf_counter()
            for t in range(tenants):
                for i in range(rows_per_tenant):
                    st.remember(f"row {t}-{i} of the tenant corpus",
                                tenant=f"t{t}")
            st.flush()
            dt = time.perf_counter() - t0
            return {
                "tenants": tenants,
                "rows_per_tenant": rows_per_tenant,
                "total_rows": tenants * rows_per_tenant,
                "ms_per_write": dt * 1000 / (tenants * rows_per_tenant),
            }
        finally:
            st.close()


def measure_isolation_at_scale(tenants: int) -> dict:
    """Re-verify at rest as the tenant count grows, not just at 25."""
    with tempfile.TemporaryDirectory() as d:
        home = Path(d) / "home"
        st = _store(home)
        try:
            for t in range(tenants):
                st.remember(f"tenant {t} content about topic{t}",
                            tenant=f"t{t:03d}")
            st.flush()
            leaks = 0
            for t in range(tenants):
                table = f"fts_t{t:03d}"
                for row in st.conn.execute(
                        f'SELECT rowid FROM "{table}" WHERE "{table}" MATCH ?',
                        (f'"topic{t}"',)):
                    owner = st.conn.execute(
                        "SELECT tenant_id FROM memories WHERE id = ?",
                        (int(row["rowid"]),)).fetchone()[0]
                    if owner != f"t{t:03d}":
                        leaks += 1
            return {"tenants": tenants, "cross_tenant_postings": leaks}
        finally:
            st.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    out: dict = {"host_cpus": os.cpu_count()}
    print(f"Tenancy scale measurements  (host {os.cpu_count()} cpus)\n")

    print("1. single-tenant write latency distribution (n=400)")
    single = measure_single_tenant_writes()
    out["single_tenant"] = single
    print(f"   mean {single['mean_ms']:.3f} ms | p50 {single['p50_ms']:.3f} | "
          f"p95 {single['p95_ms']:.3f} | p99 {single['p99_ms']:.3f}\n")

    print("2. interleaved batch cost vs tenant count (batch=200, reps=5)")
    print(f"   {'tenants':>8} {'mean ms':>10} {'p95 ms':>9} {'us/write':>10} "
          f"{'noise':>10}")
    print("   (noise = spread across identical trials: the resolution limit)")
    batches = []
    for tenants in (1, 2, 4, 8, 16, 32, 64):
        row = measure_batch_scaling(tenants)
        batches.append(row)
        print(f"   {row['tenants']:>8} {row['mean_ms']:>10.3f} "
              f"{row['p95_ms']:>9.3f} {row['us_per_write']:>10.1f} "
              f"{row['noise_pct']:>9.1f}%")
    out["batch_scaling"] = batches
    base = batches[0]["mean_ms"]
    worst = batches[-1]
    growth = worst["mean_ms"] / base
    print(f"   growth {base:.2f} -> {worst['mean_ms']:.2f} ms "
          f"({growth:.1f}x for {worst['tenants']}x the tenants)")
    print("   linear-or-better means the per-tenant grouping is holding\n")

    print("3. write cost as each tenant's corpus grows (4 tenants)")
    corpora = []
    for rows in (50, 200, 800, 3200):
        row = measure_corpus_scaling(rows)
        corpora.append(row)
        print(f"   {row['total_rows']:>6} total rows: "
              f"{row['ms_per_write']:.3f} ms/write")
    out["corpus_scaling"] = corpora

    print("\n4. isolation at rest as tenant count grows")
    iso = []
    for tenants in (1, 10, 50, 100):
        row = measure_isolation_at_scale(tenants)
        iso.append(row)
        print(f"   {row['tenants']:>4} tenants: "
              f"{row['cross_tenant_postings']} cross-tenant postings")
    out["isolation_at_scale"] = iso

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())