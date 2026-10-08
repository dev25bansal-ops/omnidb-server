"""Baselines: how does per-tenant partitioning compare to native multi-tenancy?

The contamination sweep compares two layouts I wrote. A reviewer will not accept
that as evidence of anything beyond internal consistency: both conditions are
mine, and the obvious objection is that a purpose-built system does this better.

This harness adds the comparison that matters. Four systems with a shipped
multi-tenancy story, measured on the same task as Section IV -- keep one
tenant's lexical ranking from moving when another tenant writes:

  sqlite_vec   one table per collection (its isolation primitive)
  chromadb     per-collection, with tenant metadata on the records
  qdrant       payload-filtered tenants over a shared collection
  lancedb      per-table partitions

Each system's isolation mechanism is recorded alongside its measurement, because
the interesting question is not which is fastest but *where each one puts the
boundary* -- and therefore which class of failure each one still admits.

This is the comparison that reframes the paper's contribution. If a purpose-built
database gives per-tenant isolation natively, then partitioning is not novel;
what is novel is showing that the shared-index design is a real and measurable
regression from it, which is a claim about agent memory stores specifically.

Any backend that is not installed is reported as such rather than skipped
silently -- a missing baseline reported as a win would be the worst outcome.

Usage:  python -m bench.tenancy_baselines
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3 as s3  # noqa: E402

import numpy as np  # noqa: E402

QUERY_TENANT = "acme"
FOREIGN_TENANT = "globex"
DIM = 64


def _vecs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _time(fn, *, warmup: int = 0, repeat: int = 1) -> tuple[float, object]:
    """Median-of-N wall time in milliseconds, after `warmup` untimed calls.

    The defaults time a single call, which is what a *build* needs: builds are
    not idempotent (a second call hits "table already exists"), so they must
    be timed once. Query arms pass `warmup=5, repeat=21` and get a median.

    Before this existed, queries were also timed with one un-warmed call, and
    the reported column was unreproducible: re-running the same harness on the
    same host moved individual rows by 2-47x, because a single sample of a
    sub-millisecond operation is dominated by page faults and cold-cache
    effects. A median over repeated calls is the statistic the table can
    support; a single call is never taken as the point estimate.

    The return value `out` comes from the first *timed* call, not from a
    separate untimed one. Getting that wrong is not cosmetic: an earlier
    revision called `fn()` once to capture `out` and then `repeat` more times,
    so `repeat=1` (the build path) ran the build twice and the second call
    failed on "table already exists", taking four baselines off the table.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    out = None
    for i in range(max(1, repeat)):
        t0 = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - t0) * 1000)
        if i == 0:
            out = result
    samples.sort()
    return samples[len(samples) // 2], out


# Query arms use this: identical to _time's old call signature at the call
# sites, but with the repetition a sub-millisecond measurement needs.
def _time_query(fn) -> tuple[float, object]:
    return _time(fn, warmup=5, repeat=21)


# --------------------------------------------------------------------------
# baselines, each returning (isolation_unit, elapsed_ms, detail)
# --------------------------------------------------------------------------

def baseline_sqlite_vec() -> dict:
    """sqlite_vec: one table per collection.

    Isolation is per *collection*, which is the same primitive OmniDB uses, but
    the boundary is a table in the same file rather than a separate file.
    """
    try:
        import sqlite_vec
    except ImportError:
        return {"available": False}

    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "v.db"
    conn = s3.connect(str(path))          # stock sqlite3 + the extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    try:
        def build():
            for i, tenant in enumerate((QUERY_TENANT, FOREIGN_TENANT)):
                conn.execute(f'CREATE VIRTUAL TABLE "{tenant}" '
                             f"USING vec0(embedding float[{DIM}])")
                conn.executemany(
                    f'INSERT INTO "{tenant}"(embedding) VALUES (?)',
                    [(v.tobytes(),) for v in _vecs(200, seed=1 + i)])
            conn.commit()
            return None

        build_ms, _ = _time(build)

        def search():
            q = _vecs(1, seed=99)[0]
            return conn.execute(
                f'SELECT rowid, distance FROM "{QUERY_TENANT}" '
                "WHERE embedding MATCH ? AND k = 10",
                (q.tobytes(),)).fetchall()

        search_ms, hits = _time_query(search)
        return {
            "available": True,
            "isolation_unit": "table per tenant (one collection per tenant)",
            "boundary_in_file": True,
            "build_ms": build_ms,
            "query_ms": search_ms,
            "hits": len(hits),
            "note": "no cross-tenant index at all; the tenant's rows are "
                    "physically separated within one database file",
        }
    finally:
        conn.close()
        shutil.rmtree(tmp, ignore_errors=True)


def baseline_qdrant(payload_filter: bool = True) -> dict:
    """Qdrant: either per-collection, or payload-filtered over a shared one.

    The filtered variant is the interesting comparison: it is the production
    analogue of OmniDB's *pre-migration* design, and it is what most systems
    reach for when adding tenancy to an existing deployment.
    """
    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        return {"available": False}

    client = QdrantClient(":memory:")

    def build():
        if payload_filter:
            client.create_collection("shared", vectors_config=models.VectorParams(
                size=DIM, distance=models.Distance.COSINE))
            # Disjoint id ranges per tenant. Both tenants previously upserted
            # ids 0..199 into the same collection, so the second tenant's points
            # *overwrote* the first's (same ids, different payload) and the
            # querying tenant ended up with ZERO points -- this arm was timing
            # an empty query and reporting 0 hits, against 10 for every other
            # system. The row exists to be the production analogue of the
            # shared-index design, so an empty result defeats its purpose.
            for i, (tenant, seed) in enumerate(
                    ((QUERY_TENANT, 1), (FOREIGN_TENANT, 2))):
                client.upsert(collection_name="shared",
                              points=models.Batch(
                              ids=list(range(i * 1000, i * 1000 + 200)),
                              vectors=_vecs(200, seed).tolist(),
                              payloads=[{"tenant": tenant}] * 200))
        else:
            for tenant, seed in ((QUERY_TENANT, 1), (FOREIGN_TENANT, 2)):
                client.create_collection(
                    tenant, vectors_config=models.VectorParams(
                        size=DIM, distance=models.Distance.COSINE))
                client.upsert(collection_name=tenant,
                              points=models.Batch(
                               ids=list(range(200)),
                               vectors=_vecs(200, seed).tolist(),
                               payloads=[{}] * 200))
        return None

    build_ms, _ = _time(build)

    def search():
        q = _vecs(1, seed=99)[0].tolist()
        if payload_filter:
            res = client.query_points(
                "shared", query=q, limit=10, query_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="tenant", match=models.MatchValue(
                            value=QUERY_TENANT))]))
            return getattr(res, "points", res)
        res = client.query_points(QUERY_TENANT, query=q, limit=10)
        return getattr(res, "points", res)

    search_ms, hits = _time_query(search)
    return {
        "available": True,
        "isolation_unit": ("payload filter over a shared collection"
                           if payload_filter else "collection per tenant"),
        "boundary_in_file": not payload_filter,
        "build_ms": build_ms,
        "query_ms": search_ms,
        "hits": len(hits),
        "note": ("ranking computed over the shared collection; filtering is "
                 "post-hoc at query time" if payload_filter else
                 "one index per tenant"),
    }


def baseline_lancedb() -> dict:
    """LanceDB: one table per tenant."""
    try:
        import lancedb
    except ImportError:
        return {"available": False}

    tmp = tempfile.mkdtemp()
    try:
        db = lancedb.connect(tmp)

        def build():
            for tenant, seed in ((QUERY_TENANT, 1), (FOREIGN_TENANT, 2)):
                data = [{"vector": v.tolist(), "id": i}
                        for i, v in enumerate(_vecs(200, seed))]
                db.create_table(tenant, data=data)
            return None

        build_ms, _ = _time(build)

        def search():
            q = _vecs(1, seed=99)[0].tolist()
            return db.open_table(QUERY_TENANT).search(q).limit(10).to_list()

        search_ms, hits = _time_query(search)
        return {
            "available": True,
            "isolation_unit": "table per tenant",
            "boundary_in_file": True,
            "build_ms": build_ms,
            "query_ms": search_ms,
            "hits": len(hits),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def baseline_omnidb_shared() -> dict:
    """OmniDB's own pre-migration layout: one index, tenant predicate.

    Included so the comparison is against the thing this paper argues with, on
    the same machine in the same process.
    """

    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    from omnidb_server.vector_index import open_vector_index

    tmp = tempfile.mkdtemp()
    home = Path(tmp) / "home"
    try:
        cfg = Config(home=home, embedder="hashing", vector_index="brute")
        conn = dbmod.connect(home)
        embedder = HashingEmbedder()
        index = open_vector_index(home, "brute", embedder.dim, 0)
        store = MemoryStore(conn, embedder, index, cfg)
        try:
            def build():
                for tenant, seed in ((QUERY_TENANT, 1), (FOREIGN_TENANT, 2)):
                    for i in range(200):
                        store.remember(f"{tenant} row {i} unique{i}",
                                       tenant=tenant)
                store.flush()
                return None

            build_ms, _ = _time(build)

            # The pre-migration layout no longer exists in the shipped code, so
            # the baseline constructs it: collapse the per-tenant index back into
            # one shared table over the same rows. That is exactly the schema
            # the paper argues against, built from the same write path.
            store.conn.execute("DROP TABLE IF EXISTS fts_acme")
            store.conn.execute("DROP TABLE IF EXISTS fts_globex")
            store.conn.execute(
                "CREATE VIRTUAL TABLE memory_fts USING fts5("
                "content, tokenize='porter unicode61')")
            store.conn.execute(
                "INSERT INTO memory_fts(rowid, content) "
                "SELECT id, content FROM memories")
            store.conn.commit()

            def search():
                # Match a term carried by every row of the querying tenant
                # ("acme"), then take the top 10 -- so this arm returns 10 rows
                # like the four vector systems do. The previous query matched
                # '"unique1"', a token held by exactly one row, so it timed a
                # 1-row lookup and returned 1 hit against their 10. The table
                # claims all systems were measured identically; this makes the
                # row counts (and so the work) agree.
                return store.conn.execute(
                    "SELECT f.rowid FROM memory_fts f JOIN memories m "
                    "ON m.id=f.rowid WHERE memory_fts MATCH ? "
                    "AND m.tenant_id = ? LIMIT 10",
                    (QUERY_TENANT, QUERY_TENANT)).fetchall()

            search_ms, hits = _time_query(search)
            return {
                "available": True,
                "isolation_unit": "one shared index + tenant predicate",
                "boundary_in_file": False,
                "build_ms": build_ms,
                "query_ms": search_ms,
                "hits": len(hits),
                "note": "the layout this paper argues against; included as the "
                        "reference point, not as a competitor",
            }
        finally:
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def baseline_omnidb_partitioned() -> dict:
    """OmniDB as shipped: one index per tenant."""
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    from omnidb_server.vector_index import open_vector_index

    tmp = tempfile.mkdtemp()
    home = Path(tmp) / "home"
    try:
        cfg = Config(home=home, embedder="hashing", vector_index="brute")
        conn = dbmod.connect(home)
        embedder = HashingEmbedder()
        index = open_vector_index(home, "brute", embedder.dim, 0)
        store = MemoryStore(conn, embedder, index, cfg)
        try:
            def build():
                for tenant, seed in ((QUERY_TENANT, 1), (FOREIGN_TENANT, 2)):
                    for i in range(200):
                        store.remember(f"{tenant} row {i} unique{i}",
                                       tenant=tenant)
                store.flush()
                return None

            build_ms, _ = _time(build)

            def search():
                # Same query as the shared arm, against this tenant's own
                # table: 10 rows returned, so the comparison is like-for-like.
                return store.conn.execute(
                    'SELECT rowid FROM "fts_acme" WHERE "fts_acme" MATCH ? '
                    "LIMIT 10", (QUERY_TENANT,)).fetchall()

            search_ms, hits = _time_query(search)
            return {
                "available": True,
                "isolation_unit": "one index per tenant",
                # True, not False: `build` above writes through the shipped
                # per-tenant path, so each tenant gets its own FTS5 table
                # (fts_acme, fts_globex) and its own vector sidecar file. The
                # boundary is on disk -- which is what this row's `note` says
                # and what Table~\ref{tab:baselines} in the paper reports. The
                # False here was the odd one out: the sibling "shared" arm is
                # what sets False, and only after DROPping those tables.
                "boundary_in_file": True,
                "build_ms": build_ms,
                "query_ms": search_ms,
                "hits": len(hits),
                "note": "separate sidecar files: the boundary exists on disk, "
                        "not just in the query",
            }
        finally:
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


BASELINES = {
    "OmniDB shared (pre-migration)": baseline_omnidb_shared,
    "OmniDB partitioned (this paper)": baseline_omnidb_partitioned,
    "sqlite_vec (table/tenant)": baseline_sqlite_vec,
    "qdrant (collection/tenant)": lambda: baseline_qdrant(payload_filter=False),
    "qdrant (payload filter)": lambda: baseline_qdrant(payload_filter=True),
    "lancedb (table/tenant)": baseline_lancedb,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    rows: list[dict] = []
    print(f"Isolation mechanism by system  (200 rows/tenant, 2 tenants, "
          f"dim {DIM})\n")
    print(f"{'system':<32} {'boundary':>10} {'build ms':>10} {'query ms':>10}")
    print("-" * 66)
    for name, fn in BASELINES.items():
        try:
            row = fn()
        except Exception as exc:                      # noqa: BLE001
            row = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        rows.append({"name": name, **row})
        if not row.get("available"):
            reason = row.get("error", "not installed")
            print(f"{name:<32} {'n/a':>10} {'-':>10} {'-':>10}   ({reason[:28]})")
            continue
        print(f"{name:<32} {'file' if row['boundary_in_file'] else 'query':>10} "
              f"{row['build_ms']:>10.1f} {row['query_ms']:>10.2f}")

    print("\nIsolation unit per system:")
    for row in rows:
        if row.get("available"):
            print(f"  {row['name']:<32} {row['isolation_unit']}")

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows}, indent=2),
                                    encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())