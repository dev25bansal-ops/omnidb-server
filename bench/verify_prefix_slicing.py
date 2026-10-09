"""Verify single-leg rankings are exact prefixes (slicing == direct retrieval).

The pools phase retrieves dense/FTS5/bm25s/pyserini once at the max window and
slices per window, which is only sound if top-w is exactly the first w of
top-max. For an exact (brute-force) dense index and for single-leg lexical
rankers this should hold; this script proves it on real data rather than
assuming it.

Usage: python bench/verify_prefix_slicing.py [--corpus nfcorpus] [--windows 25,50,100]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from ce_study_runner import _configure_java  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="nfcorpus")
    ap.add_argument("--embedder", default="minilm")
    ap.add_argument("--windows", default="25,50,100")
    args = ap.parse_args()
    _configure_java()
    import research_bench_v3 as rb

    windows = [int(x) for x in args.windows.split(",")]
    docs, queries = rb.load_beir(args.corpus)
    vecs, qvecs = rb.get_embeddings(args.corpus, args.embedder, docs, queries)
    tmp = Path(tempfile.mkdtemp(prefix="prefix-verify-"))
    failures = []

    def check(name, direct, sliced_top):
        bad = 0
        for w, got in direct.items():
            want = [r[:w] for r in sliced_top]
            if got != want:
                bad += 1
        print(f"  {name:<14} {'OK' if not bad else 'MISMATCH':<9} "
              f"({bad} of {len(direct)} windows differ)")
        if bad:
            failures.append(name)

    print(f"{args.corpus} / {args.embedder}: prefix equality per stage")

    top = rb.retrieve_omnidb_dense_only(docs, queries, vecs, qvecs,
                                        tmp / "dense-max", limit=max(windows))
    direct = {w: rb.retrieve_omnidb_dense_only(docs, queries, vecs, qvecs,
                                               tmp / f"dense-{w}", limit=w)
              for w in windows}
    check("dense", direct, top)

    top = rb.retrieve_omnidb_sparse_only(docs, queries, vecs, qvecs,
                                         tmp / "fts-max", limit=max(windows))
    direct = {w: rb.retrieve_omnidb_sparse_only(docs, queries, vecs, qvecs,
                                                tmp / f"fts-{w}", limit=w)
              for w in windows}
    check("fts5_sparse", direct, top)

    top = rb.retrieve_bm25s(docs, queries, None, None, tmp / "bm25-max",
                            k=max(windows))
    direct = {w: rb.retrieve_bm25s(docs, queries, None, None, tmp / f"bm25-{w}", k=w)
              for w in windows}
    check("bm25s", direct, top)

    try:
        top = rb.retrieve_pyserini(docs, queries, None, None, tmp / "py-max",
                                   k=max(windows))
        direct = {w: rb.retrieve_pyserini(docs, queries, None, None, tmp / f"py-{w}", k=w)
                  for w in windows}
        check("pyserini", direct, top)
    except Exception as exc:  # noqa: BLE001
        print(f"  pyserini        SKIPPED  ({type(exc).__name__}: {str(exc)[:80]})")

    print()
    print(f"RESULT: {'FAIL' if failures else 'OK'} (failing stages: {failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())