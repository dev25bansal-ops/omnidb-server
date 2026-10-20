"""Stage the SciDocs window-probe per-query arrays into the repository.

The SciDocs sweep wrote its output to E:/omnidb-bench-tmp/sweep-rerank-scidocs.json,
outside the repository, so 4 of the 18 hybrid window-cells (scidocs x 2 arms x
2 windows) could not be independently re-derived from the committed artifacts.
The other three corpora are committed as bench/<corpus>-rerank-window-results.json
or bench/rerank-window-results.json (fiqa). This script copies the SciDocs file
into the same place and shape so that the full Fig. 3 ECDF and the window-probe
table are reproducible from the repository alone.

Usage:
    python bench/stage_scidocs_window_results.py [--source PATH] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"E:\omnidb-bench-tmp\sweep-rerank-scidocs.json")
DEST = HERE / "scidocs-rerank-window-results.json"

# The structure the other committed window-result files use.
ARMS = ("bm25", "minilm", "bge-base")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--check", action="store_true",
                    help="verify the committed copy against the source, then exit")
    args = ap.parse_args()

    source = Path(args.source)

    if args.check:
        if not DEST.exists():
            print(f"FAIL: {DEST} does not exist")
            return 1
        if not source.exists():
            print(f"SKIP: source {source} is not present on this machine; "
                  f"cannot compare")
            return 0
        a = json.loads(source.read_text(encoding="utf-8"))
        b = json.loads(DEST.read_text(encoding="utf-8"))
        if a == b:
            print(f"OK: committed {DEST.name} is identical to {source}")
            return 0
        print(f"FAIL: committed {DEST.name} differs from {source}")
        for arm in ARMS:
            for w in ("25", "100"):
                ac = a.get(arm, {}).get("cells", {}).get(w)
                bc = b.get(arm, {}).get("cells", {}).get(w)
                if ac != bc:
                    print(f"  differs: {arm} w={w}")
        return 1

    if not source.exists():
        print(f"FAIL: source {source} does not exist")
        return 1

    raw = json.loads(source.read_text(encoding="utf-8"))
    out = {"_ndocs": raw.get("_ndocs"), "_nq": raw.get("_nq")}
    for arm in ARMS:
        if arm not in raw:
            print(f"FAIL: source has no arm {arm}")
            return 1
        out[arm] = raw[arm]
    DEST.write_text(json.dumps(out), encoding="utf-8")
    size = DEST.stat().st_size
    print(f"wrote {DEST} ({size} bytes)")
    for arm in ARMS:
        for w, c in sorted(out[arm]["cells"].items()):
            print(f"  {arm:<10} w={w:<4} n={len(c['base_ndcg10'])} "
                  f"base={c['base_mean']:.4f} rerank={c['rerank_mean']:.4f} "
                  f"delta={c['delta']:+.4f} recall={c['stage_recall_at_w']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
