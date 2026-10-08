"""Paired-difference bootstrap CIs for the 16 main-grid CE-vs-hybrid comparisons.

Table IV reports a 95% CI on the paired difference for every window-probe cell.
Table II reports Delta_CE and the family-corrected p, but no interval. This
script fills that gap so the main result carries the same evidence shape as the
probe: a paired bootstrap of the per-query differences

    Delta_i = nDCG_i(CE) - nDCG_i(hybrid)

resampled over that corpus's queries, 10,000 resamples, seed 1234 (the same
convention the probe uses).

Input:  bench/research-v3-results.ms-marco-minilm-l-6-v2.json  (creates the CE
        per-query arrays the +CE column comes from)
Output: bench/ce-grid-ci.json

Usage:
    python -m bench.ce_grid_ci --json bench/ce-grid-ci.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stats import BOOTSTRAP_N, RNG_SEED  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE / "research-v3-results.ms-marco-minilm-l-6-v2.json"
CORPORA = ["nfcorpus", "scifact", "scidocs", "fiqa"]
EMBEDDERS = ["minilm", "bge-small", "e5-small", "bge-base"]


def paired_ci(deltas: np.ndarray, n: int = BOOTSTRAP_N,
              seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Mean and 95% percentile CI of the paired difference, by bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    boot = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    data = json.loads(SRC.read_text(encoding="utf-8"))
    out = {"metadata": {
        "method": "paired-difference bootstrap over queries "
                  f"(Delta_i = nDCG_CE - nDCG_hybrid), {BOOTSTRAP_N} resamples, "
                  f"seed {RNG_SEED}",
        "source": SRC.name,
        "note": "CIs are on the paired difference, not on the marginal means.",
    }, "cells": {}}

    print(f"{'cell':<20} {'delta':>9} {'95% CI':>22} {'width':>8} {'excl 0':>7}")
    for corpus in CORPORA:
        for emb in EMBEDDERS:
            c = data[corpus][emb]
            base = np.asarray(c["omnidb"]["per_query"], dtype=float)
            ce = np.asarray(c["omnidb-ce"]["per_query"], dtype=float)
            assert len(base) == len(ce), f"{corpus}/{emb} length mismatch"
            m, lo, hi = paired_ci(ce - base)
            excl = lo > 0 or hi < 0
            out["cells"].setdefault(corpus, {})[emb] = {
                "n_queries": int(len(base)),
                "delta": round(m, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                "ci_excludes_zero": bool(excl),
            }
            print(f"{corpus + '/' + emb:<20} {m:>+9.4f} [{lo:>+8.4f},{hi:>+8.4f}]"
                  f" {hi - lo:>8.4f} {str(excl):>7}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())