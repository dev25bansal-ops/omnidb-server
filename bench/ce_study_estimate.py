"""Estimate the full CE-study matrix: cells, candidate pairs, and GPU time.

Reads measured throughput and the frozen query split; writes nothing.
Usage: python bench/ce_study_estimate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ce_study_runner import (  # noqa: E402
    CORPORA, EMBEDDERS, STAGES, WINDOWS, RERANKER_REVISIONS,
)

# Measured with bench/bench_reranker_throughput.py on the RTX 5060 Laptop GPU
# (batch 32, max_length 512, 1500 pairs).
# REALISTIC rates: distinct documents, batch 32, measured by
# bench/measure_realistic_rates.py. The earlier numbers in this file came from
# bench_reranker_throughput.py, which repeated one pair and was therefore 3-8x
# optimistic (the GPU caches an identical batch).
PAIRS_PER_SEC = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 731.6,
    "BAAI/bge-reranker-base": 93.4,
    "BAAI/bge-reranker-large": 26.8,
    "mixedbread-ai/mxbai-rerank-base-v2": 51.8,
    "ncbi/MedCPT-Cross-Encoder": 130.2,
}
RATES_SOURCE = "bench/measured-reranker-rates.json (distinct documents, batch 32)"
# Observed average pool size is below the window because some first stages
# return short lists (Pyserini especially). Factor from the smoke pools.
AVG_POOL_FRACTION = 0.85


def main() -> int:
    split = json.loads((HERE / "ce-study-query-split.json").read_text(encoding="utf-8"))
    corpora = list(CORPORA)
    stage_variants = []
    for stage in STAGES:
        if stage in ("hybrid", "dense"):
            for emb in EMBEDDERS:
                stage_variants.append((stage, emb))
        else:
            stage_variants.append((stage, "shared"))

    total_cells = len(stage_variants) * len(WINDOWS) * len(RERANKER_REVISIONS) * len(corpora)
    print(f"corpora: {corpora}")
    print(f"stage variants: {len(stage_variants)}  "
          f"(hybrid/dense x {len(EMBEDDERS)} encoders + {len(STAGES)-2} shared)")
    print(f"windows: {list(WINDOWS)}   rerankers: {len(RERANKER_REVISIONS)}")
    print(f"PRE-DECLARED FAMILY A SIZE: {total_cells} contrasts")

    # Candidate pairs: all queries are retrieved once per stage/window, but each
    # reranker scores the same pools. Pools phase cost is retrieval, not GPU.
    print()
    print("candidate pairs scored per reranker (all corpora, all stages, all windows):")
    grand_pairs = 0
    for corpus in corpora:
        n_q = split["corpora"][corpus]["n_total"]
        pairs = 0
        for _stage, _emb in stage_variants:
            for w in WINDOWS:
                pairs += int(n_q * w * AVG_POOL_FRACTION)
        grand_pairs += pairs
    print(f"  total pair-scores per reranker: {grand_pairs:,}")

    print()
    print("estimated GPU scoring time for the full matrix:")
    total_seconds = 0.0
    for model, rate in PAIRS_PER_SEC.items():
        seconds = grand_pairs / rate
        total_seconds += seconds
        print(f"  {model:<46} {seconds/3600:6.2f} h  ({rate:,.0f} pairs/s)")
    print(f"  {'TOTAL (sequential)':<46} {total_seconds/3600:6.2f} h")
    print(f"  rates from {RATES_SOURCE}")

    # Pools phase: retrieval over 4 corpora x stage variants x windows. Document
    # counts are the dominant term; measured earlier at roughly 3-6 min per
    # corpus-stage-window pair on this machine, larger corpora proportionally more.
    print()
    retrieval_cells = sum(len(WINDOWS) for _ in stage_variants) * len(corpora)
    print(f"retrieval cells (corpus x stage-variant x window): {retrieval_cells}")
    print("  Every stage is retrieved fresh per window (prefix slicing is unsound")
    print("  for the lexical stages; see bench/verify_prefix_slicing.py).")
    print("  measured: ~3-6 min per cell on small corpora, longer on scidocs/fiqa")
    print("  rough pools-phase estimate: 8-20 h wall clock (single GPU/CPU host)")
    return 0


if __name__ == "__main__":
    sys.exit(main())