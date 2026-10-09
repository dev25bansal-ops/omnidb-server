"""Measure batched reranker throughput on GPU to size the full matrix.

Usage: python bench/bench_reranker_throughput.py [--models m1,m2] [--pairs 2000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ce_study_runner import RERANKER_REVISIONS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(RERANKER_REVISIONS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cache = (HERE / ".cache3" / "hf").resolve()
    for var in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ[var] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"

    from sentence_transformers import CrossEncoder

    doc = ("Hypertension is a long-term medical condition in which the blood "
           "pressure in the arteries is persistently elevated. It usually does "
           "not cause symptoms, and is diagnosed by measuring blood pressure.")
    pairs = [(f"query number {i} about blood pressure treatment", doc)
             for i in range(args.pairs)]

    print(f"device={args.device} pairs={args.pairs} batch={args.batch_size}")
    for model_id in args.models.split(","):
        rev = RERANKER_REVISIONS[model_id]
        try:
            model = CrossEncoder(model_id, revision=rev, max_length=512,
                                 device=args.device, local_files_only=True)
            model.predict(pairs[:64], batch_size=args.batch_size,
                          show_progress_bar=False)  # warmup
            t0 = time.perf_counter()
            model.predict(pairs, batch_size=args.batch_size,
                          show_progress_bar=False)
            dt = time.perf_counter() - t0
            rate = len(pairs) / dt
            print(f"  {model_id:<46} {rate:8.1f} pairs/s  ({dt:.1f}s / {len(pairs)} pairs)")
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  {model_id:<46} FAILED {type(exc).__name__}: {str(exc)[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())