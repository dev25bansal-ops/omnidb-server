"""Realistic throughput for the five study rerankers, on distinct documents.

`bench_reranker_throughput.py` repeated ONE query-document pair, which the GPU
caches across the batch; it reported 3-8x the rate a real candidate pool
achieves, where every pair is a different document. Measured here with 600
distinct documents drawn from a fixed vocabulary, batch 32, max_length 512.

Usage: python bench/measure_realistic_rates.py [--pairs 600]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ce_study_runner import RERANKER_REVISIONS  # noqa: E402

WORDS = ("retrieval dense lexical hybrid fusion rerank window candidate pool "
         "document query relevance score evaluation benchmark corpus").split()


def make_pairs(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        doc = " ".join(rng.choice(WORDS) for _ in range(60)) + f" id{i}"
        out.append((f"query about {rng.choice(WORDS)} {i}", doc))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(RERANKER_REVISIONS))
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=str(HERE / "measured-reranker-rates.json"))
    args = ap.parse_args()

    cache = (HERE / ".cache3" / "hf").resolve()
    for var in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ[var] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import CrossEncoder

    pairs = make_pairs(args.pairs)
    rates = {}
    print(f"distinct documents, batch {args.batch_size}, device {args.device}")
    for model_id in args.models.split(","):
        rev = RERANKER_REVISIONS[model_id]
        try:
            model = CrossEncoder(model_id, revision=rev, max_length=512,
                                 device=args.device, local_files_only=True)
            model.predict(pairs[:args.batch_size], batch_size=args.batch_size,
                          show_progress_bar=False)
            t0 = time.perf_counter()
            model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            dt = time.perf_counter() - t0
            rates[model_id] = len(pairs) / dt
            print(f"  {model_id:<46} {len(pairs)/dt:8.1f} pairs/s")
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"  {model_id:<46} FAILED {type(exc).__name__}: {str(exc)[:140]}")

    Path(args.json).write_text(json.dumps(rates, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())