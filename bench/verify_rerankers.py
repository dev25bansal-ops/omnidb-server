"""Smoke-verify every pinned reranker: relevant must outrank irrelevant.

This is a *correctness* gate, not a benchmark. A model that loads but scores a
random head (e.g. a missing `score.weight`, or a CrossEncoder path that does not
match the checkpoint's intended module) will pass a "loads OK" check and fail
this one. mxbai-rerank-base-v2 is exactly that case: it is a Qwen2 causal LM
with a `1_LogitScore` module, so it needs sentence-transformers >= 5.4.

Usage: python bench/verify_rerankers.py [--models m1,m2] [--device cuda]
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

PAIRS = [
    ("What is hypertension?",
     "Hypertension is a long-term medical condition of high blood pressure."),
    ("What is hypertension?", "A poem about the ocean and sailing ships."),
    ("How do neural networks learn?",
     "Neural networks are trained by gradient descent on a loss function."),
    ("How do neural networks learn?", "The recipe calls for two cups of flour."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(RERANKER_REVISIONS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    cache = (HERE / ".cache3" / "hf").resolve()
    for var in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ[var] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"

    import transformers
    from sentence_transformers import CrossEncoder

    print(f"sentence-transformers {__import__('sentence_transformers').__version__}, "
          f"transformers {transformers.__version__}, device={args.device}")
    failures = []
    for model_id in args.models.split(","):
        if model_id not in RERANKER_REVISIONS:
            print(f"FAIL {model_id}: not a pinned study model")
            failures.append(model_id)
            continue
        revision = RERANKER_REVISIONS[model_id]
        t0 = time.perf_counter()
        try:
            model = CrossEncoder(model_id, revision=revision,
                                 max_length=args.max_length, device=args.device,
                                 local_files_only=True)
            scores = [float(x) for x in model.predict(PAIRS, batch_size=4,
                                                      show_progress_bar=False)]
            ok = (scores[0] > scores[1]) and (scores[2] > scores[3])
            verdict = "OK " if ok else "BAD"
            print(f"{verdict} {model_id:<46} rev={revision[:10]} "
                  f"scores={[round(s, 4) for s in scores]} "
                  f"sec={time.perf_counter() - t0:.1f}")
            if not ok:
                failures.append(model_id)
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {model_id:<46} {type(exc).__name__}: {str(exc)[:220]} "
                  f"sec={time.perf_counter() - t0:.1f}")
            failures.append(model_id)

    print()
    print(f"RESULT: {'FAIL' if failures else 'OK'} "
          f"({len(failures)} failing: {failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())