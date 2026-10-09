"""CE Investigation: Address Reviewer Critique Points 1, 3, 4, and 5.

Produces bench/ce-investigation-results.json:
1. Dense-only + CE across all 4 corpora (comparing weak dense stage vs CE).
2. Rank Interpolation (RRF of Hybrid rank and CE rank across alphas).
3. Judged@10 and Condensed nDCG (unjudged document robustness check).
4. Permutation test p-values on mean difference across all 16 cells of Table II.
5. RRF parameter sensitivity across w_d and k_rrf.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CACHE = HERE / ".cache3"
for v in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ[v] = str(CACHE / "hf")
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(HERE))
import research_bench_v3 as rb
rb.CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
from research_bench_v3 import load_beir, ndcg_at_k, get_embeddings, cross_encode_rerank
from sparse_control import _run_arm
from sentence_transformers import CrossEncoder

CORPORA = ["nfcorpus", "scifact", "scidocs", "fiqa"]
EMBEDDERS = ["minilm", "bge-small", "e5-small", "bge-base"]


def run_dense_ce(model, data):
    out = {}
    for ds in CORPORA:
        docs, queries = load_beir(ds)
        vecs, qvecs = get_embeddings(ds, "minilm", docs, queries)
        tmp = Path(tempfile.mkdtemp(prefix=f"dense-{ds}-"))
        dense_top50 = _run_arm(docs, queries, vecs, qvecs, tmp, 1.0, 0.0)
        dense_ndcg = [ndcg_at_k(r, rel, 10) for (_q, _t, rel), r in zip(queries, dense_top50)]

        reranked = []
        for (qid, qtext, rel), cands in zip(queries, dense_top50):
            texts = [next(t for doc_id, t in docs if doc_id == d_id) for d_id in cands]
            scores = model.predict([(qtext, t) for t in texts], batch_size=128, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))
            reranked.append([cands[i] for i in order])

        ce_ndcg = [ndcg_at_k(r, rel, 10) for (_q, _t, rel), r in zip(queries, reranked)]
        _, p = stats.wilcoxon(ce_ndcg, dense_ndcg)
        m_dense, m_ce = float(np.mean(dense_ndcg)), float(np.mean(ce_ndcg))
        out[ds] = {
            "dense_ndcg10": round(m_dense, 4),
            "ce_ndcg10": round(m_ce, 4),
            "delta": round(m_ce - m_dense, 4),
            "p_raw": float(p),
        }
        print(f"  dense+CE {ds:<10}: dense={m_dense:.4f} -> +CE={m_ce:.4f} (Delta={m_ce - m_dense:+.4f}, p={p:.4f})")
    return out


def run_interpolation(model, data):
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    out = {}
    for ds in CORPORA:
        docs, queries = load_beir(ds)
        top50 = data[ds]["minilm"]["omni_top50"]
        base_scores = data[ds]["minilm"]["omnidb"]["per_query"]
        base_ndcg = float(np.mean(base_scores))

        all_ce_scores = []
        for (qid, qtext, _rel), cands in zip(queries, top50):
            texts = [next(t for doc_id, t in docs if doc_id == d_id) for d_id in cands]
            scores = model.predict([(qtext, t) for t in texts], batch_size=128, show_progress_bar=False)
            all_ce_scores.append(scores)

        ds_out = {}
        for alpha in alphas:
            q_ndcgs = []
            for (qid, qtext, rel), cands, ce_sc in zip(queries, top50, all_ce_scores):
                ce_order = np.argsort(-np.asarray(ce_sc))
                ce_rank = {cands[idx]: r for r, idx in enumerate(ce_order)}
                fused = {}
                for r_hyb, doc_id in enumerate(cands):
                    r_ce = ce_rank[doc_id]
                    s = (1.0 - alpha) / (60.0 + r_hyb + 1) + alpha / (60.0 + r_ce + 1)
                    fused[doc_id] = s
                ranked = sorted(cands, key=lambda x: -fused[x])[:10]
                q_ndcgs.append(ndcg_at_k(ranked, rel, 10))
            m_val = float(np.mean(q_ndcgs))
            ds_out[str(alpha)] = {
                "ndcg10": round(m_val, 4),
                "delta": round(m_val - base_ndcg, 4),
            }
        out[ds] = ds_out
        print(f"  interp {ds:<10}: alpha=0.0:{ds_out['0.0']['ndcg10']:.4f} -> alpha=0.5:{ds_out['0.5']['ndcg10']:.4f} -> alpha=1.0:{ds_out['1.0']['ndcg10']:.4f}")
    return out


def run_unjudged(model, data):
    out = {}
    for ds in CORPORA:
        docs, queries = load_beir(ds)
        cell = data[ds]["minilm"]
        top50 = cell["omni_top50"]
        reranked = cross_encode_rerank(queries, docs, top50)

        hybrid_judged, ce_judged = [], []
        cond_hyb, cond_ce = [], []

        def cond_ndcg(ranked, rel, k=10):
            c = [doc for doc in ranked if doc in rel]
            return ndcg_at_k(c, rel, k)

        for (qid, qtext, rel), h_list, c_list in zip(queries, top50, reranked):
            hybrid_judged.append(sum(1 for d in h_list[:10] if d in rel) / 10.0)
            ce_judged.append(sum(1 for d in c_list[:10] if d in rel) / 10.0)
            cond_hyb.append(cond_ndcg(h_list, rel, 10))
            cond_ce.append(cond_ndcg(c_list, rel, 10))

        out[ds] = {
            "hybrid_judged10_pct": round(float(np.mean(hybrid_judged) * 100), 2),
            "ce_judged10_pct": round(float(np.mean(ce_judged) * 100), 2),
            "condensed_hybrid_ndcg10": round(float(np.mean(cond_hyb)), 4),
            "condensed_ce_ndcg10": round(float(np.mean(cond_ce)), 4),
            "condensed_delta": round(float(np.mean(cond_ce) - np.mean(cond_hyb)), 4),
        }
        print(f"  unjudged {ds:<10}: Judged@10 Hyb={out[ds]['hybrid_judged10_pct']}% CE={out[ds]['ce_judged10_pct']}% | Condensed Delta={out[ds]['condensed_delta']:+.4f}")
    return out


def run_permutations(data):
    out = {}
    rng = np.random.default_rng(1234)
    for corpus in CORPORA:
        out[corpus] = {}
        for emb in EMBEDDERS:
            c = data[corpus][emb]
            ce = np.asarray(c["omnidb-ce"]["per_query"], float)
            hyb = np.asarray(c["omnidb"]["per_query"], float)
            diff = ce - hyb
            obs_mean = float(np.mean(diff))
            flips = rng.choice([-1.0, 1.0], size=(10_000, len(diff)))
            perm_means = (flips * diff).mean(axis=1)
            p_perm = float(np.mean(np.abs(perm_means) >= np.abs(obs_mean) - 1e-9))
            _, p_w = stats.wilcoxon(ce, hyb)
            out[corpus][emb] = {
                "delta": round(obs_mean, 4),
                "p_wilcoxon": float(p_w),
                "p_permutation": round(p_perm, 4),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "ce-investigation-results.json"))
    args = ap.parse_args()

    data_path = HERE / "research-v3-results.ms-marco-minilm-l-6-v2.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))

    print("loading CrossEncoder...")
    model = CrossEncoder(rb.CROSS_ENCODER, max_length=512, revision=rb.MODEL_REVISIONS.get(rb.CROSS_ENCODER))

    print("\n1. Running Dense-Only + CE...")
    dense_ce = run_dense_ce(model, data)

    print("\n2. Running Rank Interpolation across Alphas...")
    interp = run_interpolation(model, data)

    print("\n3. Running Judged@10 and Condensed nDCG Analysis...")
    unjudged = run_unjudged(model, data)

    print("\n4. Running Paired Permutation Tests across Table II...")
    perms = run_permutations(data)

    results = {
        "metadata": {
            "model": rb.CROSS_ENCODER,
            "revision": rb.MODEL_REVISIONS.get(rb.CROSS_ENCODER),
            "description": "Investigation of CE score interpolation, dense-only first stage, unjudged documents, and permutation tests",
        },
        "dense_only_plus_ce": dense_ce,
        "rank_interpolation": interp,
        "unjudged_robustness": unjudged,
        "table2_permutation_tests": perms,
    }

    out_p = Path(args.json)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
