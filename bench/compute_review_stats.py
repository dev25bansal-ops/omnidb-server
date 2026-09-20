"""Compute all review-point statistics from the raw per-query benchmark JSONs.

Outputs:
  - Table II rows: raw paired Wilcoxon p (recomputed) + p_Holm (joint, m=18)
    + p_Holm within-family (bm25 family, hybrid family)
  - per-corpus surviving query counts (dropped-query reporting)
  - per-query CE deltas for the grid +CE vs hybrid (for an ECDF / summary)
  - Spearman rho between weak-stage recall headroom and CE delta
All numbers here are recomputed from per-query arrays, nothing is hand-entered.
"""
import json, math
import numpy as np
from scipy.stats import wilcoxon, spearmanr

def paired_wilcoxon(a, b):
    a = np.asarray(a); b = np.asarray(b)
    delta = a - b
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return 1.0
    stat = wilcoxon(nonzero, alternative="two-sided", method="approx")
    return float(stat.pvalue)

def holm(pvalues):
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered); adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (m - i)))
        adjusted[name] = running
    return adjusted

# ---- locate the four window JSONs ------------------------------------------
import os
B = "D:/OMNIDB/omnidb-server/bench"
files = {
    "fiqa":     os.path.join(B, "rerank-window-results.json"),
    "nfcorpus": os.path.join(B, "nfcorpus-rerank-window-results.json"),
    "scidocs":  "E:/omnidb-bench-tmp/sweep-rerank-scidocs.json",
    "scifact":  os.path.join(B, "scifact-window-results.json"),
}

# navigate a JSON to get per-query arrays for (stage, window)
def get_cell(d, stage, w):
    node = d[stage]
    if "cells" in node:
        node = node["cells"]
    c = node[str(w)]
    return c["base_ndcg10"], c["rerank_ndcg10"]

# Table II rows, in the order they appear in the paper:
# (corpus, stage-label, stage-key, window, published_delta, published_p)
rows = [
    ("fiqa",     "bm25",            "bm25",      25,  +0.0637, "<0.0001"),
    ("fiqa",     "bm25",            "bm25",      50,  +0.0728, "<0.0001"),
    ("fiqa",     "bm25",            "bm25",     100,  +0.0808, "<0.0001"),
    ("fiqa",     "hybrid(bge-base)","bge-base",  25,  -0.0093,  "0.1062"),
    ("fiqa",     "hybrid(bge-base)","bge-base",  50,  -0.0195,  "0.0074"),
    ("fiqa",     "hybrid(bge-base)","bge-base", 100,  -0.0407,  "<0.0001"),
    ("nfcorpus", "bm25",            "bm25",      25,  +0.0082,  "0.5040"),
    ("nfcorpus", "bm25",            "bm25",     100,  +0.0004,  "0.8945"),
    ("nfcorpus", "hybrid(bge-base)","bge-base",  25,  -0.0383,  "<0.0001"),
    ("nfcorpus", "hybrid(bge-base)","bge-base", 100,  -0.0541,  "<0.0001"),
    ("scidocs",  "bm25",            "bm25",      25,  +0.0088,  "0.0062"),
    ("scidocs",  "bm25",            "bm25",     100,  +0.0033,  "0.3718"),
    ("scidocs",  "hybrid(bge-base)","bge-base",  25,  -0.0206,  "<0.0001"),
    ("scidocs",  "hybrid(bge-base)","bge-base", 100,  -0.0475,  "<0.0001"),
    ("scifact",  "bm25",            "bm25",      25,  +0.0484,  "<0.0001"),
    ("scifact",  "bm25",            "bm25",     100,  +0.0383,  "0.0017"),
    ("scifact",  "hybrid(bge-base)","bge-base",  25,  -0.0194,  "0.0900"),
    ("scifact",  "hybrid(bge-base)","bge-base", 100,  -0.0327,  "0.0155"),
]

raw = {}
computed = []
for (corpus, label, skey, w, pub_d, pub_p) in rows:
    d = json.load(open(files[corpus], encoding="utf-8"))
    base, rr = get_cell(d, skey, w)
    p = paired_wilcoxon(rr, base)   # reranked vs base
    delta = float(np.mean(rr) - np.mean(base))
    name = f"{corpus}/{label}/w{w}"
    raw[name] = p
    computed.append((name, corpus, label, skey, w, len(base), delta, p, pub_d, pub_p))

joint = holm(raw)
fam_bm25 = holm({k: v for k, v in raw.items() if "/bm25/" in k or k.endswith("bm25/w25") or k.endswith("bm25/w50") or k.endswith("bm25/w100")})
# careful family split by stage-key: bm25 vs hybrid
fam_bm25 = holm({k: v for k, v in raw.items() if k.split("/")[1] == "bm25"})
fam_hyb  = holm({k: v for k, v in raw.items() if k.split("/")[1] == "hybrid(bge-base)"})

print("=" * 100)
print("TABLE II RECOMPUTED (raw p recomputed via paired Wilcoxon, method='approx')")
print("=" * 100)
print(f"{'row':<28}{'n':>5}{'delta':>9}{'p_raw':>10}{'pH_joint':>10}{'pH_bm25':>10}{'pH_hyb':>10}  pub_p")
for (name, corpus, label, skey, w, n, delta, p, pub_d, pub_p) in computed:
    jb = joint[name]; fb = fam_bm25.get(name, float('nan')); fh = fam_hyb.get(name, float('nan'))
    print(f"{name:<28}{n:>5}{delta:>+9.4f}{p:>10.4f}{jb:>10.4f}{fb:>10.4f}{fh:>10.4f}  {pub_p}")

# ---- surviving query counts (dropped queries, point 15) ---------------------
print()
print("=" * 100)
print("SURVIVING QUERY COUNTS (per-corpus, from per-query array lengths)")
print("=" * 100)
full_q = {"nfcorpus": 323, "scifact": 300, "scidocs": 1000, "fiqa": 648}
for corpus in ["nfcorpus", "scifact", "scidocs", "fiqa"]:
    d = json.load(open(files[corpus], encoding="utf-8"))
    base, _ = get_cell(d, "bm25", 25)
    n_surv = len(base)
    print(f"{corpus:<10} full_q={full_q[corpus]:>5}  surviving={n_surv:>5}  dropped={full_q[corpus]-n_surv:>4}")

# ---- per-query CE deltas for the grid (point 12: ECDF / summary) ------------
print()
print("=" * 100)
print("GRID +CE vs hybrid: per-query delta summary (for ECDF figure + text)")
print("=" * 100)
grid = json.load(open(os.path.join(B, "research-v3-results.json"), encoding="utf-8"))
print(f"{'corpus/embedder':<22}{'n':>5}{'%improved':>11}{'%harmed':>9}{'median':>9}{'mean':>9}  n_impr n_harm")
all_deltas_by_cell = {}
for corpus in ["nfcorpus", "scifact", "scidocs", "fiqa"]:
    for emb in ["minilm", "bge-small", "e5-small", "bge-base"]:
        h = grid[corpus][emb]["omnidb"]["per_query"]
        c = grid[corpus][emb]["omnidb-ce"]["per_query"]
        d = np.asarray(c) - np.asarray(h)
        imp = int((d > 0).sum()); harm = int((d < 0).sum())
        n = len(d)
        all_deltas_by_cell[f"{corpus}/{emb}"] = d
        print(f"{corpus+'/'+emb:<22}{n:>5}{100*imp/n:>10.1f}%{100*harm/n:>8.1f}%{np.median(d):>+9.4f}{np.mean(d):>+9.4f}  {imp:>5} {harm:>5}")

# combined ECDF data point (all 16 cells pooled) for the figure
pooled = np.concatenate(list(all_deltas_by_cell.values()))
import json as _j
out_ecdf = {
    "cells": {k: v.tolist() for k, v in all_deltas_by_cell.items()},
    "pooled": pooled.tolist(),
}
with open(os.path.join(B, "review_ce_deltas.json"), "w") as f:
    _j.dump(out_ecdf, f)
print(f"\npooled n={len(pooled)}  %improved={100*(pooled>0).mean():.1f}%  %harmed={100*(pooled<0).mean():.1f}%")

# ---- Spearman rho: recall headroom vs CE delta (point 13) -------------------
print()
print("=" * 100)
print("SPEARMAN rho: weak-stage recall headroom vs CE delta (window probe, w=100)")
print("=" * 100)
# recall headroom for weak bm25 stage = hybrid recall@w - bm25 recall@w
# CE delta = rerank-bm25 nDCG - base bm25 nDCG
pts = []
for corpus, skey_b, skey_h in [("fiqa","bm25","bge-base"),("nfcorpus","bm25","bge-base"),
                               ("scidocs","bm25","bge-base"),("scifact","bm25","bge-base")]:
    d = json.load(open(files[corpus], encoding="utf-8"))
    bb = d[skey_b]["cells"] if "cells" in d[skey_b] else d[skey_b]
    # bm25 stage recall@100
    bm25_r100 = bb["100"]["stage_recall_at_w"]
    # hybrid (bge-base) stage recall@100
    hn = d[skey_h]; hc = hn["cells"] if "cells" in hn else hn
    hyb_r100 = hc["100"]["stage_recall_at_w"]
    base, rr = bb["100"]["base_ndcg10"], bb["100"]["rerank_ndcg10"]
    ce_delta = float(np.mean(rr) - np.mean(base))
    headroom = hyb_r100 - bm25_r100
    pts.append((corpus, bm25_r100, hyb_r100, headroom, ce_delta))
    print(f"{corpus:<10} bm25_r100={bm25_r100:.4f}  hyb_r100={hyb_r100:.4f}  headroom={headroom:+.4f}  ce_delta={ce_delta:+.4f}")
hs = [p[3] for p in pts]; ds = [p[4] for p in pts]
rho, pval = spearmanr(hs, ds)
print(f"\nSpearman rho = {rho:.3f}  (p={pval:.4f})  over {len(pts)} weak-stage cells")
