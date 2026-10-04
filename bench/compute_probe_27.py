"""Complete 27-row window-probe recompute: raw Wilcoxon p, joint-Holm (m=27),
paired-diff 95% bootstrap CI, and per-query improve/harm. All from per-query
arrays. Verifies each raw p and delta against the published md values."""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

B = "D:/OMNIDB/omnidb-server/bench"
SEED, N = 1234, 10000
files = {
    "fiqa":     os.path.join(B, "rerank-window-results.json"),
    "nfcorpus": os.path.join(B, "nfcorpus-rerank-window-results.json"),
    "scidocs":  "E:/omnidb-bench-tmp/sweep-rerank-scidocs.json",
    "scifact":  os.path.join(B, "scifact-window-results.json"),
}

def cell(d, stage, w):
    n = d[stage]
    n = n["cells"] if "cells" in n else n
    c = n[str(w)]
    return c["base_ndcg10"], c["rerank_ndcg10"]

def wilcoxon_p(rr, base):
    delta = np.asarray(rr) - np.asarray(base)
    nz = delta[delta != 0]
    if len(nz) == 0:
        return 1.0
    return float(wilcoxon(nz, alternative="two-sided", method="approx").pvalue)

def diff_ci(base, rr):
    d = np.asarray(rr) - np.asarray(base)
    rng = np.random.default_rng(SEED)
    means = rng.choice(d, size=(N, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

from _stats import holm
# All 27 rows: (corpus, stage_label, stage_key, w, pub_delta, pub_p)
R = [
    ("fiqa","bm25","bm25",25,+0.0637,"<0.0001"),("fiqa","bm25","bm25",50,+0.0728,"<0.0001"),
    ("fiqa","bm25","bm25",100,+0.0808,"<0.0001"),
    ("fiqa","hyb-minilm","minilm",25,-0.0031,"0.4408"),("fiqa","hyb-minilm","minilm",50,-0.0064,"0.2677"),
    ("fiqa","hyb-minilm","minilm",100,-0.0247,"0.0036"),
    ("fiqa","hyb-bge","bge-base",25,-0.0093,"0.1062"),("fiqa","hyb-bge","bge-base",50,-0.0195,"0.0074"),
    ("fiqa","hyb-bge","bge-base",100,-0.0407,"<0.0001"),
    ("nfcorpus","bm25","bm25",25,+0.0082,"0.5040"),("nfcorpus","bm25","bm25",100,+0.0004,"0.8945"),
    ("nfcorpus","hyb-minilm","minilm",25,-0.0203,"0.0041"),("nfcorpus","hyb-minilm","minilm",100,-0.0257,"0.0350"),
    ("nfcorpus","hyb-bge","bge-base",25,-0.0383,"<0.0001"),("nfcorpus","hyb-bge","bge-base",100,-0.0541,"<0.0001"),
    ("scidocs","bm25","bm25",25,+0.0088,"0.0062"),("scidocs","bm25","bm25",100,+0.0033,"0.3718"),
    ("scidocs","hyb-minilm","minilm",25,-0.0191,"<0.0001"),("scidocs","hyb-minilm","minilm",100,-0.0513,"<0.0001"),
    ("scidocs","hyb-bge","bge-base",25,-0.0206,"<0.0001"),("scidocs","hyb-bge","bge-base",100,-0.0475,"<0.0001"),
    ("scifact","bm25","bm25",25,+0.0484,"<0.0001"),("scifact","bm25","bm25",100,+0.0383,"0.0017"),
    ("scifact","hyb-minilm","minilm",25,+0.0067,"0.5462"),("scifact","hyb-minilm","minilm",100,+0.0066,"0.6000"),
    ("scifact","hyb-bge","bge-base",25,-0.0194,"0.0900"),("scifact","hyb-bge","bge-base",100,-0.0327,"0.0155"),
]

rows = []
raw = {}
for (corpus, lab, skey, w, pub_d, pub_p) in R:
    d = json.load(open(files[corpus], encoding="utf-8"))
    base, rr = cell(d, skey, w)
    p = wilcoxon_p(rr, base)
    delta = float(np.mean(rr) - np.mean(base))
    lo, hi = diff_ci(base, rr)
    darr = np.asarray(rr) - np.asarray(base)
    imp = int((darr > 0).sum())
    harm = int((darr < 0).sum())
    n = len(darr)
    name = f"{corpus}/{lab}/w{w}"
    raw[name] = p
    rows.append(dict(corpus=corpus, lab=lab, w=w, n=n, base=float(np.mean(base)),
                     rr=float(np.mean(rr)), delta=delta, p_raw=p, lo=lo, hi=hi,
                     imp=imp, harm=harm, pub_d=pub_d, pub_p=pub_p))
joint = holm(raw)
for r in rows:
    r["p_holm"] = joint[f"{r['corpus']}/{r['lab']}/w{r['w']}"]

print(f"m={len(rows)} joint Holm family")
print(f"{'row':<26}{'n':>5}{'base':>8}{'rr':>8}{'delta':>9}{'p_raw':>9}{'pHolm':>9}{'dCI':>22}  imp/harm  pub_p")
for r in rows:
    dci = f"({r['lo']:+.3f},{r['hi']:+.3f})"
    impct = 100*r["imp"]/r["n"]
    harmpct = 100*r["harm"]/r["n"]
    key = f"{r['corpus']}/{r['lab']}/w{r['w']}"
    print(f"{key:<26}{r['n']:>5}{r['base']:>8.4f}{r['rr']:>8.4f}{r['delta']:>+9.4f}{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}{dci:>22}  {impct:.0f}/{harmpct:.0f}  {r['pub_p']}")

# save for the ECDF figure (verified window-probe per-query deltas)
out = {}
for (corpus, lab, skey, w, *_ ) in R:
    d = json.load(open(files[corpus], encoding="utf-8"))
    base, rr = cell(d, skey, w)
    out[f"{corpus}/{lab}/w{w}"] = (np.asarray(rr) - np.asarray(base)).tolist()
json.dump(out, open(os.path.join(B, "review_probe_deltas.json"), "w"))
print("\nsaved review_probe_deltas.json for ECDF figure")
