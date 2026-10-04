"""One home for the six statistics functions the bench harnesses duplicated.

F5: dcg_at_k / ndcg_at_k / recall_at_k / bootstrap_ci / paired_wilcoxon / holm
were re-implemented per harness across three generations (research_bench.py,
_v2.py, _v3.py, compute_review_stats.py, compute_probe_27.py), verified
semantically identical. Every harness now imports this module instead, so a
metric change lands once instead of silently missing the other copies.

paired_wilcoxon returns (pvalue, rank_biserial); callers written before the
tuple convention read the first element.
"""
import math

import numpy as np

BOOTSTRAP_N = 10_000
RNG_SEED = 1234


def dcg_at_k(gains, k):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(ranked, relevant, k):
    gains = [float(relevant.get(d, 0)) for d in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)
    idcg = dcg_at_k([float(g) for g in ideal], k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def bootstrap_ci(per_query, n=BOOTSTRAP_N, seed=RNG_SEED):
    arr = np.asarray(per_query, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def paired_wilcoxon(a, b):
    from scipy.stats import wilcoxon

    delta = np.asarray(a) - np.asarray(b)
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return 1.0, 0.0
    stat = wilcoxon(nonzero, alternative="two-sided", method="approx")
    ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    rp = ranks[nonzero > 0].sum()
    rm = ranks[nonzero < 0].sum()
    return float(stat.pvalue), float((rp - rm) / ranks.sum())


def holm(pvalues):
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (m - i)))
        adjusted[name] = running
    return adjusted