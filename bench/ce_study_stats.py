"""Locked inference utilities for the held-out CE study.

Sign-flip permutation tests target the paired mean difference and retain exact
zero differences as zeros. Bootstrap resamples query IDs; clustered bootstrap
resamples IDs once per corpus to avoid treating cells sharing queries as
independent.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np


def sign_flip_mean_test(deltas: Sequence[float], *, n_permutations: int = 100_000,
                        seed: int = 20261008) -> dict:
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 1 or len(d) == 0 or not np.all(np.isfinite(d)):
        raise ValueError("deltas must be a nonempty finite 1-D array")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    observed = float(d.mean())
    rng = np.random.Generator(np.random.PCG64(seed))
    exceed = 0
    chunk = min(4096, n_permutations)
    done = 0
    while done < n_permutations:
        n = min(chunk, n_permutations - done)
        signs = rng.integers(0, 2, size=(n, len(d)), dtype=np.int8) * 2 - 1
        permuted = (signs * d).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        done += n
    return {"mean_delta": observed, "p_two_sided": (exceed + 1) / (n_permutations + 1),
            "n_permutations": n_permutations, "seed": seed,
            "n_queries": len(d), "n_exact_zero": int(np.count_nonzero(d == 0))}


def paired_bootstrap_ci(deltas: Sequence[float], *, n_resamples: int = 20_000,
                        seed: int = 20261009, confidence: float = 0.95) -> dict:
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 1 or len(d) == 0 or not np.all(np.isfinite(d)):
        raise ValueError("deltas must be a nonempty finite 1-D array")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(n_resamples, dtype=np.float64)
    chunk = min(2048, n_resamples)
    start = 0
    while start < n_resamples:
        n = min(chunk, n_resamples - start)
        idx = rng.integers(0, len(d), size=(n, len(d)))
        means[start:start+n] = d[idx].mean(axis=1)
        start += n
    tail = (1.0 - confidence) * 50.0
    lo, hi = np.percentile(means, [tail, 100.0-tail])
    return {"mean_delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "confidence": confidence, "n_resamples": n_resamples, "seed": seed,
            "n_queries": len(d)}


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not pvalues:
        return {}
    for key, p in pvalues.items():
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"invalid p-value for {key}: {p}")
    ordered = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m, running, result = len(ordered), 0.0, {}
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, (m-i) * pvalues[key]))
        result[key] = running
    return result


def clustered_cell_bootstrap(cell_deltas: Mapping[str, Mapping[str, float]], *,
                             n_resamples: int = 20_000, seed: int = 20261010) -> dict:
    """Bootstrap the macro mean while preserving shared query-ID clusters.

    Each mapping is cell -> query_id -> delta. Within each corpus, one query-ID
    resample is shared by all cells; cell means are averaged, then corpora are
    macro-averaged. The reported point estimate uses this exact estimand.
    """
    if not cell_deltas:
        raise ValueError("cell_deltas is empty")
    by_corpus: dict[str, dict[str, dict[str, float]]] = {}
    for cell, vals in cell_deltas.items():
        corpus = cell.split("/", 1)[0]
        clean = {str(q): float(v) for q, v in vals.items()}
        if not clean or not np.all(np.isfinite(list(clean.values()))):
            raise ValueError(f"{cell}: values must be nonempty and finite")
        by_corpus.setdefault(corpus, {})[cell] = clean
    corpora = sorted(by_corpus)
    ids_by_corpus = {c: sorted(set().union(*(set(v) for v in by_corpus[c].values())))
                     for c in corpora}
    if any(not ids_by_corpus[c] for c in corpora):
        raise ValueError("empty corpus query cluster")

    def estimate_for(sampled_ids, cells):
        means=[]
        for vals in cells.values():
            observed=[vals[q] for q in sampled_ids if q in vals]
            if observed:
                means.append(float(np.mean(observed)))
        return float(np.mean(means))

    point = float(np.mean([estimate_for(ids_by_corpus[c], by_corpus[c])
                           for c in corpora]))
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        corpus_means=[]
        for c in corpora:
            ids=ids_by_corpus[c]
            sampled=rng.choice(ids,size=len(ids),replace=True).tolist()
            corpus_means.append(estimate_for(sampled,by_corpus[c]))
        draws[b]=float(np.mean(corpus_means))
    return {"estimate":point,"ci_low":float(np.percentile(draws,2.5)),
            "ci_high":float(np.percentile(draws,97.5)),"n_resamples":n_resamples,
            "seed":seed,"n_corpora":len(corpora),"n_cells":len(cell_deltas)}

