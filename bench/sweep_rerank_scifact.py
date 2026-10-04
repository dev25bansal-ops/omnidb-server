"""Sweep: does the fiqa reranker-window finding replicate on scifact?

The fiqa probe (bench/probe_rerank_window.py; "fiqa" section of
rerank-window-results.md) found: bge-reranker-base rerank of omnidb-hybrid
top-w is a net loss at every window and gets monotonically more negative
as w grows on both embedders, while on the weak bm25s first stage it gains
at every window with the gain growing in w. This script runs the same
design on scifact to test replication on a corpus where the hybrid first
stage is much stronger (v3: nDCG@10 0.71-0.75 vs fiqa's 0.38-0.40).

Cells: {omnidb-hybrid, bm25s} x {w=25, w=100} x {minilm, bge-base}
(bm25s is embedder-free, reported once). w=50 is skipped: the fiqa
probe showed both embedders' deltas are sandwiched between w=25 and
w=100 there, and v3 already measured hybrid w=50 deltas on scifact
(minilm +0.0016, bge-base -0.0275) — the sweep brackets it.

Method identical to the fiqa probe (all retrieval/CE/scoring helpers are
imported from probe_rerank_window, which reuses research_bench_v3's
load_beir / get_embeddings / ndcg_at_k / paired_wilcoxon): fresh hybrid
retrieval per window, single bm25s top-100 pass truncated per window,
CE scores computed once per (query, doc) over each query's union of
window candidates, windows re-sorted from the cached scores. Metric:
nDCG@10 reranked vs no-rerank per cell, paired Wilcoxon over queries.

Usage: python bench/sweep_rerank_scifact.py  (~10-15 min, all cached)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import probe_rerank_window as prw
import research_bench_v3 as rb

DATASET = "scifact"
WINDOWS = (25, 100)
MAX_W = 100
EMBED_LIST = ("minilm", "bge-base")

OUT_JSON = HERE / "scifact-window-results.json"
OUT_TABLE = HERE / "scifact-window-table.txt"


def main() -> None:
    import tempfile

    hf_root = Path("D:/OMNIDB/omnidb-server/.hf-cache")
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME"):
        os.environ[var] = str(hf_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    prw.TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"], os.environ["TEMP"] = str(prw.TMP_ROOT), str(prw.TMP_ROOT)
    tempfile.tempdir = str(prw.TMP_ROOT)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder(rb.CROSS_ENCODER, max_length=512, device=device)
    docs_all, queries = rb.load_beir(DATASET)
    rels = [rb.relevant_docs(rel) for (_q, _t, rel) in queries]
    print(f"{DATASET}: {len(docs_all)} docs, {len(queries)} queries; "
          f"reranker {rb.CROSS_ENCODER} on {device}", flush=True)

    results: dict = {}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    # ---- bm25 first stage: retrieve top-100 once, truncate per window ----
    if "cells" not in results.get("bm25", {}):
        if "ranked_top100" not in results.get("bm25", {}):
            t0 = time.perf_counter()
            bm_ranked = prw.retrieve_bm25(docs_all, queries, MAX_W)
            print(f"bm25 top-{MAX_W} indexed+retrieved in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
            results["bm25"] = {"ranked_top100": bm_ranked}
            OUT_JSON.write_text(json.dumps(results), encoding="utf-8")
        bm_ranked = results["bm25"]["ranked_top100"]
        scores = prw.ce_score_queries(ce, queries, docs_all, bm_ranked)
        cells = {}
        for w in WINDOWS:
            rr = [prw.rerank(r[:w], s) for r, s in zip(bm_ranked, scores)]
            cells[str(w)] = prw.cell_stats(queries, bm_ranked, rr, rels, w)
        results["bm25"]["cells"] = cells
        OUT_JSON.write_text(json.dumps(results), encoding="utf-8")
        for w in WINDOWS:
            c = cells[str(w)]
            print(f"  bm25 w={w}: base={c['base_mean']:.4f} "
                  f"rr={c['rerank_mean']:.4f} d={c['delta']:+.4f}", flush=True)

    # ---- omnidb-hybrid cells per embedder ----
    for emb in EMBED_LIST:
        print(f"=== {emb} ===", flush=True)
        results.setdefault(emb, {})
        if "cells" in results[emb]:
            print("  cached, skipping", flush=True)
            continue
        vecs, qvecs = rb.get_embeddings(DATASET, emb, docs_all, queries)
        by_w = prw.retrieve_omnidb(docs_all, queries, vecs, qvecs, WINDOWS)

        union = [list(dict.fromkeys(a + b)) for a, b in
                 zip(by_w[WINDOWS[0]], by_w[WINDOWS[1]])]
        t0 = time.perf_counter()
        scores = prw.ce_score_queries(ce, queries, docs_all, union)
        print(f"  CE scored {sum(len(m) for m in scores)} pairs in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

        cells = {}
        for w in WINDOWS:
            rr = [prw.rerank(r, s) for r, s in zip(by_w[w], scores)]
            cells[str(w)] = prw.cell_stats(queries, by_w[w], rr, rels, w)
            c = cells[str(w)]
            print(f"  omni w={w}: base={c['base_mean']:.4f} "
                  f"rr={c['rerank_mean']:.4f} d={c['delta']:+.4f}", flush=True)
        results[emb]["cells"] = cells
        OUT_JSON.write_text(json.dumps(results), encoding="utf-8")

    lines, v = render(results)
    print(v, flush=True)
    OUT_TABLE.write_text("\n".join(lines) + "\n\n" + v + "\n",
                         encoding="utf-8")
    print(f"wrote {OUT_TABLE}", flush=True)


def render(results: dict) -> tuple[list[str], str]:
    lines = [
        "| first stage | embedder | w | stage recall@w | nDCG@10 base | "
        "nDCG@10 reranked | delta | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = []
    for stage, emb in (("bm25", "-"), ("omnidb-hybrid", "minilm"),
                       ("omnidb-hybrid", "bge-base")):
        node = results["bm25"] if stage == "bm25" else results[emb]
        for w in WINDOWS:
            c = node["cells"][str(w)]
            lines.append(
                f"| {stage} | {emb} | {w} | {c['stage_recall_at_w']:.4f} "
                f"| {c['base_mean']:.4f} | {c['rerank_mean']:.4f} "
                f"| {c['delta']:+.4f} | {c['p_wilcoxon']:.4f} |")
            rows.append((stage, emb, w, c))
    return lines, verdict(rows)


def verdict(rows: list[tuple]) -> str:
    def get(stage: str, emb: str, w: int) -> dict:
        for s, e, ww, c in rows:
            if s == stage and e == emb and ww == w:
                return c
        raise KeyError((stage, emb, w))

    bm25 = [get("bm25", "-", w) for w in WINDOWS]
    omni = {emb: [get("omnidb-hybrid", emb, w) for w in WINDOWS]
            for emb in EMBED_LIST}
    bm_gain_grows = bm25[1]["delta"] > bm25[0]["delta"]
    bm_all_pos = all(c["delta"] > 0 for c in bm25)
    om_flat = [c for cells in omni.values() for c in cells]
    om_all_neg = all(c["delta"] < 0 for c in om_flat)
    om_mono_worse = all(omni[emb][1]["delta"] < omni[emb][0]["delta"]
                        for emb in EMBED_LIST)
    best_om = max((c["delta"] for c in om_flat), default=0.0)
    om_sig = [(emb, w, c) for emb in EMBED_LIST for w, c in
              zip(WINDOWS, omni[emb]) if c["delta"] < 0 and c["p_wilcoxon"] < 0.05]
    full = bm_all_pos and bm_gain_grows and om_all_neg and om_mono_worse
    bm_str = " -> ".join(f"w{w}:{c['delta']:+.4f}" for w, c in
                         zip(WINDOWS, bm25))
    om_str = {emb: " -> ".join(f"w{w}:{c['delta']:+.4f}" for w, c in
                               zip(WINDOWS, omni[emb])) for emb in EMBED_LIST}

    rep = [
        f"one-line: fiqa {'REPLICATES' if full else 'PARTIALLY REPLICATES / DIVERGES'} on scifact - "
        f"bm25s {bm_str} "
        f"({'gain grows with w' if bm_gain_grows else 'gain does NOT grow'}), "
        f"omni minilm {om_str['minilm']}, "
        f"omni bge-base {om_str['bge-base']} "
        f"(best hybrid delta {best_om:+.4f}).",
    ]
    for emb, w, c in om_sig:
        rep.append(f"  significant hybrid loss: {emb} w={w} "
                   f"{c['delta']:+.4f} (p={c['p_wilcoxon']:.4f})")
    return "VERDICT: " + " ".join(rep)


if __name__ == "__main__":
    main()
