"""Sweep: replicate the fiqa rerank-window probe on scidocs.

fiqa (probe_rerank_window.py, section above): BAAI/bge-reranker-base
gains only where the first stage is weak (bm25s +0.064/+0.073/+0.081 at
w=25/50/100) and loses on omnidb-hybrid at every window, monotonically
worse as the window grows. research_bench_v3 saw the same hybrid-stage
sign on scidocs at one window (omnidb+CE -0.0197 minilm / -0.0238
bge-base at top-50). This sweep tests whether that MECHANISM (reranker
value scales with first-stage weakness, not window size) generalizes:
{omnidb-hybrid, bm25s} x {25, 100} x {minilm, bge-base} on scidocs,
nDCG@10 reranked vs no-rerank, paired Wilcoxon per cell.

All heavy helpers (store wiring, fresh-per-window omnidb retrieval,
single-leg bm25s prefix, CE pair-score caching, cell stats) are reused
verbatim from probe_rerank_window; embeddings from bench/.cache3.
Results checkpoint on E: (C: is near full); the scidocs section is
APPENDED to rerank-window-results.md, never overwriting fiqa's.

Usage: python bench/sweep_rerank_scidocs.py  (~30-60 min, models cached)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import probe_rerank_window as prw
import research_bench_v3 as rb

DATASET = "scidocs"
WINDOWS = (25, 100)
MAX_W = max(WINDOWS)
EMBED_LIST = ("minilm", "bge-base")

OUT_JSON = Path("E:/omnidb-bench-tmp/sweep-rerank-scidocs.json")
OUT_MD = HERE / "rerank-window-results.md"
SECTION = "## scidocs"


def save(results: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results), encoding="utf-8")


def rows_of(results: dict) -> Iterator[tuple[str, str, int, dict]]:
    for w in WINDOWS:
        yield "bm25", "-", w, results["bm25"]["cells"][str(w)]
    for emb in EMBED_LIST:
        for w in WINDOWS:
            yield "omnidb-hybrid", emb, w, results[emb]["cells"][str(w)]


def verdict(results: dict) -> list[str]:
    bm = {w: results["bm25"]["cells"][str(w)] for w in WINDOWS}
    om = {(e, w): results[e]["cells"][str(w)]
          for e in EMBED_LIST for w in WINDOWS}
    bm_gains_all = all(bm[w]["delta"] > 0 for w in WINDOWS)
    om_nonneg = [f"{e}/w={w}" for (e, w), c in om.items() if c["delta"] >= 0]
    om_best_key = max(om, key=lambda k: om[k]["delta"])
    bm_seq = ", ".join(f"w{w}:{bm[w]['delta']:+.4f}" for w in WINDOWS)
    om_seq = "; ".join(f"{e} w{w}:{om[(e, w)]['delta']:+.4f}"
                       for e in EMBED_LIST for w in WINDOWS)
    if bm_gains_all and not om_nonneg:
        cls = "REPLICATES the fiqa pattern"
    elif bm_gains_all or not om_nonneg:
        cls = "PARTIALLY replicates the fiqa pattern"
    else:
        cls = "REFUTES the fiqa pattern"
    line1 = (f"- one-line: {cls} - bm25: {bm_seq} (p="
             + "/".join(f"{bm[w]['p_wilcoxon']:.4f}" for w in WINDOWS)
             + f"); omnidb-hybrid: {om_seq}; best omnidb-hybrid delta "
             f"{om[om_best_key]['delta']:+.4f} ({om_best_key[0]}/w={om_best_key[1]}).")

    trend = {e: om[(e, 100)]["delta"] - om[(e, 25)]["delta"] for e in EMBED_LIST}
    both_worse = all(t < 0 for t in trend.values())
    om_trend_txt = "; ".join(f"{e}: w25 {om[(e, 25)]['delta']:+.4f} -> "
                            f"w100 {om[(e, 100)]['delta']:+.4f}" for e in EMBED_LIST)
    line2 = (f"- window trend on omnidb-hybrid: {om_trend_txt} - "
             + ("worse with window on both embedders, as on fiqa"
                if both_worse else "NOT uniformly worse with window (fiqa was)"))

    r_bm = bm[100]["stage_recall_at_w"]
    r_om100 = {e: om[(e, 100)]["stage_recall_at_w"] for e in EMBED_LIST}
    line3 = (f"- first-stage weakness (recall@100: bm25 {r_bm:.4f} vs hybrid "
             f"{min(r_om100.values()):.4f}-{max(r_om100.values()):.4f}): ")
    if bm[100]["delta"] > 0:
        hybrid_base = om[("minilm", 100)]["base_mean"]
        gap = hybrid_base - bm[100]["base_mean"]
        closed = bm[100]["rerank_mean"] - bm[100]["base_mean"]
        pct = 100 * closed / gap if gap > 0 else float("nan")
        if pct < 100:
            line3 += (f"the reranker gains on the weak stage at every tested "
                      f"window, closing {pct:.0f}% of the bm25->hybrid base gap "
                      f"at w=100 (bm25+CE {bm[100]['rerank_mean']:.4f} vs "
                      f"hybrid minilm base {hybrid_base:.4f}).")
        else:
            line3 += (f"the reranker gains on the weak stage at every tested "
                      f"window; bm25+CE at w=100 ({bm[100]['rerank_mean']:.4f}) "
                      f"EXCEEDS the hybrid minilm base ({hybrid_base:.4f}).")
    else:
        line3 += "the reranker does NOT gain even on the weak bm25 stage here."

    if bm_gains_all and not om_nonneg:
        line4 = ("- mechanism: replicates fiqa - reranker value scales with "
                 "first-stage weakness, not window size; CE-reranking a strong "
                 "hybrid stage is a net loss at every window tested, so the "
                 "fiqa product call (no CE rerank on hybrid) carries over.")
    else:
        line4 = ("- mechanism: scidocs does NOT cleanly reproduce the fiqa "
                 "sign structure; reranker value here is not predicted by "
                 "first-stage weakness alone (see one-line).")
    return [line1, line2, line3, line4]


def notes(results: dict) -> list[str]:
    return [
        "- omnidb-hybrid re-retrieved per window (fused top-w depends on",
        "  k_fetch; prefix-of-100 is not the true top-25 ranking).",
        "- bm25s single-leg: top-w is the exact prefix of top-100.",
        "- CE scores cached per (query, doc) within each stage's candidate",
        "  union; identical to scoring each window's list directly.",
        "- research_bench_v3 measured omnidb+CE top-50 on scidocs at -0.0197",
        "  (minilm) / -0.0238 (bge-base); this sweep brackets that window.",
        f"- scidocs full size ({results['_ndocs']} docs, {results['_nq']} "
        "queries); embeddings from bench/.cache3 (minilm, bge-base);",
        "  windows 25 and 100 only (fiqa probe also tested 50).",
    ]


def section_lines(results: dict) -> list[str]:
    lines = [
        SECTION,
        "",
        f"scidocs sweep of the rerank-window probe above ({results['_ndocs']} docs, "
        f"{results['_nq']} queries; reranker {rb.CROSS_ENCODER}, max_length=512): "
        "same wiring as fiqa - omnidb-hybrid retrieved fresh per window, bm25s "
        "single-leg (top-w is the exact prefix of top-100), CE pair-scores "
        f"cached across windows; windows {', '.join(str(w) for w in WINDOWS)}; "
        "p: paired Wilcoxon over queries (reranked vs base, same window).",
        "",
        "| first stage | embedder | w | stage recall@w | nDCG@10 base | "
        "nDCG@10 reranked | delta | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for stage, emb, w, c in rows_of(results):
        lines.append(f"| {stage} | {emb} | {w} | {c['stage_recall_at_w']:.4f} "
                     f"| {c['base_mean']:.4f} | {c['rerank_mean']:.4f} "
                     f"| {c['delta']:+.4f} | {c['p_wilcoxon']:.4f} |")
    lines += ["", "### Verdict", "", *verdict(results), "", "### Notes", ""]
    lines += notes(results)
    return lines


def append_section(results: dict) -> None:
    cur = OUT_MD.read_text(encoding="utf-8")
    if SECTION in cur:
        print(f"{SECTION} already in {OUT_MD}; append skipped", flush=True)
        return
    lines = section_lines(results)
    OUT_MD.write_text(cur + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"appended {SECTION} to {OUT_MD}", flush=True)


def main() -> None:
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME"):
        os.environ[var] = str(Path("D:/OMNIDB/omnidb-server/.hf-cache"))
    tmp = str(OUT_JSON.parent)
    os.environ["TMP"], os.environ["TEMP"] = tmp, tmp
    tempfile.tempdir = tmp
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs_all, queries = rb.load_beir(DATASET)
    rels = [rb.relevant_docs(rel) for (_q, _t, rel) in queries]
    print(f"{DATASET}: {len(docs_all)} docs, {len(queries)} queries; "
          f"reranker {rb.CROSS_ENCODER} on {device}", flush=True)

    results: dict = {}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    results["_ndocs"], results["_nq"] = len(docs_all), len(queries)
    save(results)

    ce = CrossEncoder(rb.CROSS_ENCODER, max_length=512, device=device)

    # ---- bm25 first stage: retrieve top-100 once, truncate per window ----
    if "bm25" not in results:
        t0 = time.perf_counter()
        results["bm25"] = {"ranked_top100": prw.retrieve_bm25(docs_all, queries, MAX_W)}
        print(f"bm25 top-{MAX_W} indexed+retrieved in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        save(results)
    bm_ranked = results["bm25"]["ranked_top100"]

    if "cells" not in results["bm25"]:
        t0 = time.perf_counter()
        scores = prw.ce_score_queries(ce, queries, docs_all, bm_ranked)
        print(f"bm25 CE scored {sum(len(m) for m in scores)} pairs in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        cells = {}
        for w in WINDOWS:
            rr = [prw.rerank(r[:w], s) for r, s in zip(bm_ranked, scores)]
            cells[str(w)] = prw.cell_stats(queries, bm_ranked, rr, rels, w)
        results["bm25"]["cells"] = cells
        save(results)
        for w in WINDOWS:
            c = cells[str(w)]
            print(f"  bm25 w={w}: base={c['base_mean']:.4f} "
                  f"rr={c['rerank_mean']:.4f} d={c['delta']:+.4f} "
                  f"p={c['p_wilcoxon']:.4f}", flush=True)

    # ---- omnidb-hybrid per embedder, retrieved fresh at each window ----
    for emb in EMBED_LIST:
        print(f"=== {emb} ===", flush=True)
        results.setdefault(emb, {})
        if "cells" in results[emb]:
            print("  cached, skipping", flush=True)
            continue
        vecs, qvecs = rb.get_embeddings(DATASET, emb, docs_all, queries)
        by_w = prw.retrieve_omnidb(docs_all, queries, vecs, qvecs, WINDOWS)
        union = [list(dict.fromkeys(a + b))
                 for a, b in zip(by_w[WINDOWS[0]], by_w[WINDOWS[1]])]
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
                  f"rr={c['rerank_mean']:.4f} d={c['delta']:+.4f} "
                  f"p={c['p_wilcoxon']:.4f}", flush=True)
        results[emb]["cells"] = cells
        save(results)

    append_section(results)


if __name__ == "__main__":
    main()
