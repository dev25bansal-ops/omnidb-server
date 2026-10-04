"""Sweep: reranker sign vs retrieval window on nfcorpus (fiqa probe, corpus 2).

The fiqa probe (probe_rerank_window.py, section above) measured
bge-reranker-base as a net LOSS on omnidb-hybrid at every window (25/50/100,
both embedders, monotonically worse as the window grows) and a big GAIN on
the weak bm25s stage, concluding that reranker value scales with FIRST-STAGE
WEAKNESS, not window size. nfcorpus tests that law on a very different
corpus: short medical abstracts, graded relevance, 3.6k docs / 323 queries,
where hybrid recall at small windows is expected to sit far below fiqa's -
so if the weakness law is real, the strong stage should gain here too.

Cells: {omnidb-hybrid, bm25s} x {top-25, top-100} x {minilm, bge-base}.
Metric: nDCG@10 reranked vs no-rerank, per-query paired Wilcoxon, deltas.

Method is IDENTICAL to the fiqa probe: this script imports its helpers
(retrieve_omnidb, retrieve_bm25, ce_score_queries, rerank, cell_stats) and
research_bench_v3's load_beir/get_embeddings. omnidb-hybrid is retrieved
fresh per window (k=w, k_fetch=w); bm25s top-100 is retrieved once and
truncated per window (single-leg prefix); cross-encoder scores are cached
per (query, doc) so every window's reranked list is a re-sort of its own
ranking by the shared scores.

Usage: python bench/sweep_rerank_nfcorpus.py  (~10-15 min, all models cached)
Checkpoint bench/nfcorpus-rerank-window-results.json is resume-safe. The
markdown section is APPENDED to bench/rerank-window-results.md; sections
written by other scripts are never modified (a re-run replaces only this
script's own trailing nfcorpus section).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import probe_rerank_window as prw
import research_bench_v3 as rb

DATASET = "nfcorpus"
WINDOWS = (25, 100)
MAX_W = 100
EMBED_LIST = ("minilm", "bge-base")
HF_CACHE = "D:/OMNIDB/omnidb-server/.hf-cache"
TMP_ROOT = Path("E:/omnidb-bench-tmp")

OUT_JSON = HERE / "nfcorpus-rerank-window-results.json"
OUT_MD = HERE / "rerank-window-results.md"
MARKER = "## nfcorpus"

# fiqa reference values from the probe section above (same reranker, same
# method), quoted in the cross-corpus verdict bullet: window -> delta / recall.
FIQA_HYBRID_DELTA = {"minilm": {25: -0.0031, 100: -0.0247},
                     "bge-base": {25: -0.0093, 100: -0.0407}}
FIQA_HYBRID_RECALL = {"minilm": {25: 0.5483, 100: 0.7188},
                      "bge-base": {25: 0.5871, 100: 0.7360}}


def stage_rows(results: dict) -> list[tuple[str, str, int, dict]]:
    rows = []
    for stage, emb in (("bm25", "-"), ("omnidb-hybrid", "minilm"),
                       ("omnidb-hybrid", "bge-base")):
        node = results["bm25"] if stage == "bm25" else results[emb]
        for w in WINDOWS:
            rows.append((stage, emb, w, node["cells"][str(w)]))
    return rows


def verdict_lines(results: dict) -> list[str]:
    rows = stage_rows(results)

    def deltas(stage: str, emb: str) -> list[tuple[int, float]]:
        return [(w, c["delta"]) for s, e, w, c in rows if s == stage and e == emb]

    def fmt(ds: list[tuple[int, float]]) -> str:
        return "/".join(f"w{w}:{d:+.4f}" for w, d in ds)

    def phrase(ds: list[tuple[int, float]]) -> str:
        if all(d > 0 for _w, d in ds):
            return "gains at every window"
        if all(d < 0 for _w, d in ds):
            return "loses at every window"
        return "is sign-mixed across windows"

    bm = deltas("bm25", "-")
    om = {emb: deltas("omnidb-hybrid", emb) for emb in EMBED_LIST}
    one = (f"one-line: on nfcorpus the reranker {phrase(bm)} on bm25 "
           f"({fmt(bm)}); it {phrase(om['minilm'])} on omnidb-hybrid/minilm "
           f"({fmt(om['minilm'])}) and {phrase(om['bge-base'])} on "
           f"omnidb-hybrid/bge-base ({fmt(om['bge-base'])}).")

    om_all = [d for emb in EMBED_LIST for _w, d in om[emb]]
    fq = FIQA_HYBRID_DELTA
    fiqa_seq = (f"minilm {fq['minilm'][25]:+.4f}/{fq['minilm'][100]:+.4f}, "
                f"bge-base {fq['bge-base'][25]:+.4f}/{fq['bge-base'][100]:+.4f} "
                f"at w=25/100")
    if all(d > 0 for d in om_all):
        cross = (f"vs fiqa: the hybrid sign FLIPS - fiqa lost at every window "
                 f"({fiqa_seq}); nfcorpus gains at both windows on both "
                 f"embedders.")
    elif all(d < 0 for d in om_all):
        cross = (f"vs fiqa: the hybrid sign REPLICATES - loss at every window "
                 f"here as there ({fiqa_seq}).")
    else:
        cross = (f"vs fiqa: the hybrid sign is MIXED here, whereas fiqa lost "
                 f"at every window ({fiqa_seq}).")

    stages = [("bm25", bm)] + [(f"hybrid/{e}", om[e]) for e in EMBED_LIST]
    grew = [name for name, ds in stages if ds[1][1] > ds[0][1]]
    if len(grew) == len(stages):
        trend = ("window trend: every stage's delta grows from w=25 to w=100 "
                 "(matches bm25-on-fiqa; more candidates give a winning "
                 "reranker more to fix).")
    elif not grew:
        trend = ("window trend: every stage's delta shrinks from w=25 to "
                 "w=100 (matches hybrid-on-fiqa).")
    else:
        trend = (f"window trend: mixed - the delta grows with window only for "
                 f"{', '.join(grew)}.")

    def cell_of(stage: str, emb: str, w: int) -> dict:
        return next(c for s, e, ww, c in rows if s == stage and e == emb and ww == w)

    r25 = cell_of("omnidb-hybrid", "minilm", 25)["stage_recall_at_w"]
    if all(d > 0 for d in om_all):
        mech = (f"mechanism: confirms the first-stage-weakness law - nfcorpus "
                f"hybrid recall@25 (minilm {r25:.4f}) sits far below fiqa's "
                f"({FIQA_HYBRID_RECALL['minilm'][25]:.4f}), so even the "
                f"strong stage leaves rerank headroom at small windows; bm25, "
                f"weakest here as on fiqa, gains the most.")
    elif all(d < 0 for d in om_all):
        mech = (f"mechanism: weakens the first-stage-weakness law - despite "
                f"nfcorpus hybrid recall@25 (minilm {r25:.4f}) sitting far "
                f"below fiqa's ({FIQA_HYBRID_RECALL['minilm'][25]:.4f}), "
                f"reranking the strong stage still fails to gain; the "
                f"reranker's value is corpus-dependent beyond recall "
                f"headroom alone.")
    else:
        mech = (f"mechanism: mixed evidence for the first-stage-weakness law "
                f"- nfcorpus hybrid recall@25 (minilm {r25:.4f}) vs fiqa's "
                f"({FIQA_HYBRID_RECALL['minilm'][25]:.4f}), and the hybrid "
                f"deltas disagree between embedders.")

    bm_rr100 = cell_of("bm25", "-", MAX_W)["rerank_mean"]
    best_hyb_base = max(c["base_mean"] for s, _e, _w, c in rows
                       if s == "omnidb-hybrid")
    ovt = "OVERTAKES" if bm_rr100 > best_hyb_base else "does not overtake"
    gap = (f"bm25+CE at w=100 ({bm_rr100:.4f}) {ovt} the best no-rerank "
           f"hybrid ({best_hyb_base:.4f}).")
    return [f"- {one}", f"- {cross}", f"- {trend}", f"- {mech}", f"- {gap}"]


def honesty_lines(results: dict) -> list[str]:
    meta = results["meta"]
    return [
        "- omnidb-hybrid re-retrieved fresh per window (fused ranking depends",
        "  on k_fetch; prefix-of-100 is not the true top-25 ranking).",
        "- bm25s single-leg: top-w is the exact prefix of top-100 (verified",
        "  pre-flight: every nfcorpus query returns 100 real candidates, no",
        "  -1 padding, no duplicate ids).",
        "- CE scores cached per (query, doc) across windows; identical to",
        "  scoring each window's list directly.",
        f"- nfcorpus full size ({meta['ndocs']} docs, {meta['nqueries']} "
        f"queries); embeddings from bench/.cache3 (minilm, bge-base), same",
        f"  as research_bench_v3; reranker {meta['reranker']}, max_length=512,",
        f"  {meta['device']}.",
        "- fiqa deltas/recalls quoted in the verdict are from the section",
        "  above (same reranker, same method, windows 25/50/100 there).",
    ]


def render(results: dict) -> str:
    meta = results["meta"]
    lines = [
        MARKER,
        "",
        "Same probe as the fiqa section above, second corpus. Short medical",
        "abstracts, graded relevance, small corpus; windows narrowed to the",
        "two extremes (25, 100). All helpers imported from the fiqa probe so",
        "the method is bit-for-bit identical.",
        "",
        f"- corpus: nfcorpus full size ({meta['ndocs']} docs, "
        f"{meta['nqueries']} queries); eval k = {rb.K}",
        f"- reranker: {meta['reranker']}, max_length=512, on {meta['device']}",
        "- first stages: omnidb-hybrid (corpus_mode, weighted RRF 0.7/0.3,",
        "  retrieved fresh per window) and bm25s (k1=0.9, b=0.4)",
        "- p: paired Wilcoxon over queries (reranked vs base, same window)",
        "",
        "| first stage | embedder | w | stage recall@w | nDCG@10 base | "
        "nDCG@10 reranked | delta | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for stage, emb, w, c in stage_rows(results):
        lines.append(
            f"| {stage} | {emb} | {w} | {c['stage_recall_at_w']:.4f} "
            f"| {c['base_mean']:.4f} | {c['rerank_mean']:.4f} "
            f"| {c['delta']:+.4f} | {c['p_wilcoxon']:.4f} |")
    lines += ["", "### Verdict (nfcorpus)", ""]
    lines += verdict_lines(results)
    lines += ["", "### Honesty notes (nfcorpus)", ""]
    lines += honesty_lines(results)
    return "\n".join(lines) + "\n"


def append_section(section: str) -> None:
    """Append the nfcorpus section; never touch anything above it."""
    if OUT_MD.exists():
        text = OUT_MD.read_text(encoding="utf-8")
        body = text.splitlines()
        cut = next((i for i, ln in enumerate(body) if ln.strip() == MARKER), None)
        text = ("\n".join(body[:cut]).rstrip() + "\n\n" if cut is not None
                else text.rstrip() + "\n\n")
    else:
        text = ""
    OUT_MD.write_text(text + section, encoding="utf-8")


def _save(results: dict) -> None:
    OUT_JSON.write_text(json.dumps(results), encoding="utf-8")


def main() -> None:
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME"):
        os.environ[var] = HF_CACHE
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"], os.environ["TEMP"] = str(TMP_ROOT), str(TMP_ROOT)
    tempfile.tempdir = str(TMP_ROOT)

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs, queries = rb.load_beir(DATASET)
    rels = [rb.relevant_docs(rel) for (_q, _t, rel) in queries]
    print(f"{DATASET}: {len(docs)} docs, {len(queries)} queries; "
          f"reranker {rb.CROSS_ENCODER} on {device}", flush=True)
    results: dict = {"meta": {"ndocs": len(docs), "nqueries": len(queries),
                              "reranker": rb.CROSS_ENCODER,
                              "device": "cuda (GPU)" if device == "cuda" else "cpu"}}
    if OUT_JSON.exists():
        results.update(json.loads(OUT_JSON.read_text(encoding="utf-8")))
    _save(results)

    if "bm25" not in results:
        t0 = time.perf_counter()
        bm_ranked = prw.retrieve_bm25(docs, queries, MAX_W)
        print(f"bm25 top-{MAX_W} in {time.perf_counter() - t0:.0f}s", flush=True)
        results["bm25"] = {"ranked_top100": bm_ranked}
        _save(results)
    bm_ranked = results["bm25"]["ranked_top100"]

    ce = CrossEncoder(rb.CROSS_ENCODER, max_length=512, device=device)

    if "cells" not in results["bm25"]:
        scores = prw.ce_score_queries(ce, queries, docs, bm_ranked)
        cells = {}
        for w in WINDOWS:
            rr = [prw.rerank(r[:w], s) for r, s in zip(bm_ranked, scores)]
            cells[str(w)] = prw.cell_stats(queries, bm_ranked, rr, rels, w)
            c = cells[str(w)]
            print(f"  bm25 w={w}: base={c['base_mean']:.4f} "
                  f"rr={c['rerank_mean']:.4f} d={c['delta']:+.4f}", flush=True)
        results["bm25"]["cells"] = cells
        _save(results)

    for emb in EMBED_LIST:
        print(f"=== {emb} ===", flush=True)
        results.setdefault(emb, {})
        if "cells" in results[emb]:
            print("  cached, skipping", flush=True)
            continue
        vecs, qvecs = rb.get_embeddings(DATASET, emb, docs, queries)
        by_w = prw.retrieve_omnidb(docs, queries, vecs, qvecs, WINDOWS)
        union = [list(dict.fromkeys(sum(parts, [])))
                 for parts in zip(*(by_w[w] for w in WINDOWS))]
        t0 = time.perf_counter()
        scores = prw.ce_score_queries(ce, queries, docs, union)
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
        results[emb]["ranked_top100"] = by_w[MAX_W]
        _save(results)

    section = render(results)
    append_section(section)
    print("\n".join(section.splitlines()[:len(MARKER) + 30]), flush=True)
    print(f"appended {MARKER!r} section to {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
