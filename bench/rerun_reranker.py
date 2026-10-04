"""Re-score the stored omnidb top-50 with a (possibly new) cross-encoder.

PROVENANCE / SAFETY (fixed after a silent-clobber incident):
  - By default this writes a NEW model-tagged file and NEVER overwrites the
    main results JSON, so swapping rerankers cannot destroy the original run.
  - Retrieval is rebuilt once and cached as ``omni_top50`` in every cell.
    Re-running (or swapping to yet another reranker) then reuses those lists
    and skips the expensive store/retrieval step entirely -- only the
    cross-encoder passes.
  - The reranker id, the base top-50, and a swap timestamp are stamped per
    cell so the +CE column's provenance is auditable, not inferred.

Usage:
  python bench/rerun_reranker.py [NEW_RERANKER_ID] [--inplace] [--from FILE]

  NEW_RERANKER_ID  default: value of CROSS_ENCODER in research_bench_v3.
  --from FILE      read the source results JSON (default: research-v3-results.json).
  --inplace        overwrite the source JSON (after a timestamped .bak);
                   otherwise output is research-v3-results.<slug>.json.

No retrieval *engines* are re-evaluated; only the omnidb-hybrid top-50 is
rescored. Embeddings stay cached. The other columns (hybrid, bm25s, pyserini,
dense-only, qdrant) are carried through unchanged.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# The models were cached under bench/.cache3/hf. If HF_HOME points at a drive
# that no longer exists (it has pointed at a removed F:\), fall back to the
# bench-local cache so the run does not die on an offline load.
_hf_home = os.environ.get("HF_HOME", "")
_local_hf = HERE / ".cache3" / "hf"
if _local_hf.is_dir() and (not _hf_home or not Path(_hf_home).exists()):
    os.environ["HF_HOME"] = str(_local_hf)

from research_bench_v3 import (
    CROSS_ENCODER,
    EMBEDDERS,
    RERANK_CANDS,
    K,
    cross_encode_rerank,
    get_embeddings,
    load_beir,
    ndcg_at_k,
)

MAIN_JSON = HERE / "research-v3-results.json"
EMBEDDER_KEYS = tuple(EMBEDDERS)  # ("minilm", "bge-small", "e5-small", "bge-base")


def slug(reranker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", reranker.split("/")[-1]).strip("-").lower()


def per_query(ranked_lists, queries):
    return [ndcg_at_k(r, rel, K)
            for (_q, _t, rel), r in zip(queries, ranked_lists)]


def rebuild_top50(ds, emb, cell, cache):
    """Return the omnidb-hybrid top-RERANK_CANDS lists, rebuilding retrieval
    only if the cell does not already carry a cached ``omni_top50``."""
    if "omni_top50" in cell:
        return cell["omni_top50"], False
    from research_bench_v3 import _make_embedder

    from omnidb_server.config import Config
    from omnidb_server.server import build_stack
    from omnidb_server.vector_index import BruteForceIndex

    docs, queries = load_beir(ds)
    texts = [t for _d, t in docs]
    if emb not in cache:
        cache[emb] = get_embeddings(ds, emb, docs, queries)
    vecs, qvecs = cache[emb]
    cfg = Config.from_env(pathlib.Path(tempfile.mkdtemp()) / "rr")
    cfg.corpus_mode = True
    cfg.dense_weight, cfg.sparse_weight = 0.7, 0.3
    store = build_stack(cfg)
    store.remember_many([{"content": t} for t in texts])
    shared = _make_embedder(vecs, qvecs, texts)
    ids = [int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories ORDER BY id")]
    store._mark_dirty()
    store.index = BruteForceIndex(pathlib.Path(tempfile.mkdtemp()) / "v.npz",
                                  shared.dim)
    store.index.add(ids, vecs)
    store.embedder = shared
    ranked = []
    for (_qid, qtext, _rel) in queries:
        hits = store.recall(qtext, k=RERANK_CANDS)
        ranked.append([docs[h["id"] - 1][0] for h in hits])
    store.close()
    return ranked, True


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reranker = args[0] if args else CROSS_ENCODER
    inplace = "--inplace" in sys.argv
    src = MAIN_JSON
    if "--from" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--from") + 1])
    out = src if inplace else src.with_name(f"{src.stem}.{slug(reranker)}.json")

    if inplace and src.exists():
        bak = src.with_suffix(src.suffix + f".pre-swap-{int(time.time())}.bak")
        bak.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[inplace] backed up {src.name} -> {bak.name}")

    data = json.loads(src.read_text(encoding="utf-8"))
    # load_beir is per-dataset; cache docs/queries once
    beir: dict[str, tuple] = {}
    emb_cache: dict[str, dict] = {}  # per-dataset emb->vecs, cleared on dataset change
    n_cells = n_rebuilt = 0
    import research_bench_v3 as rb

    for ds, cell_map in data.items():
        if ds.startswith("_") or not isinstance(cell_map, dict):
            continue
        if ds not in beir:
            beir[ds] = load_beir(ds)
            emb_cache.clear()
        docs, queries = beir[ds]
        for emb in EMBEDDER_KEYS:
            c = cell_map.get(emb)
            if not isinstance(c, dict) or "omnidb" not in c or "omnidb-ce" not in c:
                continue
            prev_tag = c["omnidb-ce"].get("reranker", "(untagged)")
            ranked, rebuilt = rebuild_top50(ds, emb, c, emb_cache)
            if rebuilt:
                c["omni_top50"] = ranked
                n_rebuilt += 1
            else:
                ranked = c["omni_top50"]
            rb.CROSS_ENCODER = reranker  # cross_encode_rerank reads the module global
            t0 = time.perf_counter()
            rescored = cross_encode_rerank(queries, docs, ranked)
            dt = time.perf_counter() - t0
            pq = per_query(rescored, queries)
            c["omnidb-ce"] = {
                "per_query": pq,
                "rerank_s": round(dt, 1),
                "reranker": reranker,
                "swapped_at": round(time.time()),
            }
            n_cells += 1
            print(f"  {ds}/{emb}: +CE={sum(pq)/len(pq):.4f} "
                  f"(was {prev_tag} @ {c.get('omnidb-ce', {}).get('rerank_s', '?')}s, "
                  f"now {reranker}, {'re-retrieved' if rebuilt else 'cached top50'})",
                  flush=True)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")  # per-ds checkpoint

    print(f"rescored {n_cells} cells with {reranker} "
          f"({n_rebuilt} re-ran retrieval, {n_cells - n_rebuilt} reused cached omni_top50)")
    print(f"OUTPUT: {out.name}")
    if not inplace:
        print(f"(main {src.name} untouched; pass --inplace to overwrite)")


if __name__ == "__main__":
    main()
