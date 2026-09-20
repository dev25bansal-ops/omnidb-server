"""Resume the killed v2 recompute: fill only missing omnidb recall cells.

Reads research-v2-results.json (merged with the pre-run backup), recomputes
any omnidb cell that lacks per_query_recall, re-renders the .md, and writes
a summary of old-vs-new nDCG stability for the untouched datasets.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_bench_v2 as rb2

cache = HERE / ".cache3" / "hf"
cache.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_DISABLE_XET"] = "1"
for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
            "SENTENCE_TRANSFORMERS_HOME", "XDG_CACHE_HOME"):
    os.environ[var] = str(cache)

results = json.loads(rb2.OUT_JSON.read_text(encoding="utf-8"))

todo = [(ds, emb) for ds, dsc in results.items() for emb, cell in dsc.items()
        if not (cell.get("omnidb", {}).get("per_query_recall"))]
print(f"cells to compute: {len(todo)} -> {todo}", flush=True)

for ds, emb in todo:
    spec = rb2.EMBEDDERS[emb]
    docs, queries = rb2.load_beir(rb2.DATASETS[ds])
    texts = [t for _d, t in docs]
    vecs = rb2.embed(texts, spec, is_query=False)
    qvecs = rb2.embed([q for _q, q, _r in queries], spec, is_query=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"rb2-resume-{ds}-{emb}-"))
    cell = rb2.bench_omnidb(docs, queries, vecs, qvecs, tmp / "omnidb")
    results[ds][emb]["omnidb"] = cell
    print(f"{ds}/{emb}: nDCG={statistics.mean(cell['per_query']):.4f} "
          f"recall={statistics.mean(cell['per_query_recall']):.4f} "
          f"(n={len(cell['per_query'])})", flush=True)
    rb2.OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

# stability check: did the untouched cells' nDCG change vs the pre-run backup?
bak = json.loads((HERE / "research-v2-results.json.bak").read_text(encoding="utf-8"))
for ds in sorted(set(results) & set(bak)):
    for emb in sorted(set(results[ds]) & set(bak[ds])):
        new_pq = results[ds][emb].get("omnidb", {}).get("per_query", [])
        old_pq = bak[ds][emb].get("omnidb", {}).get("per_query", [])
        if not new_pq or not old_pq:
            continue
        dn = statistics.mean(new_pq) - statistics.mean(old_pq)
        flag = "STABLE" if abs(dn) < 5e-4 else "DRIFTED"
        print(f"stability {ds}/{emb}: old={statistics.mean(old_pq):.4f} "
              f"new={statistics.mean(new_pq):.4f} delta={dn:+.4f} {flag}",
              flush=True)

rb2.render(results)
print("render done", flush=True)
