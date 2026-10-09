"""Summarize held-out CE study and apply the locked 660-test Holm family."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from ce_study_runner import HERE, read_jsonl, apply_frozen_test_blends
from ce_study_stats import holm_adjust, paired_bootstrap_ci, sign_flip_mean_test


def expected_cells(config: dict) -> list[tuple[str, str, str, str, int]]:
    """Return the fixed family, including slots for unavailable/missing cells."""
    out=[]
    test_counts=config.get("test_query_counts",{})
    for corpus in config["corpora"]:
        for stage in config["stages"]:
            stage_embedders=(config["embedders"] if stage in ("hybrid","dense")
                             else ["shared"])
            for stage_emb in stage_embedders:
                for reranker in config["rerankers"]:
                    for w in config["windows"]:
                        out.append((corpus,stage_emb,stage,reranker,int(w),int(test_counts.get(corpus,0))))
    return out


def summarize(run_dir: Path, n_perm=100_000, n_boot=20_000):
    config=json.loads((run_dir/"run-config.json").read_text(encoding="utf-8"))
    frozen=json.loads((run_dir/"frozen-parameters.json").read_text(encoding="utf-8"))
    if frozen.get("selection_partition")!="dev":
        raise ValueError("frozen choices are not marked dev-selected")
    test_path=run_dir/"test-results.jsonl"
    if not test_path.exists(): apply_frozen_test_blends(run_dir)
    rows=list(read_jsonl(test_path))
    grouped=defaultdict(list)
    for r in rows:
        key=(r["corpus"],r["stage_embedder"],r["stage"],r["reranker"],int(r["window"]))
        grouped[key].append(r)
    expected=expected_cells(config)
    pvalues={}
    summaries=[]
    missing=[]
    for key6 in expected:
        key=key6[:5]
        cell=sorted(grouped.get(key,[]),key=lambda r:r["qid"])
        name="|".join(map(str,key))
        expected_test=key6[5] or None
        if not cell or (expected_test is not None and len(cell)!=expected_test):
            missing.append({"cell":name,"observed_test_rows":len(cell),"expected_test_rows":expected_test})
            pvalues[name]=1.0
            continue
        if len({r["qid"] for r in cell})!=len(cell):
            raise ValueError(f"duplicate test qid in {name}")
        deltas=np.asarray([r["blend_ndcg10"]-r["base_ndcg10"] for r in cell],float)
        perm=sign_flip_mean_test(deltas,n_permutations=n_perm,seed=20261008)
        ci=paired_bootstrap_ci(deltas,n_resamples=n_boot,seed=20261009)
        pvalues[name]=perm["p_two_sided"]
        summaries.append({
            "cell":name,"corpus":key[0],"stage_embedder":key[1],"stage":key[2],
            "reranker":key[3],"window":key[4],"n_test":len(cell),
            "alpha":cell[0]["frozen_alpha"],"rrf_k":cell[0]["frozen_rrf_k"],
            "base_mean":float(np.mean([r["base_ndcg10"] for r in cell])),
            "ce_only_mean":float(np.mean([r["ce_ndcg10"] for r in cell])),
            "blend_mean":float(np.mean([r["blend_ndcg10"] for r in cell])),
            "delta_mean":perm["mean_delta"],"permutation":perm,"bootstrap":ci,
            "wilcoxon_sensitivity":float(stats.wilcoxon(
                [r["blend_ndcg10"] for r in cell],[r["base_ndcg10"] for r in cell]).pvalue),
            "oracle_mean":float(np.mean([r["oracle_ndcg10"] for r in cell])),
            "judged_at10_base_mean":float(np.mean([r["judged_at10_base"] for r in cell])),
            "judged_at10_blend_mean":float(np.mean([r["judged_at10_blend"] for r in cell])),
            "unjudged_at10_base_mean":float(np.mean([r["unjudged_at10_base"] for r in cell])),
            "unjudged_at10_blend_mean":float(np.mean([r["unjudged_at10_blend"] for r in cell])),
            "condensed_base_mean":float(np.mean([r["condensed_base"] for r in cell])),
            "condensed_blend_mean":float(np.mean([r["condensed_blend"] for r in cell])),
            "condensed_delta_mean":float(np.mean([r["condensed_blend"]-r["condensed_base"]
                                                  for r in cell])),
        })
    adjusted=holm_adjust(pvalues)
    by_name={r["cell"]:r for r in summaries}
    for name,p in adjusted.items():
        if name in by_name: by_name[name]["holm_p_all_test_cells"]=p
    return {"metadata":{"run_id":run_dir.name,"family_size_predeclared":len(expected),
                         "family_size_observed":len(summaries),"missing_cells":missing,
                         "missing_cells_count_as_p1":True,
                         "permutation":"paired sign flip on held-out mean; exact zeros retained",
                         "holm":"across full predeclared family, including p=1 placeholders"},
            "family_a":summaries}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--permutations",type=int,default=100_000)
    ap.add_argument("--bootstrap",type=int,default=20_000)
    args=ap.parse_args()
    run_dir=HERE/"ce-study-runs"/args.run_id
    result=summarize(run_dir,args.permutations,args.bootstrap)
    out=run_dir/"test-summary.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("wrote",out,"family",result["metadata"]["family_size_predeclared"],
          "observed",result["metadata"]["family_size_observed"],
          "missing",len(result["metadata"]["missing_cells"]))

if __name__=="__main__": main()
