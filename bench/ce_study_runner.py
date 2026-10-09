"""Versioned, resumable CE study runner. Never writes to research-v3 artifacts.

Phases:
  pools     retrieve and save candidate pools / query text, without qrels
  dev       score development queries only and write frozen interpolation choices
  test      score held-out queries using frozen choices only

Each phase is independently resumable. Candidate IDs and raw model scores are
stored in JSONL; summaries are derived by separate analysis code.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# HF_HOME has pointed at a removed F:\ drive on this machine, which makes any
# library that touches it crash with WinError 3 at import/first-use. Point every
# HF cache variable at the bench-local cache before transformers is imported.
_LOCAL_HF = HERE / ".cache3" / "hf"
_LOCAL_HF.mkdir(parents=True, exist_ok=True)
for _var in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    _current = os.environ.get(_var)
    if not _current or not Path(_current).anchor or not Path(_current).drive or \
            not Path(Path(_current).anchor).exists() or not Path(_current).exists():
        os.environ[_var] = str(_LOCAL_HF)

from ce_study_split import partition_query_items, validate_manifest
from _stats import ndcg_at_k

RUNS = HERE / "ce-study-runs"
CORPORA = ("nfcorpus", "scifact", "scidocs", "fiqa")
EMBEDDERS = ("minilm", "bge-small", "e5-small", "bge-base")
WINDOWS = (25, 50, 100)
RERANKERS = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "ncbi/MedCPT-Cross-Encoder",
)
RERANKER_REVISIONS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
    "BAAI/bge-reranker-large": "55611d7bca2a7133960a6d3b71e083071bbfc312",
    "mixedbread-ai/mxbai-rerank-base-v2": "3ea9d4dffa7d12a4f366be8e275c349de9fc9865",
    "ncbi/MedCPT-Cross-Encoder": "71caf65d4927987813984f54c284405a13fcca49",
}
STAGES = ("hybrid", "dense", "fts5_sparse", "bm25s", "pyserini")
WINDOWS = (25, 50, 100)
ALPHAS = tuple(i / 10 for i in range(11))
RRF_KS = (20, 60, 100)
K = 10
_QRELS_CACHE = {}


def _configure_java() -> None:
    """Point JAVA_HOME at the portable JDK 21 the Pyserini path requires.

    research_bench_v3.main() does this; the study runner stages Pyserini itself,
    so it needs the same setup or jnius raises "Unable to find JAVA_HOME".
    """
    jdk = Path("D:/OMNIDB/tools/jdk-21.0.5+11")
    if jdk.exists():
        os.environ["JAVA_HOME"] = str(jdk)
        os.environ["PATH"] = str(jdk / "bin") + os.pathsep + os.environ.get("PATH", "")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def append_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path):
    if not path.exists():
        return
    # A process can stop mid-write; only a non-newline-terminated final fragment
    # is recoverable. Any malformed complete record is a hard corruption error.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last = raw.rfind(b"\n")
        keep = raw[:last + 1] if last >= 0 else b""
        path.write_bytes(keep)
        raw = keep
    for line_no, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt JSONL {path}:{line_no}") from exc


def load_reranker(model_id: str, device: str, cache_dir: Path, batch_size: int):
    from huggingface_hub import scan_cache_dir
    from sentence_transformers import CrossEncoder
    for var in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ[var] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    cached = {repo.repo_id: {rev.commit_hash for rev in repo.revisions}
              for repo in scan_cache_dir(str(cache_dir)).repos}
    if RERANKER_REVISIONS[model_id] not in cached.get(model_id, set()):
        raise FileNotFoundError(
            f"pinned revision {RERANKER_REVISIONS[model_id]} for {model_id} "
            f"is not fully present in cache {cache_dir}; refusing fallback/network load")
    max_length = 512
    model = CrossEncoder(model_id, revision=RERANKER_REVISIONS[model_id],
                         max_length=max_length, device=device,
                         local_files_only=True)
    model._study_batch_size = batch_size
    return model, max_length


def score_pairs(model, query: str, candidate_ids: list[str], text_by_id: dict[str, str]) -> list[float]:
    pairs = [(query, text_by_id[d]) for d in candidate_ids]
    scores = np.asarray(model.predict(pairs, batch_size=model._study_batch_size,
                                     show_progress_bar=False), dtype=np.float64).reshape(-1)
    if len(scores) != len(candidate_ids) or not np.all(np.isfinite(scores)):
        raise ValueError("reranker returned missing/non-finite scores")
    return scores.tolist()


def rank_rrf(base_ids: list[str], ce_scores: list[float], alpha: float, k_rrf: int) -> list[str]:
    if len(base_ids) != len(ce_scores) or len(set(base_ids)) != len(base_ids):
        raise ValueError("candidate IDs/scores not aligned or IDs duplicated")
    if not 0.0 <= alpha <= 1.0 or k_rrf <= 0:
        raise ValueError("alpha must be in [0,1] and k_rrf positive")
    ce_order = sorted(range(len(ce_scores)), key=lambda i: (-ce_scores[i], i))
    ce_rank = {base_ids[idx]: rank for rank, idx in enumerate(ce_order)}
    fused = {doc: ((1.0 - alpha) / (k_rrf + rank + 1) +
                   alpha / (k_rrf + ce_rank[doc] + 1))
             for rank, doc in enumerate(base_ids)}
    return sorted(base_ids, key=lambda d: (-fused[d], d))


def qrels_metrics(ranking: list[str], rel: dict[str, int], k: int = K) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    judged = sum(d in rel for d in ranking[:k])
    return {"ndcg10": float(ndcg_at_k(ranking, rel, k)),
            "judged_at10": judged / float(k),
            "unjudged_at10": (k - judged) / float(k)}


def oracle_ndcg(pool: list[str], rel: dict[str, int], k: int = K) -> float:
    ideal = sorted(pool, key=lambda d: (-rel.get(d, 0), d))
    return float(ndcg_at_k(ideal, rel, k))


def query_key(r):
    return (r["corpus"], r["stage_embedder"], r["stage"], r["window"], r["qid"])


def _stage_embedder(stage: str, embedder: str) -> str:
    return embedder if stage in ("hybrid", "dense") else "shared"


def _completed_keys(path: Path, fields: tuple[str, ...]) -> set[tuple]:
    return {tuple(r[f] for f in fields) for r in read_jsonl(path) or ()}


def prepare_run(args, split_manifest) -> Path:
    output = RUNS / args.run_id
    protocol_path = HERE / "ce-study-protocol.md"
    data_manifest_path = HERE / "ce-study-data-manifest.json"
    if not protocol_path.exists() or not data_manifest_path.exists():
        raise FileNotFoundError("protocol or data manifest missing")
    import subprocess
    try:
        code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=HERE.parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        code_sha = "unknown"
    config = {
        "run_id": args.run_id,
        "protocol_version": split_manifest["protocol_version"],
        "split_sha256": sha256_file(Path(args.split)),
        "protocol_sha256": sha256_file(protocol_path),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "code_sha": code_sha,
        "rerankers": {m: RERANKER_REVISIONS[m] for m in args.rerankers},
        "corpora": list(args.corpora), "stages": list(args.stages),
        "embedders": list(args.embedders), "windows": list(args.windows),
        "device": args.device, "batch_size": args.batch_size,
        "created_unix": time.time(),
    }
    config_path = output / "run-config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in existing.items() if k != "created_unix"}
        target = {k: v for k, v in config.items() if k != "created_unix"}
        if comparable != target:
            raise ValueError("resume refused: immutable run config mismatch")
        if not args.resume:
            raise FileExistsError(f"{output} exists; pass --resume")
    else:
        if args.resume and output.exists() and any(output.iterdir()):
            raise ValueError("resume refused: nonempty output without config")
        atomic_json(config_path, config)
    return output


def generate_pools(args, output, split_manifest):
    _configure_java()
    import research_bench_v3 as rb
    pool_path = output / "candidate-pools.jsonl"
    keys = _completed_keys(pool_path, ("corpus", "stage_embedder", "stage", "window", "qid"))
    for corpus in args.corpora:
        split = split_manifest["corpora"][corpus]
        all_ids = split["dev_ids"] + split["test_ids"]
        docs, queries = rb.load_beir_documents_and_queries(corpus, all_ids)
        parts = {"dev_ids": split["dev_ids"], "test_ids": split["test_ids"]}
        part_by_qid = {q: "dev" for q in parts["dev_ids"]}
        part_by_qid.update({q: "test" for q in parts["test_ids"]})
        text_by_id = dict(docs)
        q_index = {str(qid): i for i, (qid, _, _) in enumerate(queries)}
        shared_stages = {"fts5_sparse", "bm25s", "pyserini"}
        for stage in args.stages:
            stage_embedders = (args.embedders[:1] if stage in shared_stages else args.embedders)
            for stage_emb in stage_embedders:
                emb = stage_emb
                vecs = qvecs = None
                if stage in ("hybrid", "dense", "fts5_sparse"):
                    vecs, qvecs = rb.get_embeddings(corpus, emb, docs, queries)
                home = output / "work" / corpus / stage_emb / stage
                rankings = {}
                if stage == "hybrid":
                    for w in args.windows:
                        rankings[w], _ = rb.retrieve_omnidb(
                            docs, queries, vecs, qvecs, home / f"w{w}", fetch=w, limit=w)
                elif stage == "dense":
                    for w in args.windows:
                        rankings[w] = rb.retrieve_omnidb_dense_only(
                            docs, queries, vecs, qvecs, home / f"w{w}", limit=w)
                elif stage == "fts5_sparse":
                    for w in args.windows:
                        rankings[w] = rb.retrieve_omnidb_sparse_only(
                            docs, queries, vecs, qvecs, home / f"w{w}", limit=w)
                elif stage == "bm25s":
                    for w in args.windows:
                        rankings[w] = rb.retrieve_bm25s(
                            docs, queries, None, None, home / f"w{w}", k=w)
                elif stage == "pyserini":
                    for w in args.windows:
                        rankings[w] = rb.retrieve_pyserini(
                            docs, queries, None, None, home / f"w{w}", k=w)
                else:
                    raise ValueError(f"unknown stage {stage}")
                records = []
                for qid, qtext, _rel in queries:
                    for w in args.windows:
                        key = (corpus, stage_emb, stage, w, str(qid))
                        if key in keys:
                            continue
                        ranked = list(rankings[w][q_index[str(qid)]])
                        records.append({
                            "corpus": corpus, "stage_embedder": stage_emb,
                            "stage": stage, "window": w, "qid": str(qid),
                            "partition": part_by_qid[str(qid)], "query": qtext,
                            "pool_ids": ranked, "candidate_count": len(ranked),
                            "requested_window": w,
                        })
                if records:
                    append_records(pool_path, records)
                    keys.update(query_key(r) for r in records)
                print(f"pools {corpus}/{stage_emb}/{stage}: added {len(records)}", flush=True)


def score_partition(args, output, partition: str):
    pools_path = output / "candidate-pools.jsonl"
    if not pools_path.exists():
        raise FileNotFoundError("candidate pools missing; run pools phase")
    score_path = output / f"{partition}-scores.jsonl"
    completed = _completed_keys(score_path, ("corpus", "stage_embedder", "stage", "window", "qid", "reranker"))
    model_cache = HERE / ".cache3" / "hf"
    split=json.loads(Path(args.split).read_text(encoding="utf-8"))["corpora"]
    # Stream one corpus at a time: the complete pool matrix can be several GB.
    for reranker in args.rerankers:
        model, max_length = load_reranker(reranker, args.device, model_cache, args.batch_size)
        try:
            for corpus in args.corpora:
                allowed=set(split[corpus]["dev_ids"] if partition=="dev" else split[corpus]["test_ids"])
                pool_rows=(r for r in read_jsonl(pools_path)
                           if r["corpus"]==corpus and r["partition"]==partition
                           and (r["corpus"],r["stage_embedder"],r["stage"],r["window"],r["qid"],reranker) not in completed)
                # Load text only; qrels are parsed only for the active split IDs.
                import research_bench_v3 as rb
                docs, _queries = rb.load_beir_documents_and_queries(corpus, allowed)
                text_by_id=dict(docs)
                qrels_path=HERE/".cache3"/"beir"/f"{corpus}-qrels-test.tsv"
                records=[]; count=0
                for row in pool_rows:
                    if row["qid"] not in allowed:
                        raise ValueError(f"pool query outside {partition}: {corpus}/{row['qid']}")
                    pool=row["pool_ids"]
                    # A stage may legitimately return an empty pool (a lexical
                    # query whose terms match nothing). That is a real base
                    # ranking of length 0, not a harness failure: record it with
                    # zero-value metrics and no CE pairs rather than crashing.
                    scores = score_pairs(model,row["query"],pool,text_by_id) if pool else []
                    qrels=_qrels_for_query(qrels_path,row["qid"],allowed)
                    ce_order=[pool[j] for j in sorted(range(len(scores)),key=lambda j:(-scores[j],j))]
                    records.append({**row,"reranker":reranker,"revision":RERANKER_REVISIONS[reranker],
                        "max_length":max_length,"ce_scores":scores,"ce_ranking":ce_order,
                        "base_ndcg10":qrels_metrics(pool,qrels)["ndcg10"],
                        "ce_ndcg10":qrels_metrics(ce_order,qrels)["ndcg10"],
                        "oracle_ndcg10":oracle_ndcg(pool,qrels),
                        "judged_at10_base":qrels_metrics(pool,qrels)["judged_at10"],
                        "judged_at10_ce":qrels_metrics(ce_order,qrels)["judged_at10"],
                        "unjudged_at10_base":qrels_metrics(pool,qrels)["unjudged_at10"],
                        "unjudged_at10_ce":qrels_metrics(ce_order,qrels)["unjudged_at10"]})
                    count+=1
                    if len(records)>=100:
                        append_records(score_path,records); records.clear()
                    if count%1000==0:
                        print(f"{partition} {reranker}/{corpus}: {count} records",flush=True)
                if records: append_records(score_path,records)
                print(f"scored {partition}: {reranker}/{corpus} rows={count}",flush=True)
                del text_by_id, docs
        finally:
            del model
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception:
                pass

def freeze_development_parameters(output: Path, args) -> None:
    """Select alpha/k by development outcomes only; freeze choices immutably."""
    from collections import defaultdict
    dev_path=output / "dev-scores.jsonl"
    if not dev_path.exists():
        raise FileNotFoundError("dev-scores.jsonl missing; run dev phase first")
    pools={(r["corpus"],r["stage_embedder"],r["stage"],r["window"],r["qid"]):r
           for r in read_jsonl(output / "candidate-pools.jsonl") if r["partition"]=="dev"}
    split=json.loads((HERE/"ce-study-query-split.json").read_text(encoding="utf-8"))
    dev_ids={c:set(v["dev_ids"]) for c,v in split["corpora"].items()}
    grouped=defaultdict(list)
    for row in read_jsonl(dev_path):
        grouped[(row["corpus"],row["stage_embedder"],row["stage"],row["reranker"])].append(row)
    frozen={"protocol_version":"omnidb-ce-study-v1","selection_partition":"dev",
            "alpha_grid":list(ALPHAS),"rrf_k_grid":list(RRF_KS),"choices":{}}
    for cell, rows in sorted(grouped.items()):
        corpus, stage_embedder, stage, reranker = cell
        expected_queries = set()
        split_path = HERE / "ce-study-query-split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))["corpora"][corpus]
        expected_queries.update(split["dev_ids"])
        expected_windows = set(args.windows)
        observed = {(row["window"], row["qid"]) for row in rows}
        expected = {(w, qid) for w in expected_windows for qid in expected_queries}
        if observed != expected:
            raise ValueError(f"incomplete dev cell {cell}: missing={len(expected-observed)} "
                             f"extra={len(observed-expected)}")
        scores={}
        for alpha in ALPHAS:
            for k_rrf in RRF_KS:
                by_window=defaultdict(list)
                for row in rows:
                    key=(row["corpus"],row["stage_embedder"],row["stage"],row["window"],row["qid"])
                    if key not in pools:
                        raise ValueError(f"missing dev candidate pool {key}")
                    pool=pools[key]["pool_ids"]
                    rel=_qrels_for_query(HERE/".cache3"/"beir"/f"{row['corpus']}-qrels-test.tsv",row["qid"],dev_ids[row["corpus"]])
                    ranking=rank_rrf(pool,row["ce_scores"],alpha,k_rrf)
                    by_window[row["window"]].append(qrels_metrics(ranking,rel)["ndcg10"])
                scores[(alpha,k_rrf)]=float(np.mean([np.mean(v) for _,v in sorted(by_window.items())]))
        best=max(scores,key=lambda z:(scores[z],-z[0],-z[1]))
        frozen["choices"]["|".join(cell)]={"alpha":best[0],"rrf_k":best[1],
            "dev_mean_ndcg10":scores[best],"dev_rows":len(rows)}
    if not frozen["choices"]:
        raise ValueError("no development cells to tune")
    dest=output/"frozen-parameters.json"
    if dest.exists():
        if json.loads(dest.read_text(encoding="utf-8"))!=frozen:
            raise ValueError("existing frozen parameters differ; refusing overwrite")
    else:
        atomic_json(dest,frozen)
    print(f"frozen {len(frozen['choices'])} choices from dev only")

def apply_frozen_test_blends(output: Path) -> None:
    frozen=json.loads((output/"frozen-parameters.json").read_text(encoding="utf-8"))
    if frozen.get("selection_partition") != "dev":
        raise ValueError("frozen choice file is not dev-selected")
    pool_map={(r["corpus"],r["stage_embedder"],r["stage"],r["window"],r["qid"]):r
              for r in read_jsonl(output/"candidate-pools.jsonl") if r["partition"]=="test"}
    result_path=output/"test-results.jsonl"
    tmp=result_path.with_suffix(".part")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        for row in read_jsonl(output/"test-scores.jsonl"):
            cell="|".join((row["corpus"],row["stage_embedder"],row["stage"],row["reranker"]))
            if cell not in frozen["choices"]:
                raise ValueError(f"missing frozen dev choice {cell}")
            choice=frozen["choices"][cell]
            key=(row["corpus"],row["stage_embedder"],row["stage"],row["window"],row["qid"])
            pool=pool_map[key]["pool_ids"]
            ranking=rank_rrf(pool,row["ce_scores"],choice["alpha"],choice["rrf_k"])
            rel=_qrels_for_query(HERE/".cache3"/"beir"/f"{row['corpus']}-qrels-test.tsv",row["qid"])
            row={**row,"frozen_alpha":choice["alpha"],"frozen_rrf_k":choice["rrf_k"],
                 "blend_ranking":ranking,"blend_ndcg10":qrels_metrics(ranking,rel)["ndcg10"],
                 "judged_at10_blend":qrels_metrics(ranking,rel)["judged_at10"],
                 "unjudged_at10_blend":qrels_metrics(ranking,rel)["unjudged_at10"]}
            f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,result_path)
    print(f"wrote frozen test blends: {result_path}")

def _qrels_for_query(path: Path, qid: str, allowed_qids: set[str] | None = None) -> dict[str,int]:
    """Read query labels, optionally loading only a locked partition's IDs."""
    allowed_hash = None if allowed_qids is None else hashlib.sha256(
        "\n".join(sorted(allowed_qids)).encode("utf-8")).hexdigest()
    cache_key=(str(path),allowed_hash)
    if cache_key not in _QRELS_CACHE:
        import csv
        data=defaultdict(dict)
        with path.open(encoding="utf-8",newline="") as f:
            reader=csv.DictReader(f,delimiter="\t")
            for row in reader:
                query_id=str(row["query-id"])
                if allowed_qids is None or query_id in allowed_qids:
                    data[query_id][str(row["corpus-id"])]=int(row["score"])
        _QRELS_CACHE[cache_key]=dict(data)
    return _QRELS_CACHE[cache_key].get(str(qid),{})

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pools", "dev", "freeze", "test"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpora", default=",".join(CORPORA))
    parser.add_argument("--rerankers", default=",".join(RERANKERS))
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--embedders", default=",".join(EMBEDDERS))
    parser.add_argument("--windows", default=",".join(map(str, WINDOWS)))
    parser.add_argument("--split", default=str(HERE / "ce-study-query-split.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--accept-short-pools", action="store_true",
                        help="record actual short pool sizes rather than aborting")
    args = parser.parse_args()
    args.corpora = tuple(args.corpora.split(","))
    args.rerankers = tuple(args.rerankers.split(","))
    args.stages = tuple(args.stages.split(","))
    args.embedders = tuple(args.embedders.split(","))
    args.windows = tuple(int(x) for x in args.windows.split(","))
    unknown = set(args.rerankers) - set(RERANKER_REVISIONS)
    if unknown: raise ValueError(f"un-pinned rerankers: {sorted(unknown)}")
    split_path=Path(args.split)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validate_manifest(split_manifest)
    output = prepare_run(args, split_manifest)
    config_path=output/"run-config.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    split_hash=sha256_file(split_path)
    if config.get("split_sha256") != split_hash:
        raise ValueError("split file changed since run creation")
    if args.phase == "pools":
        generate_pools(args, output, split_manifest)
    elif args.phase == "dev":
        if not (output / "candidate-pools.jsonl").exists(): raise FileNotFoundError("run pools phase first")
        score_partition(args, output, "dev")
    elif args.phase == "freeze":
        freeze_development_parameters(output, args)
    elif args.phase == "test":
        frozen_path=output / "frozen-parameters.json"
        if not frozen_path.exists():
            raise FileNotFoundError("freeze dev parameters before test phase")
        frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen.get("selection_partition") != "dev":
            raise ValueError("frozen parameters do not record dev-only selection")
        score_partition(args, output, "test")
        apply_frozen_test_blends(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
