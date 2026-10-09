"""Split judged BEIR query IDs by a stable, outcome-independent SHA-256 order.

The split is query-disjoint within each corpus. It uses only corpus name and
query ID; no qrels, retrieval scores, or reranker outputs influence membership.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

PROTOCOL_VERSION = "omnidb-ce-study-v1"
DEV_FRACTION = 0.40


def split_query_ids(corpus: str, query_ids: Iterable[str], dev_fraction: float = DEV_FRACTION):
    """Return deterministic dev/test ID lists in original-independent hash order."""
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between 0 and 1")
    ids = list(map(str, query_ids))
    if len(ids) != len(set(ids)):
        raise ValueError(f"{corpus}: duplicate query IDs")
    ordered = sorted(ids, key=lambda q: hashlib.sha256(
        f"{PROTOCOL_VERSION}|{corpus}|{q}".encode("utf-8")).hexdigest())
    n_dev = int(len(ordered) * dev_fraction)
    # Keep both partitions nonempty if there are at least two judged queries.
    if len(ordered) >= 2:
        n_dev = min(max(n_dev, 1), len(ordered) - 1)
    return {"dev": ordered[:n_dev], "test": ordered[n_dev:]}


def split_manifest(corpora: dict[str, Iterable[str]]) -> dict:
    result = {"protocol_version": PROTOCOL_VERSION, "dev_fraction": DEV_FRACTION,
              "corpora": {}}
    for corpus, query_ids in sorted(corpora.items()):
        ids = list(map(str, query_ids))
        parts = split_query_ids(corpus, ids)
        def digest(values):
            payload = "\n".join(values).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()
        result["corpora"][corpus] = {
            "n_total": len(ids), "n_dev": len(parts["dev"]),
            "n_test": len(parts["test"]),
            "dev_ids": parts["dev"], "test_ids": parts["test"],
            "dev_sha256": digest(parts["dev"]),
            "test_sha256": digest(parts["test"]),
        }
    return result


def partition_query_items(queries, split: dict) -> dict[str, list]:
    """Partition aligned query tuples without changing tuple contents/order."""
    qids = [str(qid) for qid, _text, _rel in queries]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate qid in query rows")
    if set(qids) != set(split["dev"]) | set(split["test"]):
        raise ValueError("query rows do not match split manifest IDs")
    dev, test = set(split["dev"]), set(split["test"])
    return {
        "dev": [row for row in queries if str(row[0]) in dev],
        "test": [row for row in queries if str(row[0]) in test],
    }


def validate_manifest(manifest: dict) -> None:
    for corpus, cell in manifest["corpora"].items():
        dev, test = set(cell["dev_ids"]), set(cell["test_ids"])
        if dev & test:
            raise ValueError(f"{corpus}: dev/test overlap")
        if len(dev) != cell["n_dev"] or len(test) != cell["n_test"]:
            raise ValueError(f"{corpus}: count mismatch")
        all_ids = dev | test
        expected = split_query_ids(corpus, all_ids)
        if expected["dev"] != cell["dev_ids"] or expected["test"] != cell["test_ids"]:
            raise ValueError(f"{corpus}: split does not match protocol hash order")
        def digest(values):
            return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
        if digest(cell["dev_ids"]) != cell["dev_sha256"] or digest(cell["test_ids"]) != cell["test_sha256"]:
            raise ValueError(f"{corpus}: ID-list hash mismatch")


def main():
    import sys
    here = __import__('pathlib').Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from research_bench_v3 import load_beir
    corpora = {}
    for corpus in ("nfcorpus", "scifact", "scidocs", "fiqa"):
        _, queries = load_beir(corpus)
        corpora[corpus] = [qid for qid, _, _ in queries]
    result = split_manifest(corpora)
    validate_manifest(result)
    dest = __import__('pathlib').Path(__file__).resolve().parent / "ce-study-query-split.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    for c, v in result["corpora"].items():
        print(f"{c}: dev={v['n_dev']} test={v['n_test']} "
              f"dev_sha256={v['dev_sha256']} test_sha256={v['test_sha256']}")

if __name__ == "__main__":
    main()
