from __future__ import annotations

import json
import time


def test_remember_many_basic(store):
    items = [{"content": f"remember many basic item {i}"} for i in range(5)]
    results = store.remember_many(items)

    assert len(results) == 5
    ids = [r["id"] for r in results]
    assert ids == sorted(ids), "remember_many ids must be returned in ascending order"
    assert len(set(ids)) == 5, "remember_many ids must be unique"
    assert ids == list(range(ids[0], ids[0] + 5)), (
        f"expected sequential ids, got {ids}"
    )

    hits = store.recall("remember many basic item", k=10)
    hit_ids = {h["id"] for h in hits}
    for mid in ids:
        assert mid in hit_ids, f"memory {mid} not retrievable by recall"


def test_remember_many_with_agent_and_trigger(store):
    items = [
        {
            "content": "alpha agent-tagged memory about gardening",
            "agent_id": "agent-a",
            "trigger_patterns": ["garden", "plant"],
        },
        {
            "content": "beta agent-tagged memory about cooking",
            "agent_id": "agent-b",
            "trigger_patterns": ["recipe", "kitchen"],
        },
        {
            "content": "gamma memory with only agent",
            "agent_id": "agent-a",
        },
        {
            "content": "delta memory with only triggers",
            "trigger_patterns": ["delta", "river"],
        },
    ]
    results = store.remember_many(items)
    assert len(results) == 4

    by_id = {r["id"]: store.get(r["id"]) for r in results}

    assert by_id[results[0]["id"]]["agent_id"] == "agent-a"
    assert by_id[results[1]["id"]]["agent_id"] == "agent-b"
    assert by_id[results[2]["id"]]["agent_id"] == "agent-a"
    assert by_id[results[3]["id"]]["agent_id"] is None

    assert json.loads(by_id[results[0]["id"]]["trigger_patterns"]) == ["garden", "plant"]
    assert json.loads(by_id[results[1]["id"]]["trigger_patterns"]) == ["recipe", "kitchen"]
    assert by_id[results[2]["id"]]["trigger_patterns"] is None
    assert json.loads(by_id[results[3]["id"]]["trigger_patterns"]) == ["delta", "river"]


def test_remember_many_throughput(store):
    n = 500
    items = [{"content": f"batch item {i}"} for i in range(n)]

    t0 = time.perf_counter()
    results = store.remember_many(items)
    dt = time.perf_counter() - t0

    assert len(results) == n, f"expected {n} results, got {len(results)}"
    measured = n / dt if dt > 0 else float("inf")
    print(f"remember_many throughput: {measured:.1f} writes/s ({dt:.3f}s for {n})")
    assert measured >= 800.0, (
        f"remember_many throughput {measured:.1f} writes/s below floor 800/s "
        f"({n} items in {dt:.3f}s)"
    )


def test_remember_many_vs_single_parity(tmp_path):
    import conftest

    contents = [f"parity probe content token {i} python recipe" for i in range(10)]
    probe = "parity probe content python recipe"

    # single-call baseline (its own store so ids start at 1)
    single_store = conftest.make_store(tmp_path / "single")
    try:
        single_ids = [single_store.remember(c)["id"] for c in contents]
        single_hits = single_store.recall(probe, k=5)
        assert single_hits, "baseline recall returned nothing"
        single_top = single_hits[0]
    finally:
        single_store.close()

    # batch path on a parallel store with identical contents
    batch_store = conftest.make_store(tmp_path / "batch")
    try:
        batch_results = batch_store.remember_many(
            [{"content": c} for c in contents]
        )
        batch_ids = [r["id"] for r in batch_results]
        batch_hits = batch_store.recall(probe, k=5)
        assert batch_hits, "batch recall returned nothing"
        batch_top = batch_hits[0]
    finally:
        batch_store.close()

    assert batch_ids == single_ids, (
        f"id sequences diverge: single={single_ids} batch={batch_ids}"
    )
    assert batch_top["id"] == single_top["id"], (
        f"top result differs: single={single_top['id']} batch={batch_top['id']}"
    )
    batch_set = {h["id"] for h in batch_hits}
    single_set = {h["id"] for h in single_hits}
    assert batch_set == single_set, (
        f"top-k ids diverge: single={single_set} batch={batch_set}"
    )