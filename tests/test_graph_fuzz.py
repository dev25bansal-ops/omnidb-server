"""Property-based oracle test for MemoryStore.related() (analysis §6.1).

Hypothesis draws random directed graphs; a Python BFS over the undirected
projection is the oracle. related() must return exactly the nodes reachable
within h undirected hops of the start, each tagged with its true minimum
depth, and oversized max_hops must clamp to the engine cap of 8.
"""

import tempfile
from pathlib import Path

from conftest import make_store
from hypothesis import given, settings
from hypothesis import strategies as st

_NODES = range(1, 13)

edge_lists = st.lists(
    st.tuples(st.integers(1, 12), st.integers(1, 12),
              st.sampled_from(["calls", "imports"])),
    max_size=18,
).map(lambda triples: list(dict.fromkeys(triples)))


def _oracle(edges: list[tuple[int, int, str]]) -> dict[int, int]:
    adj: dict[int, set[int]] = {n: set() for n in _NODES}
    for src, dst, _label in edges:
        adj[src].add(dst)
        adj[dst].add(src)
    depth = {1: 0}
    frontier = [1]
    while frontier:
        nxt: list[int] = []
        for node in frontier:
            for nb in adj[node]:
                if nb not in depth:
                    depth[nb] = depth[node] + 1
                    nxt.append(nb)
        frontier = nxt
    return depth


def _build_graph(store, edges: list[tuple[int, int, str]]) -> dict[int, int]:
    ids = {n: store.remember(f"node {n}")["id"] for n in _NODES}
    for src, dst, label in edges:
        store.link(ids[src], ids[dst], label=label)
    return ids


@settings(max_examples=40, deadline=None)
@given(edges=edge_lists)
def test_related_matches_bfs_oracle(edges):
    oracle = _oracle(edges)
    with tempfile.TemporaryDirectory() as home:
        st = make_store(Path(home))
        try:
            ids = _build_graph(st, edges)
            rev = {mid: n for n, mid in ids.items()}
            for h in (1, 2):
                got = st.related(ids[1], max_hops=h)
                expected = {ids[n] for n, d in oracle.items() if d <= h} - {ids[1]}
                assert {r["id"] for r in got} == expected
                for row in got:
                    assert row["depth"] == oracle[rev[row["id"]]]

            clamped = st.related(ids[1], max_hops=99)
            assert all(r["depth"] <= 8 for r in clamped)
            assert {r["id"] for r in clamped} == (
                {ids[n] for n, d in oracle.items() if d <= 8} - {ids[1]}
            )
        finally:
            st.close()
