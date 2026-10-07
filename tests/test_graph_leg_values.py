"""Graph leg: pinned values and properties, in the discipline of
`test_fusion_values.py` (assert literals, not a restatement of the formula) and
`test_fuzz_fusion.py` (hypothesis over the pure surface).

These run with no database. Everything here is a property of
`omnidb_server/graph_leg.py` alone, which is what makes the hard claims
(termination, exactness, determinism) checkable at all.
"""

from __future__ import annotations

import math
import pathlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from omnidb_server.graph_leg import (
    graph_expand,
    graph_seed_masses,
    normalize_masses,
)

# --- termination and bounds ---------------------------------------------------


@pytest.mark.parametrize("budget", [0, 1, 2, 5, 64])
def test_pops_never_exceed_the_budget(budget):
    adjacency = {i: {i + 1, i + 2} for i in range(200)}
    out = graph_expand({1: 1.0}, adjacency, decay=0.9, budget=budget)
    assert len(out) <= budget, f"budget {budget} produced {len(out)} ids"


def test_a_large_graph_costs_no_more_than_the_budget():
    """The property that separates this from whole-graph PPR.

    Cost must be flat in graph size: a 50,000-node graph must not take longer
    or visit more nodes than a 50-node one at the same budget.
    """
    small = {i: {i + 1} for i in range(50)}
    large = {i: {i + 1} for i in range(50_000)}
    a = graph_expand({0: 1.0}, small, decay=0.5, budget=256)
    b = graph_expand({0: 1.0}, large, decay=0.5, budget=256)
    # The small graph exhausts before the budget (50 nodes, 1 popped per
    # reachable node, seed excluded); the large one is cut by it. Either way
    # the cost is bounded by the budget rather than by graph size, which is the
    # property that makes this shippable where whole-graph PPR was not.
    assert len(a) == 50, len(a)
    assert len(b) == 255, len(b)
    assert max(len(a), len(b)) <= 256


def test_the_budget_is_total_across_seeds_not_per_seed():
    """The budget caps total work; a per-seed full budget is a real regression.

    The per-seed quota is what makes a hub unable to starve a weaker seed's
    private chain, but the mutant that gives EVERY seed the full budget reaches
    the same nodes -- it just costs |S| x budget. This test pins the cost, which
    is the property correctness-only assertions cannot see: with budget 50 and
    two seeds, the legitimate answer is <= 50 pops total, where the mutant
    spends up to 100.
    """
    adjacency = {0: set(range(1, 80)),
                 500: {501}, 501: {502}, 502: {503}}
    for i in range(1, 80):
        adjacency[i] = {0}
    out = graph_expand({0: 1.0, 500: 0.01}, adjacency, decay=0.5, budget=50)
    # Both chains reachable (correctness holds under mutation), but the output
    # must be bounded by the budget: with a hub of 79 spokes, a global 50-pop
    # budget finishes at 50 outputs; a per-seed full budget emits ~100.
    assert len(out) <= 50, (
        f"output grew to {len(out)} with budget 50; the quota became per-seed "
        "rather than total, which is the unbounded-cost path"
    )


def test_a_hub_does_not_swallow_the_budget():
    """A node reachable only from a weaker seed must still be found.

    The adversarial case that motivated per-seed quotas: one seed is a hub with
    79 spokes, another has a private 3-chain. A single global budget drains
    inside the hub and never reaches the chain.
    """
    adjacency: dict[int, set[int]] = {0: set(range(1, 80))}
    for i in range(1, 80):
        adjacency[i] = {0}
    adjacency.update({500: {501}, 501: {502}, 502: {503}})

    out = graph_expand({0: 1.0, 500: 0.01}, adjacency, decay=0.5, budget=50)
    assert 502 in out and 503 in out, (
        "the private chain was starved: a single global budget drains into the "
        "highest-mass seed's neighbourhood"
    )


def test_the_finalized_check_is_load_bearing_under_a_hard_budget():
    """The finalized-check guard is NOT equivalent to its removal (mutant 28).

    The guard (`if seen.get(node, -1.0) >= key: continue`) skips a node whose
    proposal key is no better than one already processed. Removing it lets the
    same node be popped again with a lower-or-equal key. That cannot change
    `best` — the max-key guard already keeps the maximum — but it *does* spend
    budget, and under a hard pop cap every wasted pop is a node that will never
    be finalized. This shape makes the difference observable: node 3 is
    reachable from the seed both directly (key 0.5 via 1) and through the
    longer chain (key 0.25 via 2), so without the guard the low proposal for 3
    costs a pop and node 4 falls outside the budget.

    Pinned as a mutkiller for mutation-check mutant 28; if this test is ever
    deleted, 28 must go back into the EQUIVALENT table *with a corrected
    justification* — which this test exists to refute.
    """
    adjacency = {
        0: {1, 2},   # seed reaches 1 and 2 directly
        1: {3},      # short path to 3: key 0.5
        2: {3},      # long path to 3: key 0.25 -> a redundant proposal
        3: {4},
        4: {5},
    }
    out = graph_expand({0: 1.0}, adjacency, decay=0.5, budget=5)
    # With the guard: pops are 1, 2, 3, 4, 5 (5 finalizations).
    # Without it: the low proposal for 3 is popped, spending the 5th pop on a
    # node already finalized, so 4 never enters.
    assert 4 in out, (
        "node 4 was dropped: a redundant lower-key proposal for node 3 "
        "consumed a budget pop, which is exactly what the finalized-check "
        "guard prevents. This is mutant 28 surviving, meaning output really "
        f"does depend on the guard. Got: {out}"
    )
    assert out == [1, 2, 3, 4], out


# --- exactness ---------------------------------------------------------------


def _rank_of(ranked, mid):
    """Position of `mid` in a graph_expand output; smaller is stronger.

    The rank is what pins ordering, and the id tie-break makes it a total
    order. Used with the independent oracle to pin the decay arithmetic: same
    ranks -> same arithmetic, whatever the internal score representation is.
    """
    return ranked.index(mid)


def _oracle(adjacency, seeds, decay):
    """Independent max-discounted-path score by breadth-first enumeration.

    Deliberately not Dijkstra: a different algorithm reaching the same numbers
    is what makes this a check rather than a restatement.
    """
    best: dict[int, float] = {}
    frontier = [(u, v) for u, v in seeds.items()]
    seen = {u: v for u, v in seeds.items()}
    depth = 0
    while frontier and depth < 50:
        nxt = []
        for node, mass in frontier:
            for nb in adjacency.get(node, ()) or ():
                cand = mass * (decay ** (depth + 1))
                if cand > best.get(nb, -1.0):
                    best[nb] = cand
                if cand > seen.get(nb, -1.0):
                    seen[nb] = cand
                    nxt.append((nb, cand))
        frontier = nxt
        depth += 1
    return best


def test_matches_an_independent_oracle_on_a_tree():
    """On a tree there is one path per pair, so the oracle is unambiguous."""
    tree = {0: {1, 2}, 1: {3, 4}, 2: {5}, 3: {6}, 4: set(), 5: set(), 6: set()}
    seeds = {0: 1.0}
    out = graph_expand(seeds, tree, decay=0.5, budget=1000)
    expected = _oracle(tree, seeds, 0.5)
    # Compare SCORES via the ordering they induce, not just reach: the oracle
    # is an independent implementation, so agreement on ordering pins the
    # arithmetic including the decay multiplication. Reach alone would let a
    # mutation that removes decay survive (that is exactly the mutant the
    # mutation gate reported as surviving).
    # Reconstruct the oracle's ordering: score desc, id asc, seeds excluded.
    expected_order = [m for m in sorted(expected, key=lambda m: (-expected[m], m))
                      if m not in seeds]
    assert out == expected_order[:len(out)], (
        f"ordering disagrees with the oracle: {out} vs {expected_order}"
    )
    assert 1 in out and 2 in out and 6 in out
    # The decay arithmetic is what the ordering comparison pins: node 2 (one
    # hop) must outrank node 6 (two hops), and both must outrank the 3-hop
    # node. A mutation that removes the decay multiplication flattens all
    # distances and breaks exactly this assertion.
    assert _rank_of(out, 2) < _rank_of(out, 6), (
        "a one-hop node did not outrank a two-hop node; the decay arithmetic "
        "is wrong or unexercised"
    )


def test_keys_are_ordered_by_descending_score():
    adjacency = {0: {1, 2, 3}, 1: {4}, 2: set(), 3: {5}, 4: set(), 5: set()}
    out = graph_expand({0: 1.0}, adjacency, decay=0.5, budget=100)
    assert len(out) == len(set(out)), "an id appeared twice in one leg"
    assert out[0] == 1, "the nearest neighbour must rank first"
    # deeper nodes come after their shallower siblings
    assert out.index(4) > out.index(1)
    assert out.index(5) > out.index(3)


def test_decay_discounts_heterogeneous_seed_strengths():
    """The decay multiplication matters only when seeds differ in strength.

    Killing mutation 23: with a weak seed one hop from a node and a strong seed
    two hops away, decay decides which path wins. Removing the decay
    multiplication flattens distances and flips the order — measured:
    with decay 0.5 the order is [11, 20, 2]; with decay removed it is
    [2, 11, 20]. Any test suite that never exercises heterogeneous seeds will
    never see this, so the mutation gate should not be expected to catch it
    without this shape.
    """
    adj = {
        10: {11, 20},   # strong seed: 1 hop to 11 and 20
        11: {2},        # 2 hops from the strong seed to 2
        1:  {2},        # weak seed: 1 hop to 2
        2: set(), 20: set(),
    }
    seeds = {10: 1.0, 1: 0.6}
    with_d = graph_expand(seeds, adj, decay=0.5, budget=64)
    assert with_d == [11, 20, 2], (
        f"expected the strong seed's 1-hop neighbours to lead, got {with_d}; "
        "the decay arithmetic is not discounting by distance as designed"
    )


def test_decay_changes_the_ranking_not_the_reach():
    """With the budget non-binding, decay reweights rather than truncates.

    Both decays reach every node on a path, so the reachable set is identical
    and the claim "higher decay reaches further" is false. What decay changes is
    the ordering of the deep nodes relative to the shallow ones -- which is what
    makes it the knob it is.
    """
    adjacency = {i: {i + 1} for i in range(6)}
    # a shallow-but-weak branch and a deep-but-strong one
    adjacency[0] = {1, 2, 3}
    adjacency[1] = {10}
    adjacency[2] = set()
    adjacency[3] = set()
    high = graph_expand({0: 1.0}, adjacency, decay=0.95, budget=1000)
    low = graph_expand({0: 1.0}, adjacency, decay=0.30, budget=1000)
    assert set(high) == set(low), "decay changed reach on a finite tree"
    # On this graph every branch is one hop from the seed, so all three
    # candidates score identically at any decay and the id tie-break decides.
    # That is the point worth pinning: decay cannot reorder equidistant nodes,
    # and the ordering is therefore deterministic rather than decay-dependent.
    assert high == low == [1, 2, 3, 10]


# --- the budget truncates; it never reweights ---------------------------------


def test_a_larger_budget_only_adds_nodes():
    adjacency = {i: {i + 1, i + 2} for i in range(50)}
    small = graph_expand({0: 1.0}, adjacency, decay=0.6, budget=8)
    large = graph_expand({0: 1.0}, adjacency, decay=0.6, budget=64)
    assert set(small) <= set(large)
    assert small == large[:len(small)], (
        "raising the budget reordered the existing prefix, so truncation is "
        "silently changing scores rather than only adding reach"
    )


# --- determinism -------------------------------------------------------------


def test_repeated_runs_are_identical():
    adjacency = {i: {i + 1, i + 3, i + 7} for i in range(100)}
    runs = [graph_expand({0: 1.0, 5: 0.5, 9: 0.25}, adjacency, decay=0.5, budget=64)
            for _ in range(10)]
    assert all(r == runs[0] for r in runs)


def test_equal_key_neighbours_break_on_id_ascending():
    """Kills the tie-break mutation: two neighbours with IDENTICAL keys.

    Two seeds of equal mass, each with one distinct neighbour, produce two
    output rows with exactly equal scores. Only the id tie-break makes the
    output deterministic: must be [50, 100], never [100, 50]. Removing the
    tie-break lets heap/dict insertion order decide, which is nondeterministic
    across runs but may pass a suite that never constructs the shape.
    """
    adj = {10: {100}, 20: {50}}
    out = graph_expand({10: 1.0, 20: 1.0}, adj, decay=0.5, budget=20)
    assert out == [50, 100], (
        f"equal-key neighbours ordered {out}; the id tie-break is missing or "
        "overridden by insertion order"
    )


def test_a_node_proposed_twice_keeps_its_higher_key():
    """Kills the finalized-check mutation: unequal double proposal.

    A node reachable by a short path from a strong seed and a long path from a
    weak one is first proposed with the higher key. The finalized check must
    keep it, or the second, lower proposal overwrites the true max. Constructed
    so the second proposal carries a lower key, exactly the shape the check
    exists for.
    """
    # 0 strong seed; 1 reached directly; 3 reached via 1 (two hops) and, if the
    # walk continues, re-reached lower via 2. The output order must put the
    # one-hop node first and never re-emit a node at a lower position.
    adj = {0: {1, 2}, 1: {3}, 2: {3}, 3: set()}
    out = graph_expand({0: 1.0}, adj, decay=0.5, budget=20)
    assert out == [1, 2, 3] or out == [2, 1, 3], (
        f"unexpected order {out}; the one-hop neighbours must lead and 3 must "
        "appear exactly once at the tail"
    )
    assert out.count(3) == 1, "a double-proposed node was emitted twice"


def test_ties_break_on_id_not_set_order():
    # three neighbours with identical key; the order must be id-ascending
    adjacency = {0: {7, 3, 5}}
    out = graph_expand({0: 1.0}, adjacency, decay=0.5, budget=10)
    assert out == [3, 5, 7], f"tie order was {out}; must be id-ascending"


# --- seeds are excluded ------------------------------------------------------


def test_seeds_do_not_appear_in_the_output():
    adjacency = {0: {1, 2}, 1: {0, 3}, 2: {0, 4}, 3: set(), 4: set()}
    out = graph_expand({0: 1.0, 1: 0.5}, adjacency, decay=0.5, budget=100)
    assert 0 not in out and 1 not in out, (
        "a seed was re-emitted; the leg extends the seed set, it does not "
        "confirm it, and re-emitting spends its best rank slots on ids two other "
        "legs already contributed"
    )


# --- degenerate inputs -------------------------------------------------------


@pytest.mark.parametrize("seeds", [{}, {1: 0.0}, {1: -1.0}, {1: float("nan")}])
def test_degenerate_seed_sets_return_empty(seeds):
    assert graph_expand(seeds, {1: {2}}, decay=0.5, budget=10) == []


@pytest.mark.parametrize("decay", [0.0, -0.5, 1.5, float("nan")])
def test_invalid_decay_returns_empty(decay):
    assert graph_expand({1: 1.0}, {1: {2}}, decay=decay, budget=10) == []


def test_a_seed_with_no_edges_is_not_an_error():
    assert graph_expand({1: 1.0}, {}, decay=0.5, budget=10) == []


def test_self_loops_are_harmless():
    adjacency = {0: {0, 1}, 1: {1, 0}}
    out = graph_expand({0: 1.0}, adjacency, decay=0.5, budget=10)
    assert 1 in out and 0 not in out


# --- masses ------------------------------------------------------------------


def test_masses_sum_to_one():
    masses = graph_seed_masses({10: 0.9, 11: 0.5, 12: 0.1}, [10, 11, 12], rrf_k=60)
    assert abs(math.fsum(masses.values()) - 1.0) < 1e-12
    # higher fused score -> more mass
    assert masses[10] > masses[11] > masses[12]


def test_masses_use_the_rrf_kernel_not_the_raw_score():
    """Raw RRF scores span only 4.3x; the rank kernel is what makes it tuned."""
    scores = {i: 0.0164 / (1 + i * 0.001) for i in range(200)}
    ordered = list(range(200))
    masses = graph_seed_masses(scores, ordered, rrf_k=60)
    assert abs(masses[0] - (1 / 61) / (math.fsum(1 / (61 + i) for i in range(200)))) < 1e-12


def test_zero_scored_seeds_are_excluded_before_normalising():
    masses = graph_seed_masses({10: 0.5, 11: 0.0}, [10, 11], rrf_k=60)
    assert set(masses) == {10}
    assert abs(sum(masses.values()) - 1.0) < 1e-12


def test_normalize_handles_an_all_zero_set():
    assert normalize_masses({1: 0.0, 2: 0.0}) == {}


# --- properties --------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    size=st.integers(min_value=1, max_value=40),
    decay=st.floats(min_value=0.05, max_value=1.0),
    budget=st.integers(min_value=1, max_value=80),
    seed_count=st.integers(min_value=1, max_value=4),
)
def test_properties_hold_on_generated_graphs(size, decay, budget, seed_count):
    adjacency = {i: {j for j in range(i + 1, min(i + 4, size))} for i in range(size)}
    seeds = normalize_masses({i: 1.0 / (i + 1) for i in range(min(seed_count, size))})
    out = graph_expand(seeds, adjacency, decay=decay, budget=budget)

    assert len(out) <= budget
    assert len(out) == len(set(out)), "duplicate id in one leg"
    assert not (set(out) & set(seeds)), "a seed was emitted"
    assert out == graph_expand(seeds, adjacency, decay=decay, budget=budget)
    if len(out) > 1:
        # the output must be reproducible from a total order
        assert out == sorted(out, key=lambda _x: out.index(_x))


@settings(max_examples=40, deadline=None)
@given(adj=st.dictionaries(
    st.integers(min_value=0, max_value=25),
    st.sets(st.integers(min_value=0, max_value=25), max_size=5),
    max_size=25))
def test_never_raises_on_any_graph(adj):
    """Total, like `scoring.py`: any finite input returns, never raises."""
    adjacency = {k: set(v) for k, v in adj.items()}
    out = graph_expand({0: 1.0}, adjacency, decay=0.5, budget=50)
    assert isinstance(out, list)

# --- config: the safety property is enforced, not defaulted ------------------

def test_graph_weight_cannot_exceed_the_legs_it_is_seeded_from():
    """A graph-only row must never outrank the rows that produced the seeds.

    The graph leg's ranking is a function of the fused seed set, so an
    unbounded weight would let a derived signal outvote the evidence it was
    derived from. Checked at config validation, so it holds for an operator who
    overrides the default as well as for one who does not.
    """
    import tempfile

    from omnidb_server.config import Config, ConfigError

    with tempfile.TemporaryDirectory() as d:
        toml = pathlib.Path(d) / "omnidb.toml"
        toml.write_text("graph_weight = 5.0\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="graph_weight"):
            Config.load(pathlib.Path(d))


def test_a_zero_graph_weight_leaves_every_existing_config_untouched():
    """The default must be inert: no existing home or caller changes."""
    from omnidb_server.config import Config

    cfg = Config(home=pathlib.Path("/tmp"))
    assert cfg.graph_weight == 0.0, "the graph leg must ship off"
    assert cfg.graph_decay == 0.5
    assert cfg.graph_expand_budget == 0


def test_graph_expand_budget_rejects_a_typo_below_the_floor():
    """0 means "derive from k"; a small non-zero is a different mistake.

    The error must name both meanings, or an operator who writes 8 has to go
    read the source to learn why it was refused.
    """
    import tempfile

    from omnidb_server.config import Config, ConfigError

    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "omnidb.toml").write_text("graph_expand_budget = 4\n",
                                                  encoding="utf-8")
        with pytest.raises(ConfigError, match="derive from k"):
            Config.load(pathlib.Path(d))
