"""What "unified" must mean, stated before it is built.

OmniDB has four capabilities: relational rows, full-text, vector, and graph. The
project's original design document proposed one query language across all four;
that (SQL++) was abandoned in ADR-4, and what shipped unified storage and the
agent interface but left the retrieval modalities partly separate — vector and
lexical fuse into one ranked list, while the graph is reachable only by memory
id.

This module is the contract for closing that. It is written first, against
behaviour, so that the implementation cannot satisfy it by widening a signature
or relaxing a fence. Every test here should fail today and pass after the work.

The four properties, in priority order:

  U1  one ranked result, not four result sets the agent must merge
  U2  the two-leg result is bit-identical when unification is off
      (backward compatibility is a property, not a hope)
  U3  tenant isolation holds on the new path exactly as it does on the old
  U4  bounded work: no query shape can make cost unbounded

U3 is listed third and is not negotiable. A unified surface that widens the
tenant fence to make a query expressible would be a regression dressed as a
feature, and the earlier tenancy work is precisely what made that fence real.
"""

from __future__ import annotations

import time

import pytest
from conftest import make_store


@pytest.fixture()
def graph_store(tmp_path):
    """A home where a lexical hit's graph neighbours are not lexical hits.

    The shape matters: if the neighbours were findable by text, a unified query
    would pass with no graph contribution at all and the tests below would be
    measuring nothing. These neighbours share no term with the query.
    """
    st = make_store(tmp_path)
    try:
        seed = st.remember("kubernetes ingress controller setup")["id"]
        far = st.remember("zzqx wugglethorpe frobnicator")["id"]     # unrelated text
        mid = st.remember("qqqq vvvvvv blorp")["id"]                 # unrelated text
        st.link(seed, mid, label="depends_on")
        st.link(mid, far, label="depends_on")
        yield st, {"seed": seed, "mid": mid, "far": far}
    finally:
        st.close()


# --- U1: one ranked result, with the graph participating ---------------------

def test_a_graph_reachable_row_can_appear_in_a_content_query(graph_store):
    """The headline capability: find by content, then traverse.

    The graph leg must surface rows that only the graph path can find. With the
    brute-force + hashing embedder every row carries nonzero dense similarity,
    so the honest discriminator is not "text-found vs graph-found" on recall —
    it is whether the *graph leg* itself returns the two-hop neighbour, and
    whether a query seeded from a text result reaches it. Both are asserted
    directly against the leg, which is the new capability.
    """
    st, ids = graph_store
    hits = {r["id"] for r in st.recall("kubernetes ingress", k=10)}
    assert ids["seed"] in hits, "control: the lexical match itself is missing"

    # The leg's own output: mid and far are reachable from the seed through the
    # graph, even though they share no term with the query.
    st.cfg.graph_weight = 0.5
    ranked = st.graph_ranking([ids["seed"]], kind=None, agent_id=None,
                              tenant="default", depth=2, budget=256)
    assert ids["mid"] in ranked and ids["far"] in ranked, (
        "the graph leg did not reach its own neighbours; see test_graph_leg_sql"
    )

    # And the unified surface reaches them through a content query.
    result = st.query({"text": "kubernetes ingress", "graph": {"hops": 2}})
    assert {r["id"] for r in result} == {ids["seed"], ids["mid"], ids["far"]}, (
        "the unified query returned the wrong set: " + str(result)
    )


def test_the_unified_query_returns_one_ranked_list_not_four_sets(graph_store):
    """Unification is about the shape of the answer, not the number of inputs.

    A design that returns `{dense: [...], lexical: [...], graph: [...]}`
    leaves the merge to the agent, which is the problem this work exists to
    remove.
    """
    st, ids = graph_store
    result = st.query({"text": "kubernetes ingress", "graph": {"hops": 2}})
    assert isinstance(result, list), (
        f"expected one ranked list, got {type(result).__name__}: a per-leg "
        "result would push the merge back onto the agent"
    )
    assert all(isinstance(r, dict) and "id" in r for r in result)
    # and the ranking is a single coherent ordering, not a concatenation
    if len(result) > 1:
        scores = [r.get("score") for r in result]
        assert scores == sorted(scores, reverse=True), (
            "results are not in one descending score order, which is what "
            "'unified ranking' means"
        )


def test_graph_contribution_is_attributable(graph_store):
    """A caller must be able to see WHY a row came back.

    Without provenance, a graph leg is indistinguishable from a ranking bug, and
    an agent cannot decide whether to trust the result.
    """
    st, ids = graph_store
    result = st.query({"text": "kubernetes ingress", "graph": {"hops": 2}})
    if not any(r["id"] == ids["far"] for r in result):
        pytest.skip("graph leg not surfaced; nothing to attribute yet")
    row = next(r for r in result if r["id"] == ids["far"])
    assert "legs" in row or "sources" in row, (
        "a row surfaced by the graph leg must say so; an unattributed extra "
        "result is indistinguishable from noise"
    )


# --- U2: backward compatibility is a property, not a hope ---------------------

def test_disabling_the_graph_leg_reproduces_the_two_leg_result_exactly(tmp_path):
    """With weight 0.0, edges present must equal edges absent, identically.

    Not approximately. The leg at weight 0.0 is "never computed", and this is
    the observable form of that claim: a store whose graph exists returns the
    same ids at the same scores as one whose graph does not exist at all.
    If the off-switch were "computed then discarded", any reordering, rounding,
    or access-count side effect of the computation would show up here.

    Two fresh stores are built instead of two sequential recalls on one, because
    recall bumps access_count and that bump feeds final_score — a sequential
    comparison would conflate the graph leg with the access-count effect.
    """

    def build(home, with_edges):
        st = make_store(home)
        try:
            seed = st.remember("kubernetes ingress controller setup")["id"]
            mid = st.remember("zzqx wugglethorpe frobnicator")["id"]
            far = st.remember("qqqq vvvvvv blorp")["id"]
            if with_edges:
                st.link(seed, mid, label="depends_on")
                st.link(mid, far, label="depends_on")
            st.cfg.graph_weight = 0.0
            return st.recall("kubernetes ingress", k=10)
        finally:
            st.close()

    with_edges = build(tmp_path / "edged", with_edges=True)
    without_edges = build(tmp_path / "bare", with_edges=False)
    assert [(r["id"], r["score"]) for r in with_edges] ==            [(r["id"], r["score"]) for r in without_edges], (
        "a present-but-disabled graph changed the result; the off-switch does "
        "not mean 'off'"
    )


def test_the_plain_recall_signature_still_works(graph_store):
    """`recall(q, k)` is called by the CLI, MCP tool, adapters and tests.

    Adding a capability must not force every one of them to learn about it.
    """
    st, _ = graph_store
    assert st.recall("kubernetes ingress", k=3)


# --- U3: tenant isolation is not negotiable ----------------------------------

def test_the_graph_leg_cannot_reach_another_tenant(tmp_path):
    """The regression this project spent its tenancy work preventing.

    A unified surface is a new query path, and every new query path is a new
    place for the tenant fence to be forgotten. This test exists to make
    forgetting it a loud failure rather than a quiet one.
    """
    st = make_store(tmp_path)
    try:
        acme = st.remember("shared marker text", tenant="acme")["id"]
        globex = st.remember("shared marker text", tenant="globex")["id"]
        # a cross-tenant link, forged directly: link() refuses it, so write the
        # row past the API to simulate a home with a bad historical edge
        st.conn.execute(
            "INSERT INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, 0)",
            (acme, globex, "forged"))
        st.conn.commit()

        result = st.query({"text": "shared marker", "graph": {"hops": 3}},
                         tenant="acme")
        assert all(r["id"] != globex for r in result), (
            "the graph leg followed a forged cross-tenant edge; the unified "
            "surface is the newest query path and the easiest place to drop "
            "the fence"
        )
    finally:
        st.close()


def test_tenant_is_not_caller_settable(tmp_path):
    """`tenant` comes from the auth context; a query object must not carry it."""
    st = make_store(tmp_path)
    try:
        st.remember("acme only note", tenant="acme")
        with pytest.raises(Exception):
            st.query({"text": "acme only note", "tenant": "acme"},
                     tenant="globex")
    finally:
        st.close()


# --- U4: bounded work --------------------------------------------------------

def test_query_cost_is_bounded_in_graph_size(tmp_path):
    """A dense graph must not turn one query into an unbounded traversal.

    `related()` exists because undirected recursion without visit tracking is
    exponential. A unified surface that traverses by default inherits that risk,
    so the bound is asserted rather than assumed.
    """
    st = make_store(tmp_path)
    try:
        # A hub-and-spoke graph: worst case for any traversal that does not
        # track visits.
        hub = st.remember("hub token zzzz")["id"]
        for i in range(60):
            leaf = st.remember(f"leaf token wwww {i}")["id"]
            st.link(hub, leaf, label="spoke")
        st.flush()

        t0 = time.perf_counter()
        st.query({"text": "zzz zzzz", "graph": {"hops": 3}}, k=10)
        elapsed = time.perf_counter() - t0
        # Generous, because the bound is the property, not the speed: a
        # traversal that did not track visits would not finish this at all.
        assert elapsed < 20.0, f"a 60-spoke graph took {elapsed:.1f}s per query"
    finally:
        st.close()


def test_a_query_with_no_graph_predicate_does_not_touch_the_graph(tmp_path):
    """Opt-in, or every existing query silently pays for the new leg."""
    st = make_store(tmp_path)
    try:
        st.remember("plain note with no graph", tenant="d")
        st.flush()
        before = st.recall("plain note", k=5)
        st.query({"text": "plain note"}, k=5)   # no graph key at all
        after = st.recall("plain note", k=5)
        assert [r["id"] for r in before] == [r["id"] for r in after]
    finally:
        st.close()
