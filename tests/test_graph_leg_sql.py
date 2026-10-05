"""The graph leg's SQL half: fetching a fenced adjacency, and fusing the leg.

`omnidb_server/graph_leg.py` holds the pure math. This file covers the part that
touches the database, where the failure modes are different in kind: the tenant
fence (ADR-08), the `_SELECTIVE` filter, scope predicates, and cost bounds.

The fence is the reason these tests exist. `edges` has no `tenant_id` column —
migration 4 added one to `memories` and `queries` only — so every statement must
join `memories` twice and fence *both* endpoints. That is stricter than
`related()`, which fences only the neighbour, and the reason the xfail this
leg eventually removes could not be closed by a predicate on one side.
"""

from __future__ import annotations

import sqlite3
import time

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import open_vector_index


def _store(tmp_path, **cfg):
    cfg_obj = Config(home=tmp_path, embedder="hashing", vector_index="brute", **cfg)
    conn = dbmod.connect(tmp_path)
    embedder = HashingEmbedder()
    index = open_vector_index(tmp_path, "brute", embedder.dim, 0)
    return MemoryStore(conn, embedder, index, cfg_obj)


def _linked(tmp_path):
    """A home where a text match's neighbours are NOT text-matchable.

    If they were, a passing test could not distinguish a working graph leg from
    no graph leg at all. Yields `(ids, store)` as one object so the caller can
    iterate the generator without unpacking confusion.
    """
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        seed = st.remember("kubernetes ingress controller setup")["id"]
        mid = st.remember("qqqq vvvvvv blorp")["id"]          # no shared term
        st.link(seed, mid, label="depends_on")
        yield {"_store": st, "seed": seed, "mid": mid}
    finally:
        st.close()


# --- the leg contributes ------------------------------------------------------

def test_a_text_unreachable_neighbour_appears_when_the_leg_is_on(tmp_path):
    """The headline capability: find by content, then traverse.

    `mid` shares no searchable term with the query. Today `recall()` cannot
    return it, because the graph is reachable only by id.
    """
    for ids in _linked(tmp_path):
        ranked = ids["_store"].graph_ranking([ids["seed"]], kind=None,
                                             agent_id=None, tenant="default",
                                             depth=2, budget=64)
        assert ids["mid"] in ranked, (
            "the graph leg did not return a neighbour that no text leg could "
            "find; the leg is either not wired or the adjacency fetch is fenced "
            "too tightly"
        )


def test_the_leg_respects_its_depth_bound(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha")["id"]
        b = st.remember("hop one bravo")["id"]
        c = st.remember("hop two charlie")["id"]
        st.link(a, b, label="x")
        st.link(b, c, label="x")
        one = st.graph_ranking([a], kind=None, agent_id=None, tenant="default",
                               depth=1, budget=64)
        two = st.graph_ranking([a], kind=None, agent_id=None, tenant="default",
                               depth=2, budget=64)
        assert b in one, "depth 1 must reach the immediate neighbour"
        assert c not in one, "depth 1 must not reach two hops"
        assert c in two, "depth 2 must reach two hops"
    finally:
        st.close()


def test_a_seed_is_never_returned_by_its_own_leg(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha")["id"]
        b = st.remember("neighbour bravo")["id"]
        st.link(a, b, label="x")
        ranked = st.graph_ranking([a], kind=None, agent_id=None, tenant="default",
                                  depth=2, budget=64)
        assert a not in ranked, (
            "a seed was re-emitted; the leg extends the seed set, it does not "
            "confirm it"
        )
    finally:
        st.close()


# --- tenant fencing: the reason this file exists ------------------------------

def test_a_forged_cross_tenant_edge_is_not_followed(tmp_path):
    """`link()` refuses a cross-tenant edge; direct SQL does not.

    The fence has to hold against a home with a bad historical edge, which is
    exactly the shape `link()`'s guard cannot cover. A new query path is the
    easiest place to drop a fence, so this is asserted rather than assumed.
    """
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        acme = st.remember("shared marker text", tenant="acme")["id"]
        globex = st.remember("shared marker text", tenant="globex")["id"]
        st.conn.execute(
            "INSERT INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, 0)",
            (acme, globex, "forged"))
        st.conn.commit()

        ranked = st.graph_ranking([acme], kind=None, agent_id=None,
                                  tenant="acme", depth=3, budget=64)
        assert globex not in ranked, (
            "the graph leg followed a forged cross-tenant edge; edges has no "
            "tenant_id column, so both endpoints must be fenced"
        )
    finally:
        st.close()


def test_a_forged_edge_is_not_followed_in_either_direction(tmp_path):
    """The fence is symmetric: a foreign row must not reach in either."""
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        acme = st.remember("shared marker text", tenant="acme")["id"]
        globex = st.remember("shared marker text", tenant="globex")["id"]
        st.conn.execute(
            "INSERT INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, 0)",
            (globex, acme, "forged"))
        st.conn.commit()
        ranked = st.graph_ranking([acme], kind=None, agent_id=None,
                                  tenant="acme", depth=3, budget=64)
        assert globex not in ranked
    finally:
        st.close()


# --- dead rows ----------------------------------------------------------------

def test_a_soft_deleted_neighbour_is_not_returned(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha")["id"]
        b = st.remember("neighbour bravo")["id"]
        st.link(a, b, label="x")
        st.forget(b, tenant="default")
        ranked = st.graph_ranking([a], kind=None, agent_id=None, tenant="default",
                                  depth=2, budget=64)
        assert b not in ranked, "a soft-deleted row was surfaced by the graph leg"
    finally:
        st.close()


def test_a_dead_row_is_not_a_mass_conduit(tmp_path):
    """A live node two hops away *through* a dead row is unreachable that way.

    Follows from `_SELECTIVE` on the far endpoint: the dead intermediate is
    never emitted, so its own neighbours never enter the adjacency.
    """
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha")["id"]
        dead = st.remember("hop one bravo")["id"]
        far = st.remember("hop two charlie")["id"]
        st.link(a, dead, label="x")
        st.link(dead, far, label="x")
        st.forget(dead, tenant="default")
        ranked = st.graph_ranking([a], kind=None, agent_id=None, tenant="default",
                                  depth=3, budget=64)
        assert far not in ranked, (
            "a dead row acted as a conduit: the walk reached past it"
        )
    finally:
        st.close()


# --- scope predicates ---------------------------------------------------------

def test_kind_scoping_excludes_a_procedural_neighbour(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha", kind="semantic")["id"]
        b = st.remember("neighbour bravo", kind="procedural")["id"]
        st.link(a, b, label="x")
        ranked = st.graph_ranking([a], kind="semantic", agent_id=None,
                                  tenant="default", depth=2, budget=64)
        assert b not in ranked, (
            "a kind-scoped recall returned a row of another kind; the scope "
            "predicates must apply to the neighbour, not just the seed"
        )
    finally:
        st.close()


def test_agent_scoping_excludes_another_agents_row(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha", agent_id="one")["id"]
        b = st.remember("neighbour bravo", agent_id="two")["id"]
        st.link(a, b, label="x")
        ranked = st.graph_ranking([a], kind=None, agent_id="one",
                                  tenant="default", depth=2, budget=64)
        assert b not in ranked, "an agent-scoped recall crossed agents"
    finally:
        st.close()


# --- degradation and bounds ---------------------------------------------------

def test_an_empty_graph_returns_empty_and_never_raises(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("lonely note")["id"]
        assert st.graph_ranking([a], kind=None, agent_id=None,
                                tenant="default", depth=2, budget=64) == []
    finally:
        st.close()


def test_an_empty_seed_set_is_not_an_empty_in_clause(tmp_path):
    """SQLite rejects `IN ()`; the leg must not build one."""
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("note")["id"]
        st.link(a, a, label="x")  # self-loop only
        assert st.graph_ranking([], kind=None, agent_id=None, tenant="default",
                                depth=2, budget=64) == []
    finally:
        st.close()


def test_a_sqlite_error_degrades_to_the_two_leg_result(tmp_path):
    """A failing adjacency query must not fail recall.

    Mirrors the C6 contract: the conversion is to a handled type so the caller
    falls back, rather than a raw error escaping into the read path.
    """
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha")["id"]
        b = st.remember("neighbour bravo")["id"]
        st.link(a, b, label="x")

        # sqlite3.Connection.execute is read-only, so substitute a proxy on
        # the store's connection attribute rather than the method itself.
        class _Failing:
            def __getattr__(self, name):
                return getattr(real_conn, name)

            def execute(self, sql, *args):
                if "FROM edges" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return real_conn.execute(sql, *args)

        real_conn = st.conn
        st.conn = _Failing()
        try:
            assert st.graph_ranking([a], kind=None, agent_id=None,
                                    tenant="default", depth=2, budget=64) == []
        finally:
            st.conn = real_conn
    finally:
        st.close()


def test_cost_is_flat_in_graph_size(tmp_path):
    """The property that makes this shippable where whole-graph PPR was not."""
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=128)
    try:
        a = st.remember("hub token")["id"]
        for i in range(2_000):
            st.link(a, st.remember(f"spoke token {i}")["id"], label="s")
        st.flush()
        t0 = time.perf_counter()
        ranked = st.graph_ranking([a], kind=None, agent_id=None,
                                  tenant="default", depth=2, budget=128)
        elapsed = time.perf_counter() - t0
        assert len(ranked) <= 128
        assert elapsed < 5.0, f"a 2,000-spoke hub took {elapsed:.1f}s"
    finally:
        st.close()


def test_a_hub_does_not_starve_a_weaker_seed(tmp_path):
    """The case per-seed budgets exist for, end to end through SQL."""
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        hub = st.remember("hub token")["id"]
        for i in range(60):
            st.link(hub, st.remember(f"spoke token {i}")["id"], label="s")
        private = st.remember("private chain")["id"]
        deep = st.remember("private chain two")["id"]
        st.link(private, deep, label="p")
        st.flush()

        ranked = st.graph_ranking([hub, private], kind=None, agent_id=None,
                                  tenant="default", depth=2, budget=32)
        assert deep in ranked, (
            "a single global budget drained into the hub's spokes and never "
            "reached the weaker seed's private chain"
        )
    finally:
        st.close()


def test_repeated_calls_are_identical(tmp_path):
    st = _store(tmp_path, graph_weight=0.5, graph_expand_budget=64)
    try:
        a = st.remember("seed alpha")["id"]
        for i in range(5):
            st.link(a, st.remember(f"nbr {i} bravo")["id"], label="x")
        st.flush()
        runs = [st.graph_ranking([a], kind=None, agent_id=None,
                                 tenant="default", depth=2, budget=64)
                for _ in range(10)]
        assert all(r == runs[0] for r in runs)
    finally:
        st.close()