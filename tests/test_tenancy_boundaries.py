"""ADR-08: the tenant boundary, as originally pinned by MT-02.

The original four bypasses were recorded as `xfail(strict=True)` under the name
MT-06, when `agent_id` was held to be the boundary. ADR-08 replaced `agent_id`
with an authenticated `tenant`, and three of the four are now enforced by the
store. The fourth — a single shared vector sidecar per home — still awaits the
per-tenant HNSW work (W3-3's remaining half), and stays xfail until then.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from conftest import make_store


@pytest.fixture
def tenancy_store(tmp_path: Path) -> Iterator:
    store = make_store(tmp_path, agent_overfetch=1.0)
    yield store
    store.close()


def test_recall_dense_leg_filters_candidates_by_tenant(tenancy_store):
    """The dense (ANN) leg is fenced: a scoped recall returns only its tenant."""
    alpha = tenancy_store.remember("shared deployment marker", tenant="acme")["id"]
    tenancy_store.remember("shared deployment marker", tenant="globex")

    acme_hits = tenancy_store.recall("shared deployment marker", k=5, tenant="acme")
    assert {h["id"] for h in acme_hits} == {alpha}

    globex_hits = tenancy_store.recall("shared deployment marker", k=5,
                                       tenant="globex")
    assert alpha not in {h["id"] for h in globex_hits}


def test_related_walk_stays_within_requesting_tenant(tenancy_store):
    victim = tenancy_store.remember("globex private fact", tenant="globex")["id"]
    attacker = tenancy_store.remember(
        "acme public note", tenant="acme"
    )["id"]
    # even a hand-forged cross-tenant edge is not followed by the walk
    tenancy_store.conn.execute(
        "INSERT INTO edges(src, dst, label, created_ms) VALUES (?, ?, ?, ?)",
        (attacker, victim, "forged-edge", 1),
    )
    tenancy_store.conn.commit()

    neighbors = tenancy_store.related(attacker, max_hops=1, tenant="acme")
    assert victim not in {row["id"] for row in neighbors}


def test_stats_hides_other_tenants_and_their_volumes(tenancy_store):
    tenancy_store.remember("acme fact", tenant="acme")
    tenancy_store.remember("globex fact one", tenant="globex")
    tenancy_store.remember("globex fact two", tenant="globex")

    acme = tenancy_store.stats("acme")
    globex = tenancy_store.stats("globex")

    assert acme["tenant"] == "acme"
    assert acme["live"] == 1          # one acme row, not three
    assert globex["live"] == 2


def _exposed_stats_via_tool(store, tenant="default"):
    # the MCP tool resolves the tenant from the auth context; here we exercise
    # the store path directly with an explicit tenant, which is what the server
    # layer calls.
    return store.stats(tenant)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MT-06 (remaining): per-tenant vector sidecars. The dense leg is fenced "
        "in recall, and the sidecar is only reachable by a co-located filesystem "
        "actor, not by a tenant — but the vectors still live in one shared file, "
        "so a tenant with filesystem access to the home can read every tenant's "
        "embeddings at rest. Closes when per-tenant HNSW/BruteForce sidecars land."
    ),
)
def test_vector_sidecar_exposes_only_requesting_tenants_vectors(tenancy_store):
    tenant_a = tenancy_store.remember("acme vector", tenant="acme")["id"]
    tenancy_store.remember("globex vector", tenant="globex")
    tenancy_store.flush()
    sidecar = tenancy_store.index.path

    with np.load(sidecar) as data:
        persisted_ids = {int(m) for m in data["ids"]}

    assert persisted_ids == {tenant_a}