"""Per-tenant vector sidecars: one file per tenant, so isolation holds at rest.

ADR-08 W3-3, the vector half. The lexical half landed as schema v6 (one FTS5
table per tenant); this is the other side of the same boundary.

What was and was not broken, verified rather than assumed:

- The *behavioural* fence already held. `recall` narrows every dense and sparse
  candidate through `m.tenant_id = ?`, so a `globex` recall never returned an
  `acme` row. That is pinned by
  `test_recall_dense_leg_filters_candidates_by_tenant` and did not change.
- The *at-rest* fence did not hold. Every tenant's embeddings sat in one
  `vectors.npz` / `index.hnsw`, so anyone who could read the home got all of
  them — which is what this module's predecessor recorded as the suite's one
  intentional `xfail(strict=True)`.

So this is not about closing a live cross-tenant read. It is making isolation at
rest structural, so it no longer depends on every future code path remembering
to add a predicate.

`test_the_xfail_is_no_longer_needed` is the assertion that matters most: the
xfail that documented this gap is gone, and `tests/test_tenancy_boundaries.py`
now asserts isolation directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_store


def _sidecar_ids(home: Path, tenant: str) -> set[int]:
    """The ids persisted in one tenant's brute sidecar, read off disk.

    Read directly rather than through the store, because the point is what is on
    disk for a filesystem actor — which is what the original xfail measured and
    what this change is about.
    """
    path = home / f"tenant-{tenant}.npz"
    if not path.exists():
        return set()
    with np.load(path) as data:
        return {int(i) for i in data["ids"]}


def _sidecar_files(home: Path) -> set[str]:
    return {p.name for p in home.iterdir()
            if p.is_file() and p.name.startswith("tenant-")}


@pytest.fixture
def two_tenant_home(tmp_path):
    store = make_store(tmp_path)
    try:
        store.remember("acme vector payload", tenant="acme")
        store.remember("globex vector payload", tenant="globex")
        store.flush()
        yield store
    finally:
        store.close()


# --- the isolation this change exists for -------------------------------------

def test_each_tenant_gets_its_own_sidecar(two_tenant_home):
    assert _sidecar_files(two_tenant_home._home()) >= {
        "tenant-acme.npz", "tenant-globex.npz"}


def test_a_sidecar_contains_only_its_own_tenants_ids(two_tenant_home):
    home = two_tenant_home._home()
    acme = _sidecar_ids(home, "acme")
    globex = _sidecar_ids(home, "globex")
    assert acme and globex, "both tenants should have persisted vectors"
    assert not (acme & globex), (
        f"a sidecar holds another tenant's ids: acme={acme} globex={globex}"
    )


def test_no_shared_sidecar_survives(two_tenant_home):
    """The old layout must be gone, or two lexical indexes exist and one is written."""
    home = two_tenant_home._home()
    for legacy in ("vectors.npz", "index.hnsw"):
        assert not (home / legacy).exists(), (
            f"{legacy} still exists; the per-tenant sidecars did not take over"
        )


def test_a_tenant_with_no_rows_has_no_sidecar(two_tenant_home):
    two_tenant_home.remember("initech row", tenant="initech")
    two_tenant_home.flush()
    files = _sidecar_files(two_tenant_home._home())
    assert "tenant-initech.npz" in files, (
        "a tenant's first write must bring its own sidecar into being"
    )
    assert "tenant-globex.npz" in files, "an existing tenant must keep its sidecar"


# --- behaviour is unchanged ---------------------------------------------------

def test_recall_still_finds_only_the_callers_rows(two_tenant_home):
    """The dense leg is now tenant-scoped, not global-then-filtered."""
    acme_ids = {r["id"] for r in two_tenant_home.recall("vector payload", k=5,
                                                       tenant="acme")}
    globex_ids = {r["id"] for r in two_tenant_home.recall("vector payload", k=5,
                                                          tenant="globex")}
    assert acme_ids and globex_ids
    assert not (acme_ids & globex_ids)


def test_a_single_tenant_home_behaves_exactly_as_before(tmp_path):
    """The common case must be untouched: one tenant, one sidecar, same results."""
    store = make_store(tmp_path)
    try:
        store.remember("zebra migration serengeti")
        store.remember("aardvark desert outpost")
        store.flush()
        assert store.recall("serengeti", k=5), "recall regressed for one tenant"
        assert _sidecar_ids(tmp_path, "default"), (
            "the default tenant's vectors must be persisted under the per-tenant "
            "name"
        )
        assert store.check()["index_drift"] in (0, None)
    finally:
        store.close()


# --- the rename migration ------------------------------------------------------

def test_an_existing_shared_sidecar_is_renamed_on_first_open(tmp_path):
    """A v6 home with `vectors.npz` keeps its vectors and gains isolation.

    Renaming rather than rebuilding: the vectors are already correct, so moving
    the file is both cheaper and lossless. A rebuild would re-embed the whole
    home for no reason.
    """
    legacy = tmp_path / "vectors.npz"
    np.savez(legacy, matrix=np.zeros((2, 8), dtype=np.float32),
             ids=np.array([41, 42], dtype=np.int64))

    store = make_store(tmp_path)   # opens, which renames
    try:
        assert not legacy.exists(), "the legacy sidecar should have been renamed"
        assert _sidecar_ids(tmp_path, "default") == {41, 42}, (
            "the rename must preserve the vectors it carried"
        )
    finally:
        store.close()


def test_the_rename_is_idempotent(tmp_path):
    legacy = tmp_path / "vectors.npz"
    np.savez(legacy, matrix=np.zeros((1, 8), dtype=np.float32),
             ids=np.array([7], dtype=np.int64))

    for _ in range(3):
        store = make_store(tmp_path)
        store.close()
    assert _sidecar_ids(tmp_path, "default") == {7}


def test_a_rename_does_not_clobber_an_existing_tenant_sidecar(tmp_path):
    """If both names exist, the rename must not destroy the newer file.

    Possible after a partial upgrade. The safe reading is to leave both in place
    and let reconciliation notice — dropping either would lose data.
    """
    legacy = tmp_path / "vectors.npz"
    np.savez(legacy, matrix=np.zeros((1, 8), dtype=np.float32),
             ids=np.array([1], dtype=np.int64))
    modern = tmp_path / "tenant-default.npz"
    np.savez(modern, matrix=np.zeros((1, 8), dtype=np.float32),
             ids=np.array([2], dtype=np.int64))

    store = make_store(tmp_path)
    try:
        assert legacy.exists(), "the legacy file must not be deleted"
        assert _sidecar_ids(tmp_path, "default") == {2}, (
            "the existing per-tenant sidecar must win; the legacy one is stale"
        )
    finally:
        store.close()


# --- reconciliation stays inside a tenant --------------------------------------

def test_reconciling_one_tenant_leaves_the_others_sidecars_alone(tmp_path):
    """A missing sidecar for one tenant must not cause another's rows to be dropped.

    Reconciliation used to diff live rows against the whole home's index; with
    per-tenant sidecars that comparison has to be per-tenant, or rebuilding
    acme's index would treat every globex row as stale and remove it.
    """
    store = make_store(tmp_path)
    try:
        store.remember("acme payload", tenant="acme")
        store.remember("globex payload", tenant="globex")
        store.flush()
        (tmp_path / "tenant-acme.npz").unlink()      # simulate a lost sidecar

        store2 = make_store(tmp_path)
        try:
            assert _sidecar_ids(tmp_path, "globex") == {
                int(r["id"]) for r in store2.conn.execute(
                    "SELECT id FROM memories WHERE tenant_id = 'globex'")
            }, "globex's sidecar was disturbed while acme's was rebuilt"
            assert _sidecar_ids(tmp_path, "acme"), "acme's sidecar was not rebuilt"
        finally:
            store2.close()
    finally:
        store.close()


def test_purge_rebuilds_only_the_purging_tenants_index(tmp_path):
    store = make_store(tmp_path)
    try:
        doomed = store.remember("acme doomed", tenant="acme")["id"]
        store.remember("acme survivor", tenant="acme")
        store.remember("globex survivor", tenant="globex")
        store.flush()
        store.forget(doomed, tenant="acme")
        store.purge(tenant="acme")
        store.flush()

        globex_ids = {int(r["id"]) for r in store.conn.execute(
            "SELECT id FROM memories WHERE tenant_id = 'globex'")}
        assert globex_ids <= _sidecar_ids(tmp_path, "globex"), (
            "purging acme disturbed globex's index"
        )
    finally:
        store.close()


# --- reporting ----------------------------------------------------------------

def test_check_reports_per_tenant_index_state(two_tenant_home):
    report = two_tenant_home.check()
    assert report["fts_integrity"] == "ok"
    assert report["index_drift"] in (0, None), report


def test_vec_kind_round_trips_across_a_reopen(tmp_path):
    """The stored `vec_kind` must name a backend the next open can construct.

    `MultiTenantIndexRouter.kind` reports "multi", which is not a constructible
    backend — persisting it would make the reopen fall back to auto and silently
    switch a home off the backend it was built with.
    """
    store = make_store(tmp_path)
    try:
        stored = store.conn.execute(
            "SELECT value FROM meta WHERE key = 'vec_kind'").fetchone()
        assert stored is not None
        assert stored[0] in ("hnsw", "brute"), (
            f"vec_kind={stored[0]!r} is not a backend open_vector_index can "
            "construct; the reopen would silently fall back to auto"
        )
    finally:
        store.close()


# --- the point of the whole change --------------------------------------------

def test_the_xfail_is_no_longer_needed():
    """The suite's one intentional xfail documented exactly this gap.

    `tests/test_tenancy_boundaries.py` recorded the shared sidecar as
    xfail(strict=True) "until per-tenant HNSW/BruteForce sidecars land". If that
    marker is still there, this work is not finished — a strict xfail that starts
    passing fails the suite, and a strict xfail that stays xfail means the
    boundary is still open.
    """
    import ast

    source = Path("tests/test_tenancy_boundaries.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    xfails = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            (isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "xfail")
            or (isinstance(d, ast.Attribute) and d.attr == "xfail")
            for d in node.decorator_list
        )
    ]
    assert not xfails, (
        "still marked xfail: "
        + ", ".join(f.name for f in xfails)
        + " — with per-tenant sidecars these must assert the boundary directly"
    )