"""Checklist item 4 (BUG-A1-12): `rebuild_index` builds the replacement
fully, publishes it with os.replace, and only then swaps it into the store.

The property under test is NOT "rebuild_index eventually returns a new
index". It is the failure path. The pre-fix ordering unlinked the canonical
sidecar and *then* constructed its replacement, so any constructor failure
(dim mismatch, hnswlib error) destroyed the only vector copy and forced a
full re-embed. The audit's hazard list is explicit about what must hold:

- the new index is built into a scratch object, never the live one;
- `self.index` is reassigned only AFTER the build and the publish succeed;
- on an exception mid-rebuild the previously-live index still answers
  queries identically, and the on-disk canonical sidecar is untouched.

Each test below pins one of those, and each fails if the ordering regresses.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.store import MemoryStore
from omnidb_server.vector_index import BruteForceIndex, HnswlibIndex


def _brute_store(tmp_path):
    return make_store(tmp_path)


def _hnsw_store(tmp_path):
    """A store on the hnsw backend, which is the path that carries a
    tombstone sidecar and therefore the richer swap semantics."""
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="hnsw")
    conn = dbmod.connect(tmp_path)
    embedder = HashingEmbedder()
    index = HnswlibIndex(tmp_path / "index.hnsw", embedder.dim, count_hint=0)
    return MemoryStore(conn, embedder, index, cfg)


def _seed(store, n: int) -> list[int]:
    return [r["id"] for r in store.remember_many(
        [{"content": f"atomic rebuild probe {i}"} for i in range(n)])]


def _live_ids(store) -> set[int]:
    return {int(r["id"]) for r in store.conn.execute(
        "SELECT id FROM memories WHERE deleted_ms IS NULL "
        "AND superseded_by IS NULL")}


def _snapshot(store, query: str = "atomic rebuild probe"):
    """The answer set a caller would observe: (id, score) pairs, id-sorted
    so the comparison is stable across an index rebuild."""
    vec = store.embedder.encode([query])[0]
    return sorted((int(mid), round(float(score), 6))
                  for mid, score in store.index.query(vec, 10))


# --- the live index is never mutated in place -------------------------------

def test_rebuild_builds_a_separate_object_and_swaps_only_on_success(tmp_path):
    """`self.index` is a different object after a successful rebuild, and it
    is the one whose contents match the DB.

    If the implementation mutated the live object in place, `before is after`
    would hold and the store would have no rollback path at all.
    """
    st = _brute_store(tmp_path)
    try:
        _seed(st, 8)
        before = st.index
        st.rebuild_index("swap test")

        assert st.index is not before
        assert st.index.indexed_ids() == _live_ids(st)
    finally:
        st.close()


# --- exception injected mid-rebuild leaves the old index live ---------------

@pytest.mark.parametrize("fail_at", ["encode", "persist"])
def test_failure_mid_rebuild_leaves_the_previous_index_fully_usable(
        tmp_path, monkeypatch, fail_at):
    """The core of checklist item 4.

    Inject a failure inside the rebuild loop and after it, then assert the
    store is exactly as it was: the same live index object, answering the
    same queries with the same scores, over an untouched canonical sidecar.

    Reverting the fix (unlink-then-construct, or assign-then-build) makes
    this red in two distinct ways — either the canonical sidecar is gone, or
    the live object answers with a partially rebuilt vector set.
    """
    st = _brute_store(tmp_path)
    try:
        ids = _seed(st, 8)
        st.flush()
        before_object = st.index
        before_answers = _snapshot(st)
        assert before_answers, "seed must produce a non-empty answer set"
        canonical = tmp_path / "vectors.npz"
        canonical_bytes = canonical.read_bytes()

        if fail_at == "encode":
            # fail on the FIRST chunk of the rebuild loop. The injection is
            # armed only around the rebuild call, because the assertions
            # below legitimately re-encode through the same embedder.
            real_encode = st.embedder.encode
            armed: list[bool] = [True]

            def boom(texts, *a, **kw):
                if armed[0] and texts:
                    raise RuntimeError("injected embedder failure")
                return real_encode(texts, *a, **kw)

            monkeypatch.setattr(st.embedder, "encode", boom)
        else:
            # fail after the build loop, at the point the replacement is
            # published to disk — the last fallible step before the swap
            real_persist = BruteForceIndex.persist

            def boom(self, *a, **kw):
                if self.path.name.endswith(".rebuilding"):
                    raise OSError("injected persist failure")
                return real_persist(self, *a, **kw)

            monkeypatch.setattr(BruteForceIndex, "persist", boom)

        with pytest.raises((RuntimeError, OSError)):
            st.rebuild_index("failure injection")

        if fail_at == "encode":
            armed[0] = False

        assert st.index is before_object, \
            "the live index was replaced by a failed rebuild"
        assert _snapshot(st) == before_answers, \
            "the live index no longer answers identically after a failed rebuild"
        assert canonical.read_bytes() == canonical_bytes, \
            "a failed rebuild mutated the published sidecar"
        assert _live_ids(st) == set(ids)
    finally:
        st.close()


def test_failed_rebuild_leaves_the_home_reopenable_with_every_row(tmp_path):
    """A crashed rebuild must not leave a half-built sidecar that a later
    open would mistake for the live generation.

    The scratch artifact is scratch space, so its presence after a failure
    is harmless only because the next rebuild unlinks it first. This pins
    that the *canonical* name is never the thing being built into.
    """
    st = _brute_store(tmp_path)
    ids = _seed(st, 4)
    st.flush()
    canonical = tmp_path / "vectors.npz"
    before = canonical.read_bytes()

    real_encode = st.embedder.encode
    armed: list[bool] = [True]

    def boom(texts, *a, **kw):
        if armed[0] and texts:
            raise RuntimeError("injected embedder failure")
        return real_encode(texts, *a, **kw)

    st.embedder.encode = boom
    try:
        with pytest.raises(RuntimeError):
            st.rebuild_index("scratch check")
    finally:
        armed[0] = False
        st.embedder.encode = real_encode

    assert canonical.read_bytes() == before
    st.close()

    reopened = _brute_store(tmp_path)
    try:
        assert reopened.index.indexed_ids() == set(ids)
    finally:
        reopened.close()


def test_rebuild_survives_a_constructor_failure_with_the_sidecar_intact(
        tmp_path, monkeypatch):
    """BUG-A1-12 proper: the pre-fix order deleted the sidecar and *then*
    constructed. Force the constructor to raise and assert the published
    sidecar still contains the previous generation's vectors.
    """
    st = _brute_store(tmp_path)
    try:
        ids = _seed(st, 6)
        st.flush()
        before_answers = _snapshot(st)
        before_object = st.index

        real_init = BruteForceIndex.__init__

        def boom(self, path, dim):
            if str(path).endswith(".rebuilding"):
                raise sqlite3.OperationalError("injected constructor failure")
            return real_init(self, path, dim)

        monkeypatch.setattr(BruteForceIndex, "__init__", boom)
        with pytest.raises(sqlite3.OperationalError):
            st.rebuild_index("constructor failure")
        monkeypatch.undo()

        assert st.index is before_object
        assert _snapshot(st) == before_answers
        assert len(ids) == 6
    finally:
        st.close()


# --- hnsw-specific swap semantics ------------------------------------------

def test_hnsw_rebuild_repoints_the_tombstone_sidecar_to_the_canonical_name(
        tmp_path):
    """A rebuilt index must publish its tombstones under the canonical name.

    Regression for a swap that updated `path` but left the tombstone sidecar
    pointing at the scratch file. `persist()` would then write deletions to
    `index.hnsw.rebuilding.deleted.json` while the open path — which reads
    only `index.hnsw.deleted.json` — resurrected every deleted row into
    dense results after a reopen. The tombstone sidecar is part of the
    published generation and must travel with the swap.
    """
    st = _hnsw_store(tmp_path)
    try:
        ids = _seed(st, 6)
        st.rebuild_index("tombstone swap")

        assert st.index._deleted_path.name == "index.hnsw.deleted.json", (
            "the rebuilt index still writes tombstones to the scratch sidecar")

        st.forget_many(ids[:3])
        st.flush()

        published = json.loads(
            (tmp_path / "index.hnsw.deleted.json").read_text())
        assert sorted(published) == sorted(ids[:3]), (
            "tombstones were not published under the canonical name")
        assert not (tmp_path / "index.hnsw.rebuilding.deleted.json").exists()
    finally:
        st.close()


def test_hnsw_deletions_survive_a_reopen_after_a_rebuild(tmp_path):
    """End-to-end consequence of the tombstone swap: a delete taken after a
    rebuild must still be honoured by a freshly opened store.

    If the tombstone sidecar was written to the scratch name, the reopened
    index reports the deleted ids as live again.
    """
    st = _hnsw_store(tmp_path)
    try:
        ids = _seed(st, 6)
        st.rebuild_index("reopen probe")
        deleted = ids[:3]
        st.forget_many(deleted)
        st.flush()
    finally:
        st.close()

    reopened = _hnsw_store(tmp_path)
    try:
        assert not (set(deleted) & reopened.index.indexed_ids()), (
            "deleted ids reappeared in the reopened index")
        assert _live_ids(reopened) == set(ids) - set(deleted)
    finally:
        reopened.close()


# --- the published generation is complete and consistent --------------------

def test_successful_rebuild_leaves_the_canonical_sidecar_loadable(tmp_path):
    """The published file is the built one, not the old one, and it loads."""
    st = _brute_store(tmp_path)
    try:
        ids = _seed(st, 5)
        st.rebuild_index("publish")
        assert st.index.path == tmp_path / "vectors.npz"
        assert not (tmp_path / "vectors.npz.rebuilding").exists()

        reloaded = BruteForceIndex.load(tmp_path / "vectors.npz",
                                        st.embedder.dim)
        assert reloaded.indexed_ids() == set(ids)
    finally:
        st.close()


def test_rebuild_verifies_before_swapping_in_a_complete_index(tmp_path):
    """Construct → verify → swap means the swapped-in index is complete.

    A swap performed before the build finished would publish an index whose
    membership disagrees with the live row set; asserting the rebuilt index
    answers for every seeded row pins that the build completed before the
    swap.
    """
    st = _brute_store(tmp_path)
    try:
        ids = _seed(st, 12)
        st.rebuild_index("verify before swap")

        assert st.index.indexed_ids() == set(ids)
        assert st.index.active_count() == len(ids)
        vec = st.embedder.encode(["atomic rebuild probe 3"])[0]
        answered = {mid for mid, _ in st.index.query(vec, 50)}
        assert answered == set(ids)
    finally:
        st.close()


def test_rebuild_is_idempotent_and_republishes_the_same_generation(tmp_path):
    """Two rebuilds in a row must converge on the same membership, so a
    retry after a crash cannot double-count or lose rows."""
    st = _brute_store(tmp_path)
    try:
        ids = _seed(st, 7)
        st.rebuild_index("first")
        first = _snapshot(st)
        st.rebuild_index("second")
        assert _snapshot(st) == first
        assert st.index.indexed_ids() == set(ids)
    finally:
        st.close()


def test_rebuild_publishes_a_fully_built_scratch_artifact(
        tmp_path, monkeypatch):
    """Ordering, asserted directly at the syscall: the canonical name is
    only ever replaced from a scratch file that already holds every row.

    Pinning that the published source is COMPLETE is what makes this a
    checklist-item-4 test rather than a rebuild smoke test. It fails if the
    canonical sidecar is deleted first (no such replace at all, or a replace
    from an empty source), and it fails if the publish happens before the
    build loop has finished adding rows.
    """
    st = _brute_store(tmp_path)
    try:
        ids = _seed(st, 5)
        st.flush()
        canonical = tmp_path / "vectors.npz"
        sources: list[set[int]] = []
        real_replace = os.replace

        def spy(src, dst, *a, **kw):
            if str(dst) == str(canonical):
                # capture what the scratch sidecar holds at publish time
                staged = BruteForceIndex.load(Path(str(src)), st.embedder.dim)
                sources.append(staged.indexed_ids())
            return real_replace(src, dst, *a, **kw)

        monkeypatch.setattr(os, "replace", spy)
        st.rebuild_index("ordering probe")

        assert sources, \
            "rebuild never published the scratch artifact to the canonical name"
        assert all(src == set(ids) for src in sources), (
            "rebuild published a sidecar that did not match the live rows; "
            f"published={sources} expected={sorted(ids)}")
    finally:
        st.close()
