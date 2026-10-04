"""A purged memory's embedding must not survive in the vector sidecar.

The first audit's R7 claimed `purge` never calls `index.remove(...)` and that the
HNSW graph keeps a tombstoned entry for an erased id. Reproduced here on the
HNSW backend specifically — the brute backend re-writes its whole `.npz` from
live rows on flush, so the vector does disappear there:

    kind: hnsw  count before: 1
    count after purge:   0      <- active_count() is right
    sidecar size: 2288 -> 2288  <- the file is unchanged
    tombstones: [1]             <- the id is still tracked

So the *reported* count is correct and the *stored* embedding is not: a
soft-delete tombstone marks the vector dead, but purge removes the row without
marking the vector dead, so the raw float data for an erased memory stays on
disk indefinitely. The audit's numbers on sidecar size were right; its framing
("the hnsw graph still holds a tombstoned entry") described a different bug,
which this does not assert.

`purge` is the operation that promises erasure, so this is the same class as the
WAL and secure_delete gaps in test_erasure_completeness.py.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from conftest import make_store

TOKEN = "hnsw-erasure-probe"

hnswlib = pytest.importorskip("hnswlib", reason="hnswlib is an optional extra")


@pytest.fixture()
def hnsw_store(tmp_path):
    """A real HNSW-backed store; the brute backend masks this by rewriting."""
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    from omnidb_server.vector_index import HnswlibIndex

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="hnsw")
    conn = dbmod.connect(tmp_path)
    index = HnswlibIndex(tmp_path / "index.hnsw", 512, count_hint=16)
    st = MemoryStore(conn, HashingEmbedder(), index, cfg)
    try:
        yield st
    finally:
        st.close()


def test_purge_marks_the_purged_vector_dead(hnsw_store, tmp_path):
    """After purge, no vector for the erased id may remain live or tracked.

    `active_count()` already reads 0, which is what made this look fine. The
    gap is that the vector is neither removed nor tombstoned — it is simply
    unreferenced, and stays on disk until an unrelated compaction reclaims it.
    """
    mid = hnsw_store.remember(f"SECRET {TOKEN} content to erase")["id"]
    hnsw_store.flush()
    assert hnsw_store.index.active_count() == 1

    hnsw_store.forget(mid)
    assert hnsw_store.purge(do_vacuum=True)["purged"] == 1
    hnsw_store.flush()

    tombstone_file = tmp_path / "index.hnsw.deleted.json"
    tombstones = (json.loads(tombstone_file.read_text())
                  if tombstone_file.exists() else [])
    assert mid not in tombstones, (
        f"purged id {mid} is still tracked as a live HNSW tombstone "
        f"{tombstones}; its embedding is on disk for an erased memory"
    )
    assert hnsw_store.index.active_count() == 0


def test_the_sidecar_shrinks_after_purge(hnsw_store, tmp_path):
    """The strongest available signal that the bytes are gone.

    HNSW never shrinks in place, so a rebuild (or compaction) is the mechanism.
    Either the file is smaller, or purge explicitly marks the index for one —
    what it must not do is leave the id neither removed nor tracked.
    """
    mid = hnsw_store.remember(f"SECRET {TOKEN} content")["id"]
    hnsw_store.flush()
    sidecar = tmp_path / "index.hnsw"
    before = sidecar.stat().st_size

    hnsw_store.forget(mid)
    hnsw_store.purge(do_vacuum=True)
    hnsw_store.flush()

    after = sidecar.stat().st_size
    compaction_due = hnsw_store.conn.execute(
        "SELECT value FROM meta WHERE key = 'compaction_due'").fetchone()
    assert (after < before) or (compaction_due and compaction_due[0] == "1"), (
        f"purge left the HNSW sidecar at {after} bytes (was {before}) with no "
        "compaction scheduled; the erased embedding is still on disk and "
        "nothing will reclaim it"
    )


def test_the_erased_content_is_not_retrievable_after_purge(hnsw_store):
    """The control: purge already worked at the product level.

    Whatever is wrong with the sidecar, the memory must not come back.
    """
    mid = hnsw_store.remember(f"SECRET {TOKEN} content")["id"]
    hnsw_store.flush()
    hnsw_store.forget(mid)
    hnsw_store.purge(do_vacuum=True)
    assert hnsw_store.recall(TOKEN, k=5) == []
    assert hnsw_store.get(mid) is None


def test_the_brute_backend_does_not_leak_and_must_keep_working(tmp_path):
    """Brute re-writes from live rows, so this must hold there too — pinned
    so a future change to either backend cannot regress the other."""
    st = make_store(tmp_path)
    try:
        mid = st.remember(f"SECRET {TOKEN} content")["id"]
        st.flush()
        st.forget(mid)
        st.purge(do_vacuum=True)
        st.flush()
        assert st.index.active_count() == 0
        assert st.recall(TOKEN, k=5) == []
    finally:
        st.close()


def test_purge_still_leaves_untouched_memories_indexed(hnsw_store):
    """The control that matters most: erasing one must not empty the index."""
    keep = hnsw_store.remember("a memory that should survive")["id"]
    drop = hnsw_store.remember(f"SECRET {TOKEN} content")["id"]
    hnsw_store.flush()
    assert hnsw_store.index.active_count() == 2

    hnsw_store.forget(drop)
    hnsw_store.purge(do_vacuum=True)
    hnsw_store.flush()

    assert hnsw_store.get(keep) is not None
    assert hnsw_store.index.active_count() == 1, (
        "purging one memory emptied the whole index"
    )
    _ = pathlib  # imported for the docstring's fixture paths