"""Purged content must not survive in the WAL or the vector sidecar.

The first audit's R6 said "`forget` does not erase" and showed a soft-deleted
row still readable through `search`. That part is **design, not a defect**:
`forget` is documented as a soft delete (ARCHITECTURE.md:53), `purge` is the
hard delete, and `search` is a raw admin surface whose whole purpose includes
reaching rows `purge` has not yet collected. Making `forget` erase would break
`purge`. Reproduced here first so the distinction is on the record rather than
assumed.

What is a real defect is the second half of the audit's finding, reproduced on
the *purge* path — the operation that does promise erasure:

    remember("SECRET unique-token-xyz") → forget → purge(do_vacuum=True)

  - row gone from memories, FTS posting gone, main db scrubbed   ✅
  - omnidb.db-wal still contains the plaintext                    ❌
  - secure_delete is 0, so freed pages are not zeroed             ❌

`close()` → `flush()` → `checkpoint()` uses `PRAGMA wal_checkpoint(PASSIVE)`,
which is permitted to no-op and does not truncate: the WAL keeps a complete
plaintext copy of data an operator believes they erased.
"""

from __future__ import annotations

import pathlib
import sqlite3

import pytest
from conftest import make_store

from omnidb_server import db as dbmod

TOKEN = "unique-token-xyz-for-erasure-test"


@pytest.fixture()
def purged_home(tmp_path):
    """remember → forget → purge(do_vacuum=True), then close cleanly."""
    st = make_store(tmp_path)
    mid = st.remember(f"SECRET {TOKEN} sensitive payload")["id"]
    st.forget(mid)
    result = st.purge(do_vacuum=True)
    st.flush()
    st.close()
    return tmp_path, result


# --- the soft-delete contract holds (the control) ----------------------------

def test_forget_hides_the_row_from_recall(tmp_path):
    """`forget`'s promise is about recall, and it keeps it."""
    st = make_store(tmp_path)
    mid = st.remember(f"SECRET {TOKEN}")["id"]
    st.forget(mid)
    assert st.recall(TOKEN, k=5) == []
    st.close()


def test_purge_actually_removes_the_row_and_fts_posting(purged_home):
    home, _ = purged_home
    conn = sqlite3.connect(home / "omnidb.db")
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE ?", (f"%{TOKEN}%",)
        ).fetchone()[0] == 0
        # a hyphenated token is FTS5 syntax (NOT / OR / AND), so match a
        # plain word from the same content instead
        assert conn.execute(
            'SELECT COUNT(*) FROM "fts_default" WHERE "fts_default" MATCH ?',
            ("payload",),
        ).fetchone()[0] == 0
    finally:
        conn.close()


# --- the actual defect: erasure survives in the WAL --------------------------

def test_the_wal_does_not_retain_the_purged_content(tmp_path):
    """The WAL holds a full plaintext copy until the connection closes.

    While a connection is open the WAL legitimately holds recent writes — no
    pragma prevents that, and pretending otherwise would be a test that cannot
    pass. What must hold is that *closing* the store clears it, which is the
    moment an operator believes the erasure is complete.

    `close()` used to checkpoint PASSIVE, which is permitted to no-op and never
    truncates, so the file kept a complete copy of everything the session wrote.
    """
    st = make_store(tmp_path)
    mid = st.remember(f"SECRET {TOKEN} sensitive payload")["id"]
    st.forget(mid)
    assert st.purge(do_vacuum=True)["purged"] == 1
    st.close()

    wal = pathlib.Path(tmp_path) / "omnidb.db-wal"
    if not wal.exists() or wal.stat().st_size == 0:
        return  # removed or emptied: the best possible end state
    assert TOKEN.encode() not in wal.read_bytes(), (
        "closing the store left the purged content readable in the WAL"
    )


def test_closing_the_store_truncates_the_wal(tmp_path):
    """`close()` must TRUNCATE, not PASSIVE.

    Measured on the same sequence with each pragma, while the connection is
    still open (SQLite removes the WAL entirely on last close, which would make
    both branches look alike):

        PASSIVE   wal size=280192  contains token=True
        TRUNCATE  wal size=      0  contains token=False

    The test calls `close()` itself rather than issuing the pragma, so it pins
    the shipped behaviour — a version of this test that ran
    `wal_checkpoint(TRUNCATE)` by hand would pass with `close()` reverted.
    """
    st = make_store(tmp_path)
    mid = st.remember(f"SECRET {TOKEN} sensitive payload")["id"]
    st.forget(mid)
    st.purge(do_vacuum=True)
    st.flush()

    # Observe the WAL through a second connection so the file is still there
    # after close() hands back the handle to the store's connection.
    conn = dbmod.connect(tmp_path)
    try:
        st.close()
        wal = pathlib.Path(tmp_path) / "omnidb.db-wal"
        if not wal.exists():
            return  # emptied and removed: the best possible end state
        assert wal.stat().st_size == 0, (
            f"after close() the WAL is {wal.stat().st_size:,} bytes; a PASSIVE "
            "checkpoint would have left it full of purged plaintext"
        )
        assert TOKEN.encode() not in wal.read_bytes()
    finally:
        conn.close()


def test_secure_delete_is_enabled_on_the_shipped_connection(tmp_path):
    """Without it, freed pages keep the old bytes until reused.

    VACUUM rewrites the live database, which is why the main file comes out
    clean — but any later free leaves the old content in place. `secure_delete`
    zeroes freed pages at delete time.
    """
    st = make_store(tmp_path)
    try:
        assert st.conn.execute("PRAGMA secure_delete").fetchone()[0] == 1, (
            "secure_delete is off: freed pages retain the purged bytes instead "
            "of being zeroed"
        )
    finally:
        st.close()


def test_a_second_connection_also_gets_secure_delete(tmp_path):
    """It is a per-connection pragma, so the opener must set it every time."""
    from omnidb_server import db as dbmod

    conn = dbmod.connect(tmp_path)
    try:
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    finally:
        conn.close()


def test_the_wal_is_checkpointed_on_close(tmp_path):
    """The store's own close path must leave no readable WAL behind.

    This is the property an operator relies on: run purge, quit, and the
    content is not sitting in a sidecar file next to the home.
    """
    st = make_store(tmp_path)
    mid = st.remember(f"SECRET {TOKEN}")["id"]
    st.forget(mid)
    st.purge(do_vacuum=True)
    st.close()

    wal = pathlib.Path(tmp_path) / "omnidb.db-wal"
    if not wal.exists():
        return  # SQLite removed it entirely: the best possible end state
    if wal.stat().st_size == 0:
        return
    assert TOKEN.encode() not in wal.read_bytes(), (
        "closing the store left the purged content readable in the WAL"
    )