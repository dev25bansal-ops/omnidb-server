"""Schema v6: one FTS table per tenant, built from live rows, no global table.

ADR-08 W3-3. Until this migration the lexical leg was *fenced but not isolated*:
`recall` queried a single global `memory_fts` and narrowed with
`AND m.tenant_id = ?` on the joined row. That fence works — the tests below
pin it before and after — but it cannot give two things per-tenant tables can:

- bm25 scored against the whole corpus's term statistics, so one tenant's rows
  were ranked against every other tenant's document frequencies;
- an un-fenced path could reach another tenant's postings by forgetting a
  predicate.

`test_ranking_is_local_to_the_tenant` is the one that proves the migration was
worth doing: it fails against the shared table and passes against per-tenant
ones. Everything else here would pass either way, which is deliberate — they are
the invariants that must survive the change, not the change's justification.
"""

from __future__ import annotations

import sqlite3

import pytest
from conftest import make_store

from omnidb_server import db as dbmod
from omnidb_server.db import SCHEMA_VERSION
from omnidb_server.store import _SELECTIVE


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _fts_tables(conn) -> set[str]:
    """Real FTS5 tables only.

    An FTS5 table creates shadow tables alongside itself — `fts_acme_content`,
    `_data`, `_idx`, `_docsize`, `_config` — so a `name LIKE 'fts_%'` filter
    counts five tables per tenant. The `sql` column distinguishes them: the real
    one carries `CREATE VIRTUAL TABLE`.
    """
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND sql LIKE '%CREATE VIRTUAL TABLE%'")}


def _markers(conn) -> set[str]:
    return {r[0][len("tenant_fts:"):] for r in conn.execute(
        "SELECT key FROM meta WHERE key LIKE 'tenant_fts:%'")}


@pytest.fixture()
def multi_tenant_home(tmp_path):
    """A home with rows for two tenants plus a soft-deleted one."""
    st = make_store(tmp_path)
    try:
        st.remember("acme alpha document", tenant="acme")
        st.remember("acme beta document", tenant="acme")
        globex = st.remember("globex gamma document", tenant="globex")["id"]
        st.forget(globex, tenant="globex")
        st.flush()
    finally:
        st.close()
    return tmp_path


# --- the migration -----------------------------------------------------------

def test_schema_version_is_six():
    assert SCHEMA_VERSION == 6


def test_the_global_table_is_gone_after_migration(multi_tenant_home):
    conn = dbmod.connect(multi_tenant_home)
    try:
        assert "memory_fts" not in _tables(conn), (
            "memory_fts still exists; the migration did not drop it"
        )
    finally:
        conn.close()


def test_each_tenant_gets_its_own_table(multi_tenant_home):
    conn = dbmod.connect(multi_tenant_home)
    try:
        assert _fts_tables(conn) == {"fts_acme", "fts_globex"}
    finally:
        conn.close()


def test_every_tenant_with_rows_is_registered(multi_tenant_home):
    conn = dbmod.connect(multi_tenant_home)
    try:
        assert _markers(conn) == {"acme", "globex"}
    finally:
        conn.close()


def test_migration_backfills_every_row_including_soft_deleted(multi_tenant_home):
    """Soft-deleted rows must be indexed too.

    Migration 5 indexed everything — `INSERT INTO memory_fts ... SELECT id,
    content FROM memories` has no deleted filter — and the store relies on that:
    `forget` replaces the posting with the new content so the vector sidecar
    stays consistent. Dropping soft-deleted rows from the FTS would silently
    change what `forget`'s delete-then-insert does.
    """
    conn = dbmod.connect(multi_tenant_home)
    try:
        assert conn.execute('SELECT COUNT(*) FROM "fts_acme"').fetchone()[0] == 2
        assert conn.execute('SELECT COUNT(*) FROM "fts_globex"').fetchone()[0] == 1
    finally:
        conn.close()


def test_each_table_holds_only_its_own_tenants_rows(multi_tenant_home):
    conn = dbmod.connect(multi_tenant_home)
    try:
        acme_ids = {r[0] for r in conn.execute('SELECT rowid FROM "fts_acme"')}
        globex_ids = {r[0] for r in conn.execute('SELECT rowid FROM "fts_globex"')}
        assert not (acme_ids & globex_ids)
        for mid in acme_ids:
            owner = conn.execute(
                "SELECT tenant_id FROM memories WHERE id = ?", (mid,)).fetchone()[0]
            assert owner == "acme"
        for mid in globex_ids:
            owner = conn.execute(
                "SELECT tenant_id FROM memories WHERE id = ?", (mid,)).fetchone()[0]
            assert owner == "globex"
    finally:
        conn.close()


def test_migration_is_idempotent_on_reopen(multi_tenant_home):
    """Reopening must not rebuild or duplicate."""
    for _ in range(3):
        conn = dbmod.connect(multi_tenant_home)
        try:
            assert _fts_tables(conn) == {"fts_acme", "fts_globex"}
            assert conn.execute('SELECT COUNT(*) FROM "fts_acme"').fetchone()[0] == 2
        finally:
            conn.close()


def test_integrity_check_passes_on_every_migrated_table(multi_tenant_home):
    conn = dbmod.connect(multi_tenant_home)
    try:
        for table in sorted(_fts_tables(conn)):
            conn.execute(f'INSERT INTO "{table}"("{table}") VALUES(\'integrity-check\')')
    except sqlite3.Error as exc:
        pytest.fail(f"integrity-check failed after migration: {exc}")
    finally:
        conn.close()


# --- invariants that must survive the change ---------------------------------

def test_lexical_recall_still_finds_a_tenants_own_rows(multi_tenant_home):
    st = make_store(multi_tenant_home)
    try:
        hits = st.recall("alpha", k=5, tenant="acme")
        assert any("acme alpha" in r["content"] for r in hits)
    finally:
        st.close()


def test_lexical_recall_still_excludes_another_tenants_rows(multi_tenant_home):
    """The fence the migration replaces — pinned so it cannot silently regress."""
    st = make_store(multi_tenant_home)
    try:
        acme_ids = {r["id"] for r in st.recall("document", k=10, tenant="acme")}
        globex_rows = st.conn.execute(
            "SELECT id FROM memories WHERE tenant_id = 'globex'").fetchall()
        globex_ids = {r[0] for r in globex_rows}
        assert acme_ids, "acme got nothing back; the fence is not the problem"
        assert not (acme_ids & globex_ids), "acme's recall returned globex rows"
    finally:
        st.close()


def test_writes_after_migration_land_in_the_right_table(multi_tenant_home):
    """A new tenant created *after* the migration must get a table on demand."""
    st = make_store(multi_tenant_home)
    try:
        st.remember("globex delta document", tenant="globex")
        st.remember("initech epsilon document", tenant="initech")
        st.flush()
        assert st.conn.execute('SELECT COUNT(*) FROM "fts_globex"').fetchone()[0] == 2
        assert st.conn.execute('SELECT COUNT(*) FROM "fts_initech"').fetchone()[0] == 1
    finally:
        st.close()


def test_a_tenant_created_after_migration_is_registered(multi_tenant_home):
    st = make_store(multi_tenant_home)
    try:
        st.remember("initech epsilon", tenant="initech")
        assert "initech" in _markers(st.conn)
    finally:
        st.close()


def test_update_replaces_the_posting_in_the_right_table(multi_tenant_home):
    st = make_store(multi_tenant_home)
    try:
        mid = st.remember("acme original wording", tenant="acme")["id"]
        st.update_memory(mid, content="acme revised wording", tenant="acme")
        st.flush()
        rows = st.conn.execute(
            'SELECT COUNT(*) FROM "fts_acme" WHERE "fts_acme" MATCH ?',
            ('"revised"',)).fetchone()[0]
        assert rows == 1, "the updated content is not searchable in acme's table"
        stale = st.conn.execute(
            'SELECT COUNT(*) FROM "fts_acme" WHERE "fts_acme" MATCH ?',
            ('"original"',)).fetchone()[0]
        assert stale == 0, "the old posting survived the update"
    finally:
        st.close()


def test_purge_removes_postings_from_the_right_table(multi_tenant_home):
    """Purge drops a row's posting from *its* tenant's table, and only that one."""
    st = make_store(multi_tenant_home)
    try:
        # The fixture soft-deletes a globex row, not an acme one, so acme has
        # nothing purgeable until this test creates some — otherwise it would
        # pass for the wrong reason (0 purged, 0 postings removed).
        doomed = st.remember("acme doomed document", tenant="acme")["id"]
        st.forget(doomed, tenant="acme")
        before = st.conn.execute('SELECT COUNT(*) FROM "fts_acme"').fetchone()[0]

        result = st.purge(tenant="acme")
        st.flush()

        assert result["purged"] == 1, result
        after = st.conn.execute('SELECT COUNT(*) FROM "fts_acme"').fetchone()[0]
        assert after == before - 1, f"acme postings went {before} -> {after}"
        # globex's soft-deleted row is not purgeable by an acme-scoped purge
        assert st.conn.execute('SELECT COUNT(*) FROM "fts_globex"').fetchone()[0] == 1
    finally:
        st.close()


def test_a_batch_mixing_tenants_lands_each_row_in_its_own_table(multi_tenant_home):
    """The write path groups by tenant; a mixed batch is the case that proves it."""
    st = make_store(multi_tenant_home)
    try:
        st.remember_many([
            {"content": "acme mixed one", "tenant": "acme"},
            {"content": "globex mixed two", "tenant": "globex"},
            {"content": "acme mixed three", "tenant": "acme"},
            {"content": "initech mixed four", "tenant": "initech"},
        ])
        st.flush()
        assert st.conn.execute(
            'SELECT COUNT(*) FROM "fts_acme" WHERE "fts_acme" MATCH ?',
            ('"mixed"',)).fetchone()[0] == 2
        assert st.conn.execute(
            'SELECT COUNT(*) FROM "fts_globex" WHERE "fts_globex" MATCH ?',
            ('"mixed"',)).fetchone()[0] == 1
        assert st.conn.execute(
            'SELECT COUNT(*) FROM "fts_initech" WHERE "fts_initech" MATCH ?',
            ('"mixed"',)).fetchone()[0] == 1
    finally:
        st.close()


def test_the_default_tenant_gets_a_table_too(tmp_path):
    """Every pre-tenant row lands under `default`, so it must be searchable."""
    st = make_store(tmp_path)
    try:
        st.remember("a plain local note about marmalade")
        st.flush()
        assert st.conn.execute('SELECT COUNT(*) FROM "fts_default"').fetchone()[0] == 1
        assert st.recall("marmalade", k=5)
    finally:
        st.close()


# --- the property the migration exists for -----------------------------------

def test_ranking_is_local_to_the_tenant(multi_tenant_home):
    """One tenant's bm25 scores must not move when another tenant adds rows.

    This is the test that fails against the shared table and passes against
    per-tenant ones. `bm25()` derives IDF from whatever table it queries, so on
    the global `memory_fts` a large foreign corpus reweights a rare term.

    Asserting the *scores*, not just the row order, because the order is the
    weaker property: with two matching rows the order can survive a reweight
    while the scores move. Measured against the shared table — acme's score for
    "alpha" goes from -1e-06 to -1.256e-06 after 400 globex rows mentioning the
    same term, with acme's own corpus untouched. Retrieval is being tuned by
    another tenant's data, and no amount of correct tenant filtering fixes it.
    """
    st = make_store(multi_tenant_home)
    try:
        query = "alpha"

        def scores() -> list[tuple[int, float]]:
            rows = st.conn.execute(
                'SELECT f.rowid, bm25("fts_acme") AS score FROM "fts_acme" f '
                f'JOIN memories m ON m.id = f.rowid '
                f'WHERE "fts_acme" MATCH ? AND m.{_SELECTIVE} AND m.tenant_id = ? '
                'ORDER BY score LIMIT 10',
                (f'"{query}"', "acme"),
            ).fetchall()
            return [(int(r["rowid"]), float(r["score"])) for r in rows]

        before = scores()
        assert before, "the control query found nothing to rank"

        # Pile a large, term-saturated corpus onto a different tenant.
        for i in range(400):
            st.remember(f"globex filler {i} mentions alpha repeatedly", tenant="globex")
        st.flush()

        assert scores() == before, (
            "acme's lexical scores changed when an unrelated tenant added rows; "
            "bm25 is still computing IDF over a shared corpus"
        )
    finally:
        st.close()

def _v5_home(tmp_path, rows=()):
    """A home left at schema v5, as the v6 migration will actually meet it.

    Built by migrating forward and then rewinding, because there is no other way
    to get a v5 database out of a codebase that is now at v6 — and the point of
    these two tests is what happens when the migration *runs*, which requires a
    home that has not run it yet.
    """
    st = make_store(tmp_path)
    try:
        for content, tenant in rows:
            st.remember(content, tenant=tenant)
        st.flush()
    finally:
        st.close()
    import sqlite3

    conn = sqlite3.connect(tmp_path / "omnidb.db")
    try:
        # DROP TABLE on an FTS5 virtual table leaves its five shadow tables
        # (_content, _data, _idx, _docsize, _config) behind, and a later
        # CREATE VIRTUAL TABLE with the same name then fails on them. Drop the
        # shadows too, so this really is a v5 home rather than a broken hybrid.
        for shadow in conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'fts\\_%' ESCAPE '\\'"
        ).fetchall():
            conn.execute(f'DROP TABLE IF EXISTS "{shadow[0]}"')
        conn.execute("CREATE VIRTUAL TABLE memory_fts USING fts5("
                     "content, tokenize='porter unicode61')")
        conn.execute("INSERT INTO memory_fts(rowid, content) "
                     "SELECT id, content FROM memories")
        conn.execute("DELETE FROM meta WHERE key LIKE 'tenant_fts:%'")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
    finally:
        conn.close()
    return tmp_path



# --- crash safety: a torn migration must never be published -------------------

def test_a_failed_migration_does_not_publish_v6(tmp_path, monkeypatch):
    """If v6 raises partway, the home must stay openable at v5.

    The guarantee is the same one v2 and v4 have: DDL, data movement and the
    version bump share one transaction. Injected by failing the population step
    partway through — the shape of a disk-full or a locked file mid-migration,
    which is exactly when the first tenant's table already exists and the second
    does not.
    """
    import omnidb_server.db as dbmod_ref

    _v5_home(tmp_path, [("acme alpha", "acme"), ("globex beta", "globex")])

    real = dbmod_ref._populate_per_tenant_fts

    def exploding(conn):
        # Run the real thing, then fail — so the first tenant's table is on disk
        # and the migration is genuinely torn rather than uniformly broken.
        real(conn)
        raise dbmod_ref.DbError("injected failure after the first tenant")

    monkeypatch.setattr(dbmod_ref, "_populate_per_tenant_fts", exploding)

    from omnidb_server.db import DbError, connect
    with pytest.raises(DbError, match="migration to v6 failed"):
        connect(tmp_path)

    # Nothing published: the version is still 5, and no half-built index
    # survives — the rollback takes the first tenant's table with it.
    import sqlite3
    conn = sqlite3.connect(tmp_path / "omnidb.db")
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name LIKE 'fts%'").fetchone() is None, \
            "the rolled-back migration left an FTS table behind"
    finally:
        conn.close()

    monkeypatch.setattr(dbmod_ref, "_populate_per_tenant_fts", real)
    conn = connect(tmp_path)   # retry must succeed
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
        assert "memory_fts" not in {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()


def test_the_postcondition_refuses_to_publish_a_short_table(tmp_path, monkeypatch):
    """A tenant whose postings came up short must block the version bump.

    The postcondition is the last thing standing between a partial migration and
    a home that opens claiming to be v6. Fault-inject one row short and assert
    the open fails rather than publishing a home whose lexical index is missing
    a memory.
    """
    import omnidb_server.db as dbmod_ref

    _v5_home(tmp_path, [("acme alpha", "acme"), ("acme beta", "acme")])

    real = dbmod_ref._populate_per_tenant_fts

    def short_fill(conn):
        real(conn)
        # Drop one posting behind the migration's back, so the table holds one
        # row fewer than `memories` and the postcondition has to catch it. The
        # table is looked up rather than named: which tenant exists depends on
        # the rows the fixture wrote, and a hardcoded name here would fail for
        # the wrong reason.
        (table,) = conn.execute(
            "SELECT name FROM sqlite_master WHERE sql LIKE '%CREATE VIRTUAL TABLE%'"
        ).fetchone()
        conn.execute(f'DELETE FROM "{table}" '
                     f'WHERE rowid = (SELECT MAX(rowid) FROM "{table}")')

    monkeypatch.setattr(dbmod_ref, "_populate_per_tenant_fts", short_fill)

    from omnidb_server.db import DbError, connect
    with pytest.raises(DbError, match="expected 2"):
        connect(tmp_path)


def test_a_home_with_no_rows_migrates_and_opens(tmp_path):
    """An empty home must reach v6, not stall on the missing first tenant.

    `_populate_per_tenant_fts` iterates the tenants present in `memories`, so a
    home with no rows creates no FTS table at all. That is correct — there is
    nothing to index — but it means the very first write is what brings a table
    into existence, and `schema()` / `recall` before that write must still work
    rather than raise on a missing table.
    """
    from omnidb_server.db import SCHEMA_VERSION

    st = make_store(tmp_path)
    try:
        assert st.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert _fts_tables(st.conn) == set(), "an empty home should have no FTS table"
        # reads must not raise on the absent table
        assert st.recall("anything", k=5) == []
        assert st.schema()["schema_version"] == SCHEMA_VERSION
        assert st.check()["fts_integrity"] == "ok"
        # and the first write brings its tenant's table into being
        st.remember("first note", tenant="default")
        st.flush()
        assert _fts_tables(st.conn) == {"fts_default"}
    finally:
        st.close()


def test_recall_on_an_empty_home_does_not_create_a_table(tmp_path):
    """A read must not be a write.

    `_fts_table` creates the table on demand, so a read path that resolves it
    would turn a `recall` on an empty home into a DDL write — surprising, and it
    would make a read-only diagnostic open dirty a home it was only inspecting.
    """

    st = make_store(tmp_path)
    try:
        st.recall("nothing here", k=5)
        assert _fts_tables(st.conn) == set(), (
            "recall created an FTS table; a read must not write"
        )
    finally:
        st.close()
