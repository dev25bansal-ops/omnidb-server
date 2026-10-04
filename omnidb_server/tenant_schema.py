"""Per-tenant FTS5 table lifecycle (ADR-08, storage-design appendix).

## Why this exists

The single `memory_fts` table is an **external-content** FTS5 table bound to the
whole `memories` table (`db.py:31`). External-content tables cannot be filtered
per tenant, so lexical isolation requires **one standalone FTS table per
tenant**, synchronised explicitly in the store's write path.

This module owns the *lifecycle* of those tables — create on first use, drop on
teardown — and the crash-safety contract that a half-created tenant can never
be reachable.

## The crash-safety contract

A tenant is reachable **iff its registration row exists** in `meta`, not iff its
table exists. `ensure_tenant_fts` creates both in one transaction, so:

- a crash before commit leaves neither the table nor the marker → the tenant
  simply does not exist, and the next open creates it cleanly;
- a crash after commit leaves both → the tenant is fully there;
- there is no state where the marker exists but the table does not, and no
  reverse, because the two statements are atomic.

The write path checks the marker, not the table, so it can never submit work to
a table that was torn in half.

## Identifier safety

Table names are `fts_<tenant>`. `tenant` is `validate_tenant_id`d first — the
character set is `[A-Za-z0-9_-]` and length-capped — so the name cannot carry a
quote, a statement terminator, or a path separator. The double-quotes in the
DDL below are defence in depth, not the primary control; the primary control is
that the id is validated before it is ever interpolated.
"""

from __future__ import annotations

import sqlite3

from .tenancy import TenantError, validate_tenant_id

# The FTS5 tokenizer the single-table `memory_fts` used — kept identical so a
# query behaves the same before and after the split.
_TOKENIZER = "porter unicode61"

_FTS_PREFIX = "fts_"
_TENANT_META_PREFIX = "tenant_fts:"


def fts_table_name(tenant: str) -> str:
    """The (quoted) per-tenant FTS table name.

    Returns a double-quoted SQL identifier so a tenant id with a hyphen
    (`acme-prod`) is legal. The tenant is validated first, so the returned
    string can only contain the validated character set plus the fixed prefix.
    """
    validate_tenant_id(tenant)
    return f'"{_FTS_PREFIX}{tenant}"'


def _marker(tenant: str) -> str:
    return f"{_TENANT_META_PREFIX}{tenant}"


def register_and_create(conn: sqlite3.Connection, tenant: str) -> bool:
    """Create the tenant's FTS table if absent; return True if it was new.

    Idempotent and atomic. The registration row is the reachability criterion:
    it is written in the same transaction as the CREATE, so a torn tenant is
    impossible by construction.
    """
    validate_tenant_id(tenant)
    table = fts_table_name(tenant)
    marker = _marker(tenant)

    if conn.execute(
        "SELECT 1 FROM meta WHERE key = ?", (marker,)
    ).fetchone() is not None:
        return False  # already registered → already usable

    try:
        conn.execute("BEGIN")
        # IF NOT EXISTS is the idempotency backstop; the marker check above is
        # the fast path. A concurrent create from another connection loses the
        # race here at the marker insertion, not at the table.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
            f"content, tokenize='{_TOKENIZER}')"
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, '1') "
            "ON CONFLICT(key) DO NOTHING",
            (marker,),
        )
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise TenantError(f"cannot create FTS table for tenant {tenant!r}: {exc}") from exc
    return True


def drop_tenant(conn: sqlite3.Connection, tenant: str) -> None:
    """Drop the tenant's FTS table and its registration, atomically.

    The registration is removed in the same transaction as the DROP, so a
    crash cannot leave a marker pointing at a missing table (which the write
    path would then submit to).
    """
    validate_tenant_id(tenant)
    table = fts_table_name(tenant)
    marker = _marker(tenant)
    try:
        conn.execute("BEGIN")
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("DELETE FROM meta WHERE key = ?", (marker,))
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise TenantError(f"cannot drop FTS table for tenant {tenant!r}: {exc}") from exc


def list_registered(conn: sqlite3.Connection) -> list[str]:
    """Every tenant with a registered FTS table, sorted for stable output."""
    rows = conn.execute(
        "SELECT key FROM meta WHERE key LIKE ? ORDER BY key",
        (_TENANT_META_PREFIX + "%",),
    ).fetchall()
    return [r[0][len(_TENANT_META_PREFIX):] for r in rows]


def is_registered(conn: sqlite3.Connection, tenant: str) -> bool:
    validate_tenant_id(tenant)
    return conn.execute(
        "SELECT 1 FROM meta WHERE key = ?", (_marker(tenant),)
    ).fetchone() is not None


def parse_marker(key: str) -> str | None:
    """Given a meta key, return the tenant it names, or None if unrelated."""
    if isinstance(key, str) and key.startswith(_TENANT_META_PREFIX):
        return key[len(_TENANT_META_PREFIX):]
    return None