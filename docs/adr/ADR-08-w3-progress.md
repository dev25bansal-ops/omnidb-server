# ADR-08 rebuild — Wave 3 progress

**Status: W3-0 and W3-1 complete.** The foundation is in place; the storage
layer (W3-2/W3-3) is the bulk of the remaining 6–8 weeks and is not started.

| Wave | Scope | State |
|---|---|---|
| W3-0 | ADR + design | ✅ `docs/adr/ADR-08-multi-tenant-rebuild.md` |
| W3-1 | HTTP transport + auth | ✅ modules + tests + CLI entry |
| W3-2 | tenant_id schema + runtime DDL | ✅ schema v4 + `tenant_schema.py` |
| W3-3 | per-tenant FTS + HNSW lifecycle | ▶ triggers removed + explicit sync done; per-tenant sidecar tables remain |
| W3-4 | ownership checks (SEC-02) | ⏳ |
| W3-5 | rewrite 4 tenancy xfails | ⏳ |
| W3-6 | full verification | ⏳ |

## W3-2 landed

- **Schema v4** (`db.py`): `memories.tenant_id` and `queries.tenant_id`, both
  `NOT NULL DEFAULT 'default'`, plus `memories_tenant_idx`. Pre-tenant homes
  keep working; the column defaults to one tenant.
- **`tenant_schema.py`**: the per-tenant FTS5 table lifecycle, with the
  crash-safety contract written into `ADR-08`'s storage appendix — a tenant is
  reachable **iff its registration row exists in `meta`**, written in the same
  transaction as the CREATE. Table names are `fts_<tenant>`, double-quoted, and
  only reachable through `validate_tenant_id`'d ids.
- **Torn-v4 healing**: `_heal_torn_v4` mirrors the v2 AUDIT-I10 repair — a
  home whose `tenant_id` column already exists but is stuck at `user_version <
  4` heals rather than raising "duplicate column".

## A finding worth recording

The AST SQL-injection guard (`test_sql_injection_surface.py`) **caught my own
W3-2 work twice**, and both times it was right:

1. `table = fts_table_name(tenant)` spliced into `CREATE VIRTUAL TABLE` — a
   runtime table name. This one is *legitimate* (per-tenant DDL cannot use a
   literal), so `fts_table_name` was added to the classifier's allow-list with
   a note explaining why it is safe (validated id, fixed prefix, quoted).
2. `for table in ("memories", "queries")` in `_heal_torn_v4` — the loop
   variable spliced into `PRAGMA table_info({table})`. This one was **not**
   legitimate: there are exactly two tables, and a loop variable was buying
   nothing. I *unrolled it* into literals rather than teaching the classifier
   to swallow loop variables.

The second case is the guard doing its job: the unrolled form is provably safe
by construction, where the loop form needed a human to reason about where
`table` came from. A fail-closed guard repeatedly forcing static-verifiable
forms is the outcome the design intended.

## Verification

```
447 passed, 5 skipped, 4 xfailed
ruff: All checks passed
mypy: Success, 25 source files
```

The 4 xfailed are still the tenancy bypasses — correct, the boundary is not
enforced yet. That flips in W3-4/W3-5.

## Next

W3-3: switch `recall.py`'s lexical leg from the single `memory_fts` to the
per-tenant `fts_<tenant>` table, synchronise it in the store's write path
(removing the three external-content triggers), and per-tenant HNSW sidecars.
This is the first change that actually *enforces* the boundary on read and
write, and it is where the byte-level correctness of the whole rebuild is won
or lost.
| W3-4 | ownership checks (SEC-02) | ⏳ |
| W3-5 | rewrite 4 tenancy xfails | ⏳ |
| W3-6 | full verification | ⏳ |

## What landed in W3-1

- **`omnidb_server/tenancy.py`** — the credential layer. `TenantRegistry`
  resolves a bearer key to a tenant **inside** OmniDB, hashing keys into
  digests so a leaked config never hands over working credentials,
  `hmac.compare_digest` sweep so lookup time does not leak a prefix match.
  `validate_tenant_id` constrains ids to `[A-Za-z0-9_-]` with a length cap,
  because a tenant id **reaches a filename** and must not traverse.
- **`omnidb_server/http_server.py`** — streamable-HTTP transport. Wires
  OmniDB's own `TenantTokenVerifier` into MCP's `token_verifier=` +
  `auth=` pair. `build_registry` refuses to start an empty (credential-less)
  multi-tenant listener.
- **`attach_tools(mcp, store)`** extracted from `create_mcp` so stdio and HTTP
  share **one** tool-surface definition — a drift between the two transports
  is a security drift (a tool missing its tenant check on one of them).
- **CLI**: `serve-http --host --port` and `tenant-key` commands.

## The trust model, in one sentence

A caller **cannot name its own tenant** — it never supplies a `tenant_id`, and
the only thing that resolves the tenant is a key OmniDB verifies itself. That
is what turns `agent_id` from a label into a boundary.

## What this explicitly does NOT do yet

The tools still accept and honour `agent_id`, and the store still has **no
tenant column**. The auth layer can now *say* who is calling; nothing below it
*enforces* the boundary yet. That is exactly the scope of W3-2 through W3-5,
and until those land the four `xfail(strict=True)` tenancy tripwires correctly
stay red-or-skipped.

## Two defects I caught in my own W3-1 work

1. **I reversed `token_verifier=` and `auth=`.** FastMCP takes them as *separate*
   arguments; passing a backend where `AuthSettings` was expected is a silent
   no-op that leaves the listener unauthenticated. Found by reading the actual
   `FastMCP.__init__` source rather than guessing. `test_build_auth_returns_verifier_and_settings_pair`
   pins the two-tuple shape.
2. **`AccessToken` requires `scopes`.** Omitting it raised a pydantic
   `ValidationError` at verify time, which under HTTP would surface as a 500 —
   the exact oracle that `verify_token`'s return-None path exists to avoid.
   Added `scopes=[]` with a note that a key grants whole-tenant access and an
   empty scope list is the honest rendering.

## Verification state

```
437 passed, 5 skipped, 4 xfailed
ruff: All checks passed
mypy: Success, 24 source files
```

The 4 `xfailed` are the tenancy tripwires — correct, because the boundary is
only half-built. They must turn into real passing assertions as W3-4/W3-5 land,
and each `strict=True` marker must be deleted the moment its bypass closes.

## Next

W3-2 (`tenant_id` schema + a runtime DDL mechanism the forward-only ledger
cannot express) and W3-3 (per-tenant FTS5 and HNSW lifecycle) are the two big
pieces. W3-2 alone carries the `SCHEMA_VERSION` bump that six test modules pin,
and `memory_fts` is an external-content table whose triggers have hard-coded
table names — so per-tenant FTS is runtime table creation, not a column, and it
must be crash-safe (a half-created tenant must never be reachable).