# ADR-08 — Multi-tenant rebuild

**Status:** accepted · **Date:** 2026-09-29 · **Supersedes:** ADR-5 (single-writer, one home)

## Context

OmniDB 0.1.3 has a `agent_id` column on `memories` (`db.py:60`) and an index
`memories_agent_idx` (`db.py:62`). It is **not** a tenant boundary. It is a
self-declared, nullable label: `NULL` is visible to every caller, and no
mutation path checks it.

The audit found **four confirmed bypasses**, all pinned as
`@pytest.mark.xfail(strict=True)` in `tests/test_tenancy_boundaries.py:28,57,77,95`:

1. **Dense leg** — `recall.py:67-68` queries the ANN globally, then filters at
   `:99-100` *after* fusion. `agent_overfetch` *increases* the pool for scoped
   recalls. Both a confidentiality leak and an information channel.
2. **`related()`** — `store.py:817-855` walks `edges` with **no tenant predicate
   at all**. Forging one edge reads the victim's content. Direct and complete.
3. **`stats()`** — `store.py:963-982` does `GROUP BY agent_id` over every live row
   and returns it to any caller. An information channel.
4. **Sidecar at rest** — one `vectors.npz` / HNSW graph holds every tenant's
   vectors.

The four `xfail(strict=True)` markers are the right way to record a
deliberately-unfixed defect. They are the reason this ADR exists: the repo
decided to keep them, and the decision is now reversed.

## Decision

The owner has answered three questions that fix the shape of the work.

**1. One home, many tenants, with per-tenant FTS and HNSW tables.**

Not the 0-LOC option (one home per tenant). The chosen option buys true
isolation on every leg including the lexical one and the vector index, at the
cost of runtime DDL that the current forward-only migration ledger cannot
express.

**2. Per-tenant API key, resolved by OmniDB from an `Authorization` header.**

OmniDB owns the auth surface and verifies the key itself. A client-supplied
`tenant_id` parameter would be a more durable `agent_id` — the same defect with
a new column name.

**3. Clean break. Existing homes stay on the old schema.**

Export/import, which the project already implements
(`omnidb_server/portability.py`). No in-place migration.

## The constraint that forced a transport change

`create_mcp(store).run()` (`server.py:306`) is **stdio**. stdio has no headers —
one process, one stdin/stdout pipe, one identity. A bearer token per request is
not expressible.

Per-tenant tables require per-request identity. Therefore:

**Streamable HTTP becomes a first-class transport, alongside stdio.**

This is not a free change. It moves OmniDB from "local tool" to "network
listener", and it invalidates parts of the existing threat model:

| Was (stdio, single-user) | Becomes (HTTP, multi-tenant) |
|---|---|
| SEC-03 ReDoS is **self-inflicted** — the agent that writes a trigger pattern is the one that reads it | **Cross-tenant DoS.** One tenant wedges the shared RLock and every tenant's recall queues behind it |
| The home lock is a correctness feature (one writer) | The home lock is **no longer a security boundary** — it serialises writers, it does not authorise them |
| `llm_base_url` egress is operator-controlled | Unchanged, but a malicious tenant can no longer choose it — good |
| `query-log` is a local diagnostic | `query-log` becomes a **cross-tenant** disclosure (SEC-07, 7.6) |

**SEC-03's severity is now correctly graded as it was always suspected to be.**
The audit's narrowing ("P2 as shipped, P0 under model B") resolves to P0.

## What this supersedes

- **ADR-5, single-writer / one home per deployment.** Still true for stdio.
  False for HTTP, where one process serves N tenants.
- **`docs/MULTI_TENANCY_THREAT_MODEL.md` §9** — the open question it ended on
  ("the tenant definition is unresolved") is now answered.
- **The four `xfail(strict=True)` tripwires.** They become real tests. A fix
  turns them green; `strict=True` means forgetting to remove the marker fails
  the suite. **That is the desired behaviour and the markers must be deleted
  as each bypass closes** — leaving one behind after the fix is a red test.

## Consequences

**Accepted costs:**

- Runtime DDL. `memory_fts` is an **external-content** table (`db.py:31`,
  `content='memories'`) synchronised by three triggers whose table names are
  **not parameterisable** (`db.py:36-47`). Per-tenant FTS means one table and
  one trigger set per tenant, created and dropped at runtime. This is the
  single largest piece of new machinery.
- The forward-only `(version, ddl)` ledger in `db.py:78` cannot express
  "create a tenant". A second mechanism is required, and it must be
  crash-safe — a half-created tenant must not be reachable.
- `SCHEMA_VERSION` goes to 4. **Six test modules pin `== SCHEMA_VERSION`** and
  one tests `SCHEMA_VERSION+1` rejection; they move together.
- Two transports means two test surfaces. The stdio conformance tests
  (`test_stdio_conformance.py`) do not cover HTTP.

**Gains:**

- All four MT-02 bypasses close with an *enforced* boundary rather than a
  documented one.
- SEC-02 (id-addressed mutation without ownership, CVSS 9.5) becomes fixable —
  it is blocked on there being an authenticated principal to check against.
- `search` (SEC-01, 7.6) becomes scopeable, because the authorizer can now be
  given a tenant that the row predicates must match.
- Per-tenant HNSW means one tenant's corpus cannot degrade another's recall,
  and a tenant's vectors are not readable from a co-located filesystem.

**Explicitly NOT in scope:** distributed or multi-writer (ADR-5 stands for the
storage engine; the single-writer discipline remains, enforced by the home lock
plus the store RLock). Cross-tenant aggregate queries are impossible by design
and no API will offer them.

## Storage design (appendix — the per-tenant FTS problem)

`memory_fts` is an **external-content** table (`db.py:31`, `content='memories'`),
synchronised by three triggers whose target table names are hard-coded
(`db.py:36-47`). External-content FTS binds the FTS table to the **whole**
content table — it cannot be filtered per tenant. So per-tenant lexical
isolation cannot be "a column on `memory_fts`"; it requires **one FTS table per
tenant, synchronised explicitly**.

**Decision:** switch from external-content + triggers to **per-tenant,
standalone FTS5 tables** (`fts_<tenant>`), synchronised in the store's write
path, exactly as the vector index already is. The three triggers are removed.

This makes the lexical leg a first-class, explicitly-maintained per-tenant
artifact, isomorphic to the HNSW index. It also removes the trigger
indirection that is invisible to the crash-window markers — a FTS drift today
is only caught by `check()`, after the fact; explicit sync makes it part of the
same dirty→index→persist ordering that governs the vector side.

**Crash-safety requirement:** a tenant is registered in `meta` (key
`tenant_fts:<tenant>`) **in the same transaction** that creates its table. A
half-created tenant must never be reachable, so the reachability criterion is
"the registration row exists", not "the table exists". Because tenant ids are
constrained to `[A-Za-z0-9_-]` (`tenancy.validate_tenant_id`), a table name
derived from one cannot be an injection or escape vector.

**What does NOT change in the same breath:** the HNSW sidecar naming
(`tenancy.index_name`) and, critically, the write-path ordering in `store.py`
(`_after_commit`: dirty flag first, index mutation, compaction, debounced
persist). FTS sync slots into that ordering as one more mark-dirty side effect,
not as a new parallel mechanism.

## W3-3 design — per-tenant FTS and HNSW (implementation plan)

This is the first change that *enforces* the boundary on read and write. It is
the byte-level crux of the rebuild, so the design precedes the code and every
decision below is load-bearing.

### Tenancy is an explicit argument, not a contextvar

The MCP tool bodies are `async` and run the store through
`anyio.to_thread.run_sync` (`server.py:16`), so a `contextvars.ContextVar` set
in the request context **does not propagate** to the store worker thread.
`sidio`/`current_tenant()` in `http_server.py` works only because it is read on
the async side — the store cannot see it.

Therefore every store method that touches a tenant-owned table gains an
explicit `tenant: str = "default"` argument. The server layer resolves the
tenant from the access token and passes it down; **no tool parameter carries a
tenant**, so the value is only ever introduced by the trusted auth path, never
by the caller.

### FTS sync is in-transaction, not post-commit

The vector index is rebuilt-able, so it lives in `_after_commit` behind a dirty
flag. FTS is different only in cost, not in correctness: the three triggers
(`db.py:36-47`) today synchronise `memory_fts` **inside the same transaction**
as the `memories` write. Removing the triggers and replacing them with explicit
sync must preserve that atomicity, or a crash leaves FTS and rows divergent —
which is worse than the status quo and would be invisible to the vector-only
reconciliation.

So: every write method issues its FTS statements **before** `conn.commit()`, in
the same transaction. No FTS dirty flag, no FTS rebuild-on-open — because
in-transaction sync cannot tear, exactly as the triggers cannot. This keeps the
crash-window story identical to what already ships.

### Table lifecycle is a deployment operation, not a request path

`CREATE VIRTUAL TABLE` is DDL; SQLite commits any open transaction when DDL
runs, so it cannot participate in a write transaction. The store write path
therefore **assumes the tenant's FTS table already exists** and fails with a
clear error if it does not — it never creates it on the write path.

Tables are materialised once, ahead of traffic:
- at startup, `build_stack` calls `tenant_schema.register_and_create` for every
  tenant already registered in `meta`;
- a new tenant enters via `omnidb-server tenant-add <id>`, which registers and
  creates atomically (the operation `tenant_schema` already implements).

This keeps DDL off the hot path and makes the reachability criterion (a
registered tenant) match the deployed reality (a created table).

### HNSW follows the same shape, later in the same wave

Per-tenant HNSW (`tenancy.index_name`) is the vector-side analog. It is cheaper
to reason about than FTS because the index is already explicitly maintained and
already rebuild-able; the work is repointing the store at `tenant-<id>.*`
sidecars and scoping `index.query` per tenant. FTS is the harder half and lands
first; HNSW lands once the FTS read/write split is proven.

### agent_id becomes a within-tenant subdivision, not a boundary

`agent_id` stays, but is always applied **conjointly** with `tenant_id`. Its
nullable "search all" semantics (`? IS NULL OR agent_id = ?` in `recall.py:70`)
now means "all agents *within this tenant*", because the tenant predicate is a
hard equality that already narrowed the candidate set. This is the exact
property that makes the four MT-02 bypasses impossible: there is no query shape
that escapes the tenant column.

### Order of implementation

1. Remove the three `memory_fts` triggers; add explicit FTS sync to
   `remember_many`, `update_memory`, and `purge` (insert / replace / delete).
2. Point `recall.py`'s lexical leg at `fts_<tenant>`, and replace the
   `agent_id` conditions with `tenant_id = ?` plus the (conjunctional)
   `agent_id`.
3. Startup materialisation in `build_stack` + the `tenant-add` CLI command.
4. Per-tenant HNSW sidecars.
5. W3-4 (ownership checks) and W3-5 (rewrite the four xfails) ride on this.

Each step is followed by the full suite; a step that leaves FTS unsynchronised
anywhere is a red test because the lexical leg would return nothing for that
path.

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| One home per tenant | 0 LOC and closes all four bypasses. Rejected by the owner: it forfeits the shared-corpus story and pushes tenancy to deployment topology. |
| `tenant_id` column, filtered reads | ~3-5 days instead of 6-8 weeks, but FTS5 has no partition key so lexical isolation degrades to a post-filter join, and the sidecar still holds every tenant's vectors at rest. Buys the column, not the isolation. |
| Gateway-injected header | Offloads auth, but makes isolation depend on a proxy stripping a header correctly — an assumption outside this codebase. OmniDB verifying its own keys is auditable here. |
| In-place migration from `agent_id` | Cheaper, but every existing home becomes one tenant with `NULL` as its key, and the old rows carry a label that was never a boundary. The clean break is honest about that. |
