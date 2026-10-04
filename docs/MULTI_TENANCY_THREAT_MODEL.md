# Multi-Tenancy Threat Model

Status: **DRAFT — requires owner decisions in §9 before it can be approved.**
Scope: hosting `omnidb-server` as a service used by mutually distrustful tenants.
Supersedes the assumption recorded in `docs/AUDIT.md:15` that
"`agent_id` is a retrieval label, not an authenticated tenant boundary."

Baseline reference: commit `6c99e9f`. All `file:line` citations were read in that tree.

---

## 1. Current state: there is no tenant boundary

There is no authentication, session, token, principal, or tenant primitive anywhere in
`omnidb_server/`. The only credential in the codebase is the *outbound* LLM API key
(`config.py:104`, `consolidate.py:111`). `agent_id` is a plain, caller-supplied
`str | None` on every entry point. It is 100% self-declared and forgeable.

`docs/AUDIT.md:15` is therefore accurate **about today**. It is a description of a
single-user local tool, not a security property that survives hosting.

### Bypass inventory

Ranked by how little skill each attack requires:

| # | Attack | Requires | Evidence |
|---|---|---|---|
| 1 | `search("SELECT * FROM memories")` reads every tenant, plus `meta` and `queries` | nothing — `agent_id` is not consulted at all | `server.py:53`, `store.py:737-815` |
| 2 | `update_memory` / `forget` / `forget_many` overwrite or delete any tenant's rows | only a victim row id; no ownership check exists | `server.py:69,76,81`; `store.py:539-543,607-635` |
| 3 | `remember(links=[victim_id])` then `related(victim_id)` exfiltrates the victim's graph | weld own node to a foreign id; traversal has no agent filter | `store.py:712-716`, `server.py:63`, `store.py:830-855` |
| 4 | `recall(agent_id=None)` silently returns **all** agents | omitting the argument is not an error | `recall.py:70,100,148` |
| 5 | `consolidate` blends every tenant's episodes into one LLM transcript and writes mixed-agent facts back as *unscoped* (thus universally visible) rows | one call | `consolidate.py:53-70,198,241-242` |
| 6 | `stats` enumerates every `agent_id` and its count | nothing | `store.py:975-982` |

Attack 5 is the most serious: it is a *confidentiality breach triggered by a normal
maintenance operation*, and its output is written into the store where other tenants'
retrieval will pick it up.

### Two subtler defects

- **The dense retrieval leg is global.** `vector_index.query()` takes no agent
  parameter (`vector_index.py:102,196`); `recall.py:68` runs ANN over the whole corpus
  and filters by agent *afterwards*, in SQL. `agent_overfetch` (`config.py:90-93`)
  exists precisely because this drops scoped results. Post-filtering a global index is
  both a correctness bug and an information channel.
- **The vector index has no partition.** One global `vectors.npz` / `index.hnsw`,
  keyed by memory id (`vector_index.py:266-267`).

---

## 2. Schema: tenancy is missing where it is needed

Tables as created by `db.py`:

| Table | Tenant column today | Verdict |
|---|---|---|
| `memories` | `agent_id TEXT`, nullable, no FK (`db.py:59-63`) | advisory label, not ownership |
| `edges` | **none** (`db.py:49-55`) | graph traversal crosses tenants structurally |
| `memory_fts` | **none** (`db.py:29-34`) | FTS5 content mirror is global |
| `meta` | **none** — `key` is the only PK (`db.py:11-14`) | global KV: `embedder`, `vec_dim`, `vec_kind`, `index_dirty`, `pending_reembed` (`store.py:188-191,234-292`) |
| `queries` | `agent_id TEXT` nullable, unauthenticated (`db.py:66-73`) | cross-tenant telemetry channel |

Adding a `tenant_id` column is not a migration you can layer on top: `edges`,
`memory_fts`, and the vector index must all become tenant-partitioned, and `meta` must
be namespaced or split. This is a rebuild of the storage contract, not an `ALTER TABLE`.

---

## 3. Target identity model

Principle, taken from the MCP spec's security guidance: **identity is derived from the
authenticated token, never supplied by the client**
(`modelcontextprotocol.io/specification/2025-11-25/basic/security_best_practices`).

```
client ──bearer JWT──▶ gateway (validates audience/signature)
                          └─ resolves tenant_id server-side
                               └─▶ omnidb-server (receives a trusted tenant context,
                                    never a client string)
```

Requirements:

- **T1** Transport is Streamable HTTP. The legacy HTTP+SSE transport is formally
  deprecated as of spec revision `2026-07-28`.
- **T2** Server acts as an OAuth 2.1 Resource Server. Clients MUST use PKCE `S256` and
  MUST send the RFC 8707 `resource` parameter; the server MUST reject tokens not issued
  for it as audience. Token passthrough is prohibited.
- **T3** `tenant_id` comes from a validated claim resolved server-side. It is **not** a
  JWT-embedded snapshot of authorization decisions — those freeze at login; permissions
  are resolved against live state.
- **T4** Every existing `agent_id` parameter is demoted to a *within-tenant label*.
  Ownership is `tenant_id`; `agent_id` becomes an optional tag under it.
- **T5** Returned memory content is untrusted input. A stored string is an attack on
  whoever retrieves it; see §6 A5.

### Spec-stability warning

`2026-07-28` removed the `initialize`/`initialized` handshake, the `Mcp-Session-Id`
header, and SSE resumability, moving the protocol stateless-first. Session and client
registration mechanics are the parts still churning — Dynamic Client Registration is
deprecated in favour of Client ID Metadata Documents. **Do not build state that depends
on a protocol-level session id.** The statelessness helps horizontal scaling, but
anything assuming sticky sessions is built on sand.

---

## 4. Isolation guarantees (each must be testable)

- **G1** No tool accepts a tenant identity from the client.
- **G2** Every read and write is confined to the caller's tenant by a predicate that
  the caller cannot express or omit.
- **G3** Denying a tenant's token stops in-flight reads within a bounded interval.
- **G4** Two tenants storing identical content cannot observe each other's existence,
  counts, timing, or cache behaviour.
- **G5** Graph traversal (`related`, `link`) cannot cross a tenant boundary, including
  through an edge a tenant creates toward a foreign id.
- **G6** No maintenance or background job mixes tenants in one LLM context, prompt, or
  write batch.
- **G7** Vector retrieval is filtered *inside* the index, not post-hoc.
- **G8** A raw-SQL surface (`search`) either does not exist or is confined to a
  tenant-scoped read-only view.
- **G9** Backups, exports, restores, and re-embed jobs are per-tenant and cannot
  reconstruct another tenant's rows.

G6 and G7 are new requirements — neither is satisfiable by the current design at all.

---

## 5. Enforcement layers

Authorisation is enforced twice, deliberately:

1. **Gateway** — signature, audience, expiry, tenant resolution, rate limit.
2. **Database row-level security** — defence in depth, so an application bug does not
   become a cross-tenant read.

### RLS constraints that must be honoured

- Policy predicates use `current_setting('app.tenant_id')`; subqueries in policies are
  evaluated per row and will dominate query cost.
- **PgBouncer transaction pooling shares a session across clients.** `SET app.tenant_id`
  leaks to the next client's transaction. Use `SET LOCAL` inside an explicitly opened
  transaction so it resets on commit — or use session pooling. This is the single most
  common way RLS deployments leak.
- Superuser, `BYPASSRLS`, and the table owner bypass RLS unless
  `ALTER TABLE … FORCE ROW LEVEL SECURITY`. The app role must be non-privileged.
- Referential-integrity checks bypass RLS — a foreign-key path can become a covert
  channel. Relevant here because `edges` and `superseded_by` are FKs into `memories`.
- `row_security=off` is needed for some backup paths; keep those roles out of the
  request path.
- Index `tenant_id` on every guarded table, or policies turn sequential scans into
  latency incidents.

---

## 6. Threats

| ID | Actor | Path | Status today | Control |
|---|---|---|---|---|
| A1 | Hostile tenant | `search` raw SQL | **fully open** | G8 |
| A2 | Hostile tenant | id-addressed mutation | **fully open** | G2/G5 |
| A3 | Hostile tenant | graph join to victim, then read | **fully open** | G5 |
| A4 | Hostile tenant | omit `agent_id` → all | **fully open** | G1/G2 |
| A5 | Hostile tenant | store prompt-injection payload, let another tenant retrieve it into their LLM | **unmitigated, not covered by MCP spec** | treat returned content as untrusted; tenant-isolated writes |
| A6 | Any tenant | `consolidate` leaks all tenants into one prompt | **closed 2026-09-30** (tenant fenced on all four row accesses; `tests/test_consolidation_tenant.py`) | G6 |
| A7 | Confused-deputy via proxy | client-supplied identity + static upstream creds | applies the moment a gateway exists | T2/T3, per-client consent |
| A8 | Insider / owner role | `BYPASSRLS`, table owner | n/a | FORCE RLS, non-privileged app role |
| A9 | Cross-tenant timing | shared global ANN index, shared cache | **inherent** | G7 plus per-tenant index or partition |
| A10 | ReDoS via stored `trigger_patterns` (`recall.py:41,165` compiles arbitrary user regex; `validation.py:28` bounds only count/length) | any tenant can degrade the shared process | **open** | bound match time; per-tenant CPU budget |

A5 and A10 are the ones a conventional auth design misses. A5 is the reason this is an
agent-memory service and not just a database: the payload is text another LLM will obey.
A10 is a denial-of-service against the shared process, which per-tenant isolation of
*data* does not fix.

---

## 7. Breaking changes implied

1. `search(sql)` removed, or replaced by a tenant-scoped view with no general SQL.
2. DONE 2026-09-30: `consolidate` is tenant-scoped, so one transcript never contains two
   tenants. Also fixes `consolidate.py:230-236`, which currently reports success when
   only one chunk of ten succeeded.
3. `recall` rejects a missing agent scope rather than defaulting to all.
4. `edges`, `memory_fts`, vector index gain tenant partitioning; `meta` is namespaced.
5. Every mutation takes an ownership check (`update_memory`, `forget`, `forget_many`).
6. Home lock (`server.py:197-232`) and filesystem sidecar persistence are retired —
   both encode one-process-per-tenant.
7. Embedder identity moves from per-home to per-tenant. Note the current design
   *refuses* a second embedder in one home (`store.py:197-212`), so today all tenants
   must share one model and dimension. Per-tenant embedders are a new capability, and
   with them comes per-tenant tuning: measured fusion weights range from 0.7 (nfcorpus)
   to ≤0.3 (scidocs), so quality is corpus-dependent and a fleet needs per-tenant
   calibration nobody has yet measured.
8. New: transport, auth, tenant resolution, RLS, and logging. There is currently no
   logging framework at all — zero `getLogger` in the package — and `doctor.py` always
   exits 0 (`cli.py:133-137`), so nothing can probe the process.

---

## 8. Explicit non-goals

- Not claimed: encryption at rest, field-level crypto, key custody, or compliance scope.
- Not claimed: anti-abuse billing or quota economics.
- Not addressed: multi-region data residency.
- Not addressed: the local single-user mode staying compatible. A dual mode is
  achievable but is a separate design decision, not a free by-product.

---

## 9. Owner decisions required

1. **What is a tenant?** Organisation, workspace, individual end-user, or the
   agent-running product? Everything below forks on this.
2. **Who issues tokens?** Build an OAuth provider, or put an existing IdP/gateway in
   front and have the server only validate audience?
3. **How hostile are tenants to each other?** SaaS-with-billing (assume determined
   attackers, full RLS, isolated indexes) versus same-company teams (lighter isolation,
   shared index acceptable).
4. **Is per-tenant embedder choice required?** If no, a shared index becomes viable and
   much of §7.7 disappears. If yes, per-tenant tuning and cost attribution are needed.
5. **What is the tenancy unit for `consolidate`'s LLM spend?** Per-tenant quota, or
   central subsidy? Unbounded per-tenant distillation is a cost and abuse surface.

Until 1 and 3 are answered, the datastore decision in `DATASTORE_SPIKE.md` cannot be
closed — not because the analysis is missing, but because the requirement is undefined.
