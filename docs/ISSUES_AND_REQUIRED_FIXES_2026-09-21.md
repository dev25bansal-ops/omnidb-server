# Issues & Required Fixes — 2026-09-21

Current-state catalog of problems in `omnidb-server` 0.1.3 requiring remediation, superseding
[`OmniDB Vault/06 Analysis/Issues and Required Fixes (Sep 2026).md`](../../../OmniDB%20Vault/06%20Analysis/Issues%20and%20Required%20Fixes%20(Sep%202026).md)
for severity, priority, and sequencing.

**This is a delta, not a duplicate.** The 09-11 catalog stays the record of what was found and fixed;
§8 lists what closed since. What changed here:

1. **The severity calibration changed.** The 09-11 catalog scores itself against *"a local-first,
   single-user tool — no network attack surface"* and states at §3: *"No CVSS scores are assigned
   below: none of these are remotely exploitable."* The chosen deployment target is now a **hosted
   multi-tenant service** (`MULTI_TENANCY_THREAT_MODEL.md`). Under that target the 09-11 security
   ratings are void. This document assigns vectors.
2. **Newly found items the 09-11 catalog does not contain at all** — two of them Critical
   (BUG-15, SEC-05), plus a licensing defect (LEGAL-01).
3. **Three claims are retracted** on evidence — see §9. Two were in the 09-11 catalog's open list and
   one is a first-draft error of my own corrected in place.

## 0. How to read this

Evidence class:

| Mark | Meaning |
|---|---|
| **[V]** | Verified today by direct reading of code/config at `D:\OMNIDB\omnidb-server`, with `file:line`. |
| **[M]** | Measured by a command run today (LOC, coverage, `git ls-files`, `ls`). |
| **[A]** | Reported by one of five exhaustive line-by-line reader passes — 22 source modules; 39 test files + both CI workflows; 9 docs + 35 bench scripts/artifacts + scripts + examples; 36 vault notes; and `omni.txt` / the 73-page PDF / `paper/` / `.remember/` / `_verify*/`. Each with `file:line`, spot-checked by me against the source. |
| **[D-doc]** | Asserted by project documentation, not re-verified. A claim with a citation, not a fact. |

Baseline measured today: `omnidb_server/` = **22 `.py` files, 4,603 LOC**; `tests/` = **38 `test_*.py`
files, 5,307 LOC** plus `conftest.py` and **no `__init__.py`** [M]. Runtime Python 3.14.4, `mcp`
1.29.1, `hnswlib` importable. **Coverage 86.8%**
(2,433 stmts / 321 missed) from a live `pytest --cov` run today [M]. Git baseline `d92c4a1`,
118 tracked files, working tree clean before this document.

Severity is calibrated to **hosted multi-tenant** impact: **Critical** = cross-tenant exposure or
permanent memory loss · **High** = one tenant degrades/denies others, or wrong results served as
correct · **Medium** = quality/scale headroom, recoverable data problems, ops blind spots ·
**Low** = maintainability, docs drift.

Priority: **P0** before any tenant-facing deployment · **P1** first two weeks · **P2** this quarter ·
**P3** backlog.

### CVSS method

Scores are computed from the CVSS 3.1 base equations, not asserted. The earlier draft of this
document hand-derived them and got the arithmetic wrong — a `min(5.274, 7.52 × ISC)` expression that
is not in the spec, `ISC = 0.50` where the spec says `A:H = 0.56`, and the Scope-*Unchanged* `PR:L`
coefficient used inside a Scope-*Changed* vector. Those three errors inflated every score in §1.
**CV-01 is therefore closed, not pending:** the vectors below were recomputed with the coefficients
from the specification, and the implementation was validated against two published reference vectors
before any number was taken from it:

| Reference vector | Computed | Published | Match |
|---|---|---|---|
| `AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H` (Log4Shell) | 10.0 | 10.0 | ✓ |
| `AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H` | 7.5 | 7.5 | ✓ |

The two coefficients that were wrong and are now applied correctly: `PR:L` is **0.62** under Scope
Unchanged but **0.66** under Scope Changed, and `UI:N` is **0.85**, never 1.0.

Corrected scores for the vectors in this catalog:

| Vector | Base | Was (draft) |
|---|---|---|
| `AV:N/AC:L/PR:L/UI:N/S:C/C:N/I:N/A:H` — SEC-02 | **7.3 High** | ~~8.1~~ |
| `AV:N/AC:L/PR:N/UI:N/S:C/C:N/I:N/A:H` — SEC-02 if import is unauthenticated | 8.1 High | — |
| `AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:N/A:N` — SEC-05, hosted | **7.3 High** | ~~7.7 as `AV:L`~~ |
| `AV:L/AC:L/PR:L/UI:N/S:C/C:H/I:N/A:N` — SEC-05, local stdio today | 6.2 Medium | — |
| `AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N` — MT-02 | 5.9 Medium | 5.9 (correct) |

**One limit that arithmetic cannot remove.** SEC-05 is rated **Critical** here and scores **7.3**.
That gap is real and is a known blind spot of CVSS v3.1 base, not a mistake: the vector model has no
term for *how many tenants a single row leaks to*, so a per-row cross-tenant disclosure and a
single-tenant disclosure of the same content score identically. Use the severity column for triage
and the score for tooling import; do not let a 7.3 talk you out of a Critical.

**Residual caveat (CV-02 [P2, 0.5 h]):** these are still base-only — no temporal or environmental
metrics — and the `AV` axis depends on the hosting decision in `MULTI_TENANCY_THREAT_MODEL.md` §9
that has not been made. Re-score after that decision, and have a second party confirm the vectors
before any of it reaches a customer or auditor.

---

## 1. Security

### SEC-05 — Consolidation silently de-scopes mixed-agent chunks, writing private transcripts' distillates as globally visible facts [A][V]

**Category:** Security / Cross-tenant disclosure · **Severity:** **Critical** · **Priority:** **P0**
**Effort:** 0.5 d + test · **Deps:** none (independent of MT-02)
**CVSS 3.1:** hosted `AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:N/A:N` — **7.3 High**; as shipped over stdio
`AV:L/…` — 6.2 Medium. The score understates the exposure; see the CVSS-limit note in §0.
**New.** Absent from the 09-11 catalog, which rated SEC-01 "Low-Med / P3 / Q4".

**Problem.** This is the concrete version of the tenancy leak, and it is already in the shipped code.
`_episodes` selects fresh episodic rows **across all agents ordered by `created_ms`**
(`consolidate.py:53-70`). `run_consolidation` then slices that mixed list into chunks
(`consolidate.py:221-223`) and computes the chunk's scope as:

```python
# omnidb_server/consolidate.py:240-242
chunk_source_ids = [e["id"] for e in chunk]
chunk_agents = {e.get("agent_id") for e in chunk}
chunk_agent = chunk_agents.pop() if len(chunk_agents) == 1 else None
```

With the default `consolidation_chunk_episodes = 10`, any chunk spanning two agents yields
`len(chunk_agents) > 1` → `chunk_agent = None` → and `store.remember(fact, agent_id=None, …)`
(`consolidate.py:299-300`) writes a **globally visible** semantic fact distilled from a private
agent's transcript. `agent_id IS NULL` rows are matched by *every* scope
(`recall.py:61-70`, `_episodes` docstring at `store.py`/`recall.py` "None means no scope
restriction"), so the fact is now readable by all tenants.

The in-file comments assert the opposite contract — `:66-67` *"carry the episode's agent scope through
extraction so distilled facts stay inside their scope"* and `:297-298` *"the fact inherits the
chunk's agent scope"*. Dedup is correctly partitioned per scope (`:264-272`, `:148-154`). The
**chunking defeats what the partitioning promises.** Under the single-user model this was invisible;
it is the exact disclosure the multi-tenancy threat model predicts, reached without any hostile
caller action — ordinary consolidation of a shared home does it.

**Fix.** Chunk by `(agent_id, created_ms)` and never mix scopes in one LLM call, rather than by row
order. Add a regression test with two agents' interleaved episodes asserting
`fact["agent_id"] is not None` and equals the source episode's agent. Also cap the per-chunk scope
computation defensively: if a mixed chunk ever reaches the writer, refuse rather than emit
`agent_id=None`.

### SEC-02 — One 512-char trigger pattern can wedge the whole store, because every operation shares one reentrant lock [A][V]

**Category:** Security / Availability · **Severity:** **Critical** (hosted multi-tenant) ·
**Priority:** **P0** · **Effort:** 4–8 h (RE2 path) or 1–2 h (prefix + screening)
**CVSS 3.1:** `AV:N/AC:L/PR:L/UI:N/S:C/C:N/I:N/A:H` — **7.3 High** (`PR:L` = 0.66 under changed
scope; 8.1 would require `PR:N`, i.e. skill import reachable without any authentication, which the
hosted design does not propose).
**Supersedes** `Enhancements and Modifications (Sep 2026).md` E-16 (rated P2).

**Correction first.** An earlier draft of this section claimed there was *"no cap on pattern length,
no cap on pattern count, and no runtime bound on the match."* **That was wrong** and is retracted
(§9). Three controls do exist and are well built:

- `validation.py:14-15` — `MAX_TRIGGER_PATTERNS = 32`, `MAX_TRIGGER_PATTERN_CHARS = 512`, applied to
  creation *and* import alike (`:35-48`), documented at `:11-13` as the *admission* bound.
- `recall.py:29-30` — `_MAX_TRIGGER_QUERY_CHARS = 4096`, `_MAX_TRIGGER_EVALUATIONS = 2048`.
  Oversized queries **skip trigger matching entirely** (`recall.py:141-143`).
- `recall.py:161,170` — the evaluation counter breaks **both** loop levels; verified no count escape.

**The residual problem, stated precisely:** bounding the *number* and *length* of patterns and the
*subject* length does not bound **backtracking time for a single pattern.** `recall.py:22-28` says so
itself:

> *"AUDIT-I06: a compiled-pattern cache bounds memory, not matching work. Python's backtracking
> engine has no execution-time interrupt that can be relied on from this thread, so the recall path
> caps its own inputs."*

Capping inputs is a mitigation, not a bound. `(a+)+$` (well under 512 chars) against a 4,095-`a`
subject followed by `b` is still super-polynomial. So the attacker's recipe is: `skill import` one
bundle with one crafted pattern, then issue one recall with a long query.

**Why that is store-wide rather than request-scoped** — this is the amplifier that makes it Critical
and that the 09-11 single-user framing could not see. `MemoryStore` serialises essentially every
public operation behind one reentrant lock (`store.py:158` `self._op_lock = threading.RLock()`,
`@_serialized` at `:76-90` and ~21 call sites), because one `sqlite3.Connection` is shared with
`check_same_thread=False` (`db.py:87`). The lock is the *correct* fix for BUG-02. Its consequence is
that a recall wedged inside `compiled.search(query)` (`recall.py:167`) holds the store lock
indefinitely, so **every** subsequent operation from **every** tenant queues forever. `server.py:15-16`
dispatches tools on `anyio.to_thread.run_sync` worker threads, so the pool fills and the stdio server
stops responding with no timeout and no error. One tenant, one regex, whole process dead — until an
operator kills it.

**Reproduction (design; not executed — the honest gap).** Import a bundle with
`trigger_patterns: ["(a+)+$"]`, then `recall` a query of 4,000 `a`s plus `b`, then call `stats()`
from a second client and observe it block. Confirm the wedge holds the RLock. Effort to build: 30 min.

**Fix, in order of value.** (1) Screen at import for nested quantifiers / unbounded repetition —
catches the classic shapes, cheap. (2) Use a linear-time engine (`grex`-class RE2 binding, or `re2`)
for trigger matching on a hosted build; this is the only option that removes the class rather than
narrowing it. (3) Run trigger matching off-lock on a bounded thread pool with a wall-clock deadline,
mirroring `search`'s existing 5.0 s deadline (`store.py:32,773`) — note the *store* lock cannot be
interrupted, so the deadline must live outside it. (4) `--trust-triggers` for operator bundles, never
tenants. Item 3 alone is insufficient: a wedged thread leaks a worker even when the caller gives up.

### BUG-15 — `re-embed --model X` produces a home the package can never re-open [A]

**Category:** Security-adjacent / Data availability · **Severity:** **Critical** · **Priority:** **P0**
**Effort:** 0.5 d · **Deps:** none
**New.** Not in the 09-11 catalog. Highest-impact single defect found in this pass.

**Problem.** `make_embedder` constructs `FastEmbedEmbedder()` with **no model argument**
(`embeddings.py:84`), so it always resolves to the class default `BAAI/bge-small-en-v1.5`
(`:69`), and `Config` has **no field for the fastembed model name**. But
`omnidb-server re-embed --embedder fastembed --model some/other-model` (`cli.py:169`) succeeds and
pins `meta.embedder = "fastembed:some/other-model"` (`portability.py:247-250`). Every subsequent
normal open builds `"fastembed:BAAI/bge-small-en-v1.5"`, and `_check_consistency`
(`store.py:197-212`) **refuses the home** with `OMN-006`:

> `database was built with embedder "fastembed:some/other-model" but
> "fastembed:BAAI/bge-small-en-v1.5" is configured`

There is no config surface to fix it. The home reopens only by re-running `re-embed` (which *can*
read the stored name, `portability.py:101-103`). This contradicts `README.md:184-185` — *"`re-embed
--embedder fastembed --yes` migrates an existing home in place"* — for every non-default model.

**In a hosted product this is a customer-triggerable, self-inflicted total lockout:** a documented,
encouraged operation to obtain better retrieval quality bricks the memory home. Fix: add
`fastembed_model` to `Config` (env + TOML), thread it through `make_embedder`, and make
`re-embed` refuse a model it cannot later reconstruct.

### SEC-06 — Bearer token and full transcript sent over an unvalidated URL scheme [A][V]

**Category:** Security / Credential exposure · **Severity:** High · **Priority:** P1 · **Effort:** 1 h
**CVSS 3.1 (est.):** `AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N` — 5.9 Medium

`consolidate.py:101-113` POSTs the **entire episode transcript** plus
`headers={"Authorization": f"Bearer {api_key}"}` to `f"{base_url.rstrip('/')}/chat/completions"`.
Nothing in `config.py:21-46` or `_validate_config` (`:195-232`) rejects a non-`https`
`llm_base_url`, so an operator typo (`http://internal-llm/…`) silently leaks both the credential and
tenant content in cleartext. It is also the *only* outbound data path in the package. Fix: require
`https` whenever `llm_api_key` is set; allow `http` only for loopback. Note `README.md:134-140`'s
threat model says the LLM endpoint is the only egress, which is true but **incomplete**:
`llm.py:16` probes `http://127.0.0.1:11434/v1/models` (loopback, gated off by default at
`config.py:109`) — a second destination worth naming.

### SEC-07 — Skill-zip members are read with no decompressed-size limit [A]

**Category:** Security / Availability · **Severity:** Medium · **Priority:** P1 · **Effort:** 1 h
`skills.py:160,164` call `zf.read()` with **no bound on decompressed size**, so a small bundle can
inflate to gigabytes before any content cap applies — a memory DoS preceding the regex one. Mitigated
but not eliminated by `validation.py`'s content caps *after* read. Fix: total-bytes budget across the
archive plus per-member cap, checked before `read()`.

### MT-02 — `agent_id` carries no authentication; three tables have no tenant column at all [V]

**Category:** Security / Authorization · **Severity:** Critical · **Priority:** P0
**Effort:** blocked on a decision, then 1–2 wk · **CVSS 3.1 (est.):** 8.1 High
**Do not** add a `tenant_id` beside `agent_id`; that reproduces the defect under a new name.

Two independent documentary sources plus the code agree this was never a security boundary:
`Goals and Non-Goals.md:40` — *"Multi-tenant isolation | Single-tenant per process in v1 | v2:
namespace + RBAC"*, a declared non-goal; and `Comprehensive Analysis (Aug 2026).md:73` —
*"Multi-agent namespaces: `agent_id` column (doc schema had it, we cut it)"*, i.e. removed from the
schema and later re-added as recall-scoping convenience. `edges`, `memory_fts`, and `meta` carry no
tenant column (`MULTI_TENANCY_THREAT_MODEL.md` §2), so graph traversal and the FTS index are
structurally unscoped regardless of caller input. Note `agent_id` is **nullable** (`db.py:60`, v2) and
`NULL` means "visible to everyone" (`recall.py:61-70`) — the default is the permissive branch, which
is how SEC-05 becomes a disclosure.

Gate: the owner decisions in `MULTI_TENANCY_THREAT_MODEL.md` §9 — *what a tenant is*, and *how hostile
tenants are to each other*. Then derive identity from the token (the MCP spec is explicit that user
identity comes from the token, not the client), add the column to all three tables, and enforce in
the database so a forgotten `WHERE` cannot leak.

### MT-03 / MT-04 — No authn/authz layer; single-writer forecloses scaling [V]

- **MT-03** Severity High, P0, 1–2 wk. No token validation, audience check, or middleware exists
  anywhere in-process. `Comprehensive Analysis (Aug 2026).md:56` already priced it: *"Remote HTTP
  transport correctly deferred (implies OAuth per MCP spec — its own project)."* Build against
  resource-server semantics with PKCE `S256` and RFC 8707 `resource` in **both** authorization and
  token requests; validate audience; token passthrough is prohibited. Beware the moving target: the
  `2026-07-28` revision removed the `initialize` handshake and session id and deprecated Dynamic
  Client Registration for Client ID Metadata Documents. Do not implement a handshake the next
  revision deletes.
- **MT-04** Severity High, P0 design. One `flock`'d home (`server.py:197-241`, exit code propagated
  by `cli.py:131`) is correct locally and a wall when hosted: no second process, and **`recall` is a
  write path** — access-count bump (`recall.py:215-222`) plus an optional commit at `:229-233` — so
  read-replica routing silently corrupts scoring rather than failing loudly.

### OPS-01 — Plaintext at rest, no key management, no retention, erasure is manual [V]

Severity Medium, P1, 1–2 d. The shipped posture is "use OS disk encryption" (`README.md:139,155-157`)
and SQLCipher is deliberately unbundled (`README.md:163`) — defensible locally, not hosted. Also:
`queries.query` stores verbatim recall/search text (`store.py:103-107`), dumped in cleartext by
`omnidb-server query-log` (`cli.py:209-213`), which in a hosted product is tenant content in a log
table with no export control. `forget` is soft delete; `purge` is manual; nothing enforces a retention
ceiling on content.

### MT-06 — Every documented tenancy bypass has zero regression tests, so MT-02's blast radius is unmeasured [V][A]

**Severity:** High · **Priority:** P0 (for the hosting target) · **Effort:** 1 d · **Deps:** MT-02

`docs/MULTI_TENANCY_THREAT_MODEL.md:28-33` enumerates the concrete isolation bypasses — the global ANN
leg (`recall.py:68`), the unscoped `walk()` with no `agent_id` predicate and `LIMIT 200`
(`store.py:830-855`), the cross-agent `stats()` enumeration (`:975-982`), and the single shared
`vectors.npz` sidecar (`vector_index.py:266-267`). A search across all 39 test modules finds **no test
that asserts any of these four behaviours**, in either direction: none pins "this leak exists today"
and none guards "this leak must not widen."

That makes the picture more precise than "the threat model is unvalidated." MT-02 is not a list of
unknown risks — it is a list of **four known, code-confirmed, untested** ones, each with a line
number. The register cannot be closed by design review alone, because nothing in CI would notice a
fifth appearing. **Fix:** write the four bypasses as *xfail-with-reason* tests today. They encode the
current contract, fail loudly if a bypass silently widens, and flip to passing as each is fixed —
turning the tenancy migration from a prose document into a mechanically tracked checklist, which is
also the only way SEC-05's fix gets a regression guard.

### QA-07 — Recorded as a control that PASSED [V][A]

A catalog listing only defects teaches the next reader that nothing was checked. These passed:

- **No injectable SQL.** The earlier draft said "exactly three SQL strings are built by
  interpolation." **That count was wrong** — a `grep` for f-string SQL finds **30 sites** across
  `db.py`, `store.py`, `recall.py`, `skills.py`, `portability.py`, `consolidate.py` and the two
  adapters (§9 retraction #5). The *conclusion* survives, but it has to be argued on the right
  ground, which is what each site splices:

  | Spliced value | Sites | Caller-controllable? |
  |---|---|---|
  | `_SELECTIVE` (module constant) | 11 | No |
  | `",".join("?" * n)` placeholder runs | 5 | No — the `?` count only |
  | Migration DDL text / `PRAGMA user_version = {int}` | `db.py:132`, `:139` | No |
  | **Column-name lists** | `store.py:493`, `:587` | **Checked by hand, see below** |
  | Literal fragment strings (`agent_id IS NULL OR …`) | remainder | No |

  The two column-name sites are the only ones where an *identifier*, not a value, reaches the string,
  so they are the only plausible injection surface. Both are clean: `store.py:587`'s `sets` can hold
  only the literals `"content = ?"` and `"metadata = ?"`; `store.py:493`'s `all_columns` starts from
  the fixed `base_columns` and can only grow by keys from `extras`, whose three producers are
  `("agent_id", …)` at `:473`, `("trigger_patterns", …)` at `:476` and `("id", int(…))` at `:479` —
  literals in this module, never a caller-supplied field name. Every caller-influenced *value* binds
  with `?`. (`embeddings.py:30`'s `%` is modulo arithmetic; `llamaindex.py:53` splices
  `_GET_ALL_CAP`, a module literal.) `fts_match_query` cannot inject FTS syntax — `\w+` discards
  every metacharacter [A].

  **This is a structural property that no test enforces.** QA-21 [P2, 2 h]: add a test that walks
  every `execute`/`executemany` argument in `omnidb_server/` and asserts each SQL literal is either
  parameter-free or interpolates only names present in a fixed allowlist. Without it, the next
  contributor adding a fourth column name to `extras` breaks the invariant silently — and the
  `search` tool's authorizer, not the string builder, is the only thing standing behind it.
- **No path traversal.** No MCP tool accepts a filesystem path; only argv does. `skills.import_skill`
  uses fixed member names with `zf.read`, never `extract`, so no zip-slip [A].
- **No secrets in source — scan now complete.** No credential, key, token, password, private key or
  cookie value appears in **any** file in either tree, and the credential-shaped literals are
  enumerable: the redaction placeholder `"***set***"` (`cli.py:150`), the self-labelled
  `# export OMNIDB_LLM_API_KEY=sk-no-key-needed` (`examples/offline-llm-consolidation.md:33`),
  `llm_api_key = ""` (`:41`), the words `bearer JWT` in an ASCII diagram
  (`docs/MULTI_TENANCY_THREAT_MODEL.md:76`), and `https://api.example.com/v1|v2` as *fixture content
  to store* (`docs/uat.md:92-93`). A regex sweep for `sk-[A-Za-z0-9_-]{16,}`, `sk-ant-`,
  `gh[poabsr]_`, `github_pat_`, `xox[baprs]-`, `AKIA{16}`, `AIza{35}`, `ya29.`, JWT-shaped strings and
  `-----BEGIN … PRIVATE KEY-----` across `omni.txt`, the 73-page PDF, all 28 authored `paper/` files,
  all 113 `.remember/` files, all 12 `_verify*/` files and every doc/bench/script/example returned
  **zero real matches**; the only hits were Brave's own `sk-button-*` CSS class names and the
  substring `sk-r` inside a word. `.gitignore` does not list `.env` — correct, because none exists
  and the product reads config from `OMNIDB_*` (`config.py:128-173`). **What is present instead is
  path disclosure, covered as QA-22.**
- **The concurrency fix is real.** `check_same_thread=False` is made safe by `_serialized` + `RLock`
  genuinely covering the connection-global `set_authorizer` / `set_progress_handler` hazard
  (`store.py:76-90`, `:782-783`, `:805-807`) — the correct resolution of BUG-02/ARCH-01, and the
  same lock that makes SEC-02 store-wide.

---

## 2. Software bugs

### ARCH-11 — The temporal-supersession differentiator is unreachable from MCP and the CLI [V][A]

**Severity:** High · **Priority:** P1 · **Effort:** 2–4 h
`MemoryStore.supersede` (`store.py:668-696`) and `MemoryStore.link` (`:637-656`) have **zero
in-package callers**; the only invocations are tests (`test_audit_remediation.py:143-145,156,290`,
`test_foundation_v2.py:132`, `test_cli_ops.py:15,63`, `test_agent_flow.py:45,54`). `link` is largely
superseded by the inline `_validate_links` write path (`:520-526`), but `supersede` is the mechanism
behind the marketed "memory that updates instead of accreting" (`Comprehensive Analysis
(Aug 2026).md:70` calls it the sleeper feature; `Implementation Status.md:33` advertises "a manual
`supersede()` API"). **A manual Python API is not a product surface**: an agent talking MCP cannot
supersede anything, so in deployment memory can only accrete. Expose both as tools, or stop citing
temporal supersession as a differentiator. Related: `BUG-09`'s cycle rejection and `DEBT-01`'s
purge-covers-superseded both exist in code and are tested, so the plumbing is there and only the entry
point is missing.

### BUG-13 — `doctor` always exits 0, so no orchestrator can ever see a failure [V]

**Severity:** High · **Priority:** P1 · **Effort:** 15 min + 1 test
`run_doctor` computes and returns the aggregate verdict — `doctor.py:180`
`return {"ok": all(check["ok"] …), "checks": checks}` — and the caller throws it away:
`cli.py:133-137` prints the JSON and `return 0` unconditionally. Note the sibling fix *did* land:
`cli.py:129-131` is `return serve(home)` with the comment *"propagate the lock-conflict exit code
(3)"*. So ARCH-05's `serve` half is closed and the `doctor` half never was — which is why "ARCH-05
fixed" in the 09-11 catalog is correct but incomplete. A health probe that reports `"ok": false` for
every check while exiting 0 is worse than no probe. Fix: `return 0 if report["ok"] else 4`.

### BUG-16 — Five more availability/correctness defects, each small, each with a named site [A]

| ID | Site | Finding | Sev | Pri | Effort |
|---|---|---|---|---|---|
| BUG-16a | `store.py:941-946`, `portability.py:208-213` | Rebuild `unlink`s the old sidecar **before** constructing the replacement; a constructor failure destroys the persisted index (recoverable from rows, but the file is gone and the new index exists only in memory) | Med | P2 | 1 h |
| BUG-16b | `vector_index.py:175-178` vs `:180-186` | `HnswlibIndex.update` never grows capacity while `add` does; `update_memory` (`store.py:604`) calls it, so an update-heavy workload on a full `cap` hits hnswlib capacity errors | Med | P2 | 2 h |
| BUG-16c | `doctor.py:82-84` | A corrupt `pending_reembed` meta value is swallowed to `-1`, then filtered by `> 0`, so a home **with pending recovery work reports clean** — the one check an operator trusts | Med | P1 | 30 m |
| BUG-16d | `db.py:156` | `_heal_torn_v2` matches v2 DDL **by exact string**; any reformat of `db.py:60-61` silently disables the AUDIT-I10 repair path | Med | P2 | 1 h |
| BUG-16e | `server.py:210-232` | `_acquire_home_lock` closes the fd only under `except OSError` (`:229-231`); another exception type after `os.open` leaks the fd and therefore the lock | Low | P2 | 15 m |

BUG-16c deserves emphasis over the others: it is a false-negative in a diagnostic, and the whole
class (silently-green checks) is what §2 of `Issues and Required Fixes (Sep 2026).md` was written to
catch. Also carried: `skills.py:175-183` `overwrite=True` deletes then re-inserts across many
transactions, so a mid-import failure leaves the skill half-installed **and the original gone**
(Med/P2, L-15); `langgraph.py:112-115` put's write-then-retire spans two transactions, and
`_find_row` orders `BY id` returning the *first* match (`:88-90`), so a crash between them leaves two
live rows and `get` returns the **older stale value** — the exact BUG-06 symptom, still reachable
across the crash boundary (Med/P2, L-18).

### BUG-14 — Five exception classes, no shared base [V][A]

Severity Medium, P2, 0.5–1 d. `StoreError`, `ConfigError`, `DbError`, `VectorIndexError`,
`EmbeddingError` are all direct `RuntimeError` children with no common ancestor, enumerated only at
`cli.py:270-271`. A host mapping exceptions to JSON-RPC error objects needs five `except` clauses and
will miss the sixth; under multi-tenancy a mishandled exception at the boundary risks leaking
internals to another tenant. Compounding: `StoreError.__init__` (`:53-55`) bakes the code into the
message (`f"[{code}] {message}"`) *and* keeps `self.code`, so callers cannot recover the bare message
without bracket-parsing. Introduce `OmniDBError(RuntimeError)` as the ancestor.

### BUG-17 — `test_j10` is a green test that will fail the first CI run, and verifies nothing today [V][M]

**Severity:** High (CI-blocking, and it is the *stability* test) · **Priority:** P0 · **Effort:** 2–3 h
**New.** Four independent defects in one test, `tests/test_post_publish.py:175-200`:

1. **It asserts against a file that is not in version control.** `:191-192`
   `results_path = …/ "bench" / "soak-results.txt"` then `assert results_path.exists()`. That path
   matches `.gitignore:30` (`bench/soak-results*.txt`) and is absent from `git ls-files bench/`. A
   fresh CI checkout has no such file, so the assertion **fails on the first CI run**, and on this
   machine it passes only because a Sep-18 file is sitting there locally. A test whose input is an
   untracked, gitignored artifact cannot gate anything.
2. **Its subprocess is decorative.** `:185-189` runs `python -m bench.soak --minutes 1` through
   `subprocess.run(capture_output=True)` and **discards the return value**, so a crash, a non-zero
   exit, or `parser.error` inside the harness is invisible. Worse, the harness deliberately refuses to
   write the asserted file — `soak.py:19` *"Writes soak-results-v2.txt, leaving soak-results.txt
   historical evidence intact"* and `:137-138` `parser.error("the historical soak-results.txt must not
   be overwritten")`. So the run costs ≥60 s and produces exactly zero input for the assertions.
3. **The key it asserts no longer exists.** `:195` `assert "rss:" in text` and `:198`
   `line.startswith("rss:")`, but `soak.py:248-249,341-342` emit `rss_start_mb:` / `rss_peak_mb:`.
   Any future run that *did* write the target file would fail on the renamed key.
4. **It has already failed once and was not fixed.** `.pytest_cache/v/cache/lastfailed` records
   `tests/test_post_publish.py::test_j10_soak_requires_psutil` as failing — the sibling name, i.e. the
   assertion broke, the test was renamed to `…_reports_psutil_status_honestly`, and it now passes
   against the stale artifact rather than against live behaviour.

The cost is also the suite's largest: **60.35 s of a 109.33 s run**. That is a **54.9% floor**, not the
"~73%" this document's first draft repeated; the honest range is 55–65% depending on the machine.

**Fix.** Delete the subprocess; make `soak.py` a callable that returns its report dict; assert on that
dict's keys (`rss_start_mb`, `p95_ratio`, `validation_gate_passed`) in-process; gate the slow path on
`-m slow` or the existing `OMNIDB_RUN_PERF` mechanism; and if a file artifact must be checked, check a
file the test writes itself under `tmp_path`.

### BUG-18 — The test suite is runnable from exactly one working directory, with no signal from the other [V][M]

Severity Low (Medium for a new contributor), P2, 1 h. `D:\OMNIDB\.pytest_cache\v\cache\nodeids` is
`[]` and `lastfailed` is `{"omnidb-server/tests": true}` — a **bare directory key**, which pytest
writes only for a collection/import error. So the last invocation from the workspace root collected
zero tests and died at import, and `.benchmarks/` was created in the same second (Sep 18 03:15) by the
same aborted run. There is no root-level `pyproject.toml`, `conftest.py` or README line saying the
suite must be run from inside `omnidb-server/`, and no `conftest.py`/`__init__.py` in `tests/`, so
`--import-mode=importlib` additionally breaks 14 test modules that rely on the default prepend-mode
`sys.path` injection. Fix is a root pointer or a `tests/conftest.py` that raises a one-line
"run pytest from omnidb-server/" message instead of an ImportError stack.

### Closed this pass — see §8
`BUG-12` (unbounded `k`/`k_fetch`) is **fixed**: `recall.py:12-15` defines `_MAX_RECALL_K = 1000`
with the BUG-12 comment, applied at `:57-58,60,66`, and batched placeholders are additionally bounded
by `store.conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER)` at `:93`. `SEC-04` (`max_content_chars`,
default 100,000) and `QA-02` (unknown-reranker now `StoreError`, `rerank.py:76`) also closed. My
first draft listed BUG-12 as open on the strength of the 09-11 catalog; the code disagrees, and the
code wins (§9).

---

## 3. Performance bottlenecks

All figures below are **single-writer, single-tenant, pre-migration** numbers. They are not SLOs, and
§5 explains why the vault's headline figures do not apply to the shipped configuration at all.

Measured today [M/A, from `docs/perf-report.md`, `bench/*`]: cold start **26 ms** · `remember`
**3.8 k/s** · `remember_many(1000)` **~29 k/s** · recall p50 **0.37 ms** at <1k rows · soak
**1,242.95 ops/s** mixed (brute, 512-d hashing, 4,000 live) · WAL peak **1.84 MB** · RSS 37→69 MB.

| ID | Item | Evidence | Sev | Pri | Effort |
|---|---|---|---|---|---|
| PERF-09 | **No quantized backend; the 100k+ path does not exist.** `SQ8`/`int8` appear nowhere in `vector_index.py`; brute is full-float. The research note called this *"the top gap"* and it was never closed. | [A] `vector_index.py:17-125`, `:123` `np.savez`, `:267` `vectors.npz`; `ANN State of the Art.md:32` | High | P2 | 3–5 d |
| PERF-10 | **Idle RAM is O(corpus) per tenant.** `.npz` is a zip container and cannot be mmap'd. At 768-d/100k that is ~300 MB resident — **multiplied by tenant count**, since each home holds its own index. Converts a "someday" note into a capacity-planning number. | [A] `vector_index.py:114-125` + `ANN:19` sizing math | High | P2 | 4–8 h |
| PERF-11 | **Retrieval quality is capped by toy rerankers** — `REGISTRY` holds only `none` / `lexical_overlap` / `recency_first` (`rerank.py:37,42,54`); no cross-encoder. `Glossary.md:18` *defines* rerank as "SQ8 shortlist then exact-float32 rescoring", which exists only as a bench experiment (`bench/baseline.py:36-40`). | [A] | Med | P2 | 0.5–1 d |
| PERF-12 | **`persist` is a full O(N) snapshot**; the PERF-01 *policy* fix bounds its frequency but not its cost. `vector_index.py:116-121` states the model: `N × (4·dim + 8)` bytes per flush. | [A] | Med | P2 | 1–2 w (E-11) |
| PERF-08′ | Global-ANN-then-filter on the dense leg. Over-fetch shipped (`agent_overfetch = 3.0`, `recall.py:61-66`) but the **recall-quality eval is still pending** — the mechanism exists with its acceptance test missing. | [V-doc] | Med | P1 | 4–8 h |
| PERF-13 | **`import_jsonl` encodes all content in one call** with no chunking (`portability.py:484`), unlike the other two bulk paths which chunk at 512/256 — a multi-GB restore spikes RAM by the whole corpus. | [A] | Med | P2 | 2 h |

**Correction to an earlier draft:** I wrote that `_check_cold_start_budget` and the flush cadence were
"gaps". The `check`/`purge` maintenance operations *do* exist and are correctly implemented — but on
`MemoryStore` (`store.py:860`, `:893`), **not** on `portability.py` as `ARCHITECTURE.md:22` claims;
see DOC-06.

---

## 4. Architectural problems

### ARCH-08 — Zero observability: no logging framework at all [V]

**Severity:** Medium (P1) · **Effort:** 1–2 d · **Deps:** none
`grep -rn "getLogger\|basicConfig" omnidb_server/` → **no matches**. No logger, no correlation id, no
tenant attribution, nothing to ship to a pipeline. `Comprehensive Analysis (Aug 2026).md:107`
prescribed optional `--log-file` JSON with timings and no content; never built. This is a
**prerequisite for §1**, not a downstream nicety: SEC-02 and SEC-05 are undebuggable in production
without attribution, and `consolidate.py:235` storing `str(exc)` in `chunk_errors` (printed to stdout
by `cli.py:221-224`, and able to contain `llm_base_url`) is currently the closest thing to a log.

### ARCH-07 — `store.py` grew 37% while its decomposition sat in the backlog [V][A]

**Severity:** Medium (P2, was Low/P3) · **Effort:** 2–3 d
09-11 logged QA-04 as *"store.py at 815 lines mixes 5 concerns — Low, P3, Q4."* It is **1,115 LOC**
today [M], and the `store.py ↔ recall.py` cycle is still held apart only by function-body imports
(`recall.py:10` reaching into `store` privates, `store.py`'s lazy `from .recall import run_recall`) —
which `QA-04` itself flagged as *"signals the boundary wants formalizing"*. Re-rated because every
tenant predicate from MT-02 lands in this file. `ARCHITECTURE.md`'s module map is the seam;
`recall.py:10` shows where it actually cuts.

### ARCH-12 — The benchmark gates are advisory, and one is self-defeating [A]

**Severity:** Medium (P2) · **Effort:** 0.5 d
- `bench/baseline.py:136-138` computes `best = max(hnsw_scores + [min(sq8_scores)])` then gates
  `best >= 0.95`. That is a **max over the sweep**: it passes on the single best cell (1.0 at
  384-d/1k/ef128) while the worst cell in the same sweep is 0.17 (`bench/results.md:12,29`). A gate
  that reads the maximum cannot fail. `Performance Targets (v1 spec).md:32-34`'s "regression gate
  blocks the merge, ±5%" was never built — no InfluxDB, no Grafana, no nightly — so the only recall
  gate in the repo is the one that cannot trip.
- Three committed stability runs share one `workload_sha256` (`c85dd173…`) and `source_identity_sha256`
  yet disagree: `soak_stab.txt` `validation_gate_passed: False` with `p95_ratio: 4.156`, vs
  `soak_stab1.txt` `True`/2.379, vs `soak_stab2.txt` `True`/2.632. The threshold (~3.0) is crossed by
  **run-to-run variance, not a code change**, and a committed failing gate sits unexplained beside two
  passing ones.
- The soaks themselves run against a home that is **99.7% dead rows** (`live_final: 110`,
  `deleted_final: 42702`) — so the headline stability numbers are measured on a store whose brute
  index never grows (`dead_ratio` returns constant `0.0`, `vector_index.py:60-61`), and
  `_reconcile_index` must later diff ~40k tombstones (`store.py:256-262`).

### ARCH-09 / ARCH-13 — The transport is the least-observed, least-tested surface [V-doc][A]

**Severity:** Medium (P2) · **Deps:** ARCH-08
`server.py` coverage **63.5%** — the lowest of any substantive module, 61 missed statements,
`245-315` uncovered [M, measured today]. The gap is structural: stdio tests run out-of-process, so the
in-process transport never executes under coverage (`Post-Publish Analysis (Aug 2026).md:65`, J-15,
still open). In a hosted deployment the transport **is** the product. Two related facts:
`tests/test_stdio_conformance.py` is a **self-written 75-line smoke test**, not the official suite, so
**ADR-6's gate ("official conformance suite passes for the tools profile, run in CI") is unmet** —
`Implementation Status.md:17` claiming conformance via "official client" overstates it, since client
≠ conformance suite. And `server.py:22` uses `FastMCP`, which `MCP Ecosystem.md:21` explicitly flags
as *"popular but community, not official"*.

### ARCH-14 — Dead directory and dead markers [V][A]
Low/P3/5 min. `omnidb_server/_internal/` is empty with no `__init__.py`, referenced nowhere [V];
combined with `namespaces = true` (`pyproject.toml:47`), any stray non-`.py` file added there becomes
an implicit namespace package in the wheel. `pyproject.toml:75` excludes a directory
`build_check_dist` that does not exist. `docs/__init__.py` is 0 bytes [V] — a package marker left in a
docs tree. `py.typed` ships while `server.py:228` carries a `# type: ignore` and several params are
unannotated (`store.py:317`, `:93`), so the marker promises more than the code delivers under the
intentionally lenient mypy config (`pyproject.toml:74`: *"a green gate that stays green, not full
strictness"* — honest, but that is the statement of an advisory gate).

---

## 5. Documentation and knowledge-base integrity

Treated as a defect class because `ARCHITECTURE.md:5-7`, `tests/test_audit_remediation.py:4`, and
`tests/test_enhancements_p1.py:3` all cite the vault as the authority for ADR numbers used in code
comments [V]. When the vault is wrong, the code inherits it.

### LEGAL-01 — The package claims Apache-2.0 and ships no license file [V][A]

**Severity:** High · **Priority:** P0 for any publication · **Effort:** 5 min
`pyproject.toml:10` is `license = { text = "Apache-2.0" }` and `MANIFEST.in:3` is `include LICENSE`,
but **`ls LICENSE*` → no such file** [V, verified directly]. The built sdist therefore advertises a
license it does not carry, and PyPI's license field is already populated for published 0.1.x releases.
This is the clearest packaging defect in the repository and the cheapest to fix. Add `LICENSE`, and
prefer PEP 639 (`license = "Apache-2.0"` + `license-files`) over the deprecated table form.

### DOC-06 — `ARCHITECTURE.md` contains six specific factual errors [A][V]

**Severity:** Medium · **Priority:** P1 · **Effort:** 2 h
1. `:22` assigns `check` and `purge` to `portability.py`; both are on `MemoryStore`
   (`store.py:860`, `:893`) and **`portability.py` defines neither**. `:20` then *also* credits them
   to `store.py` — double-credit plus an invention.
2. `:33` describes *"an importer-only explicit-id hook"*; `remember(_explicit_id=…)` (`store.py:381`)
   has **zero callers** — `import_jsonl` writes its own `INSERT` (`portability.py:491-504`). The
   behavior exists; the hook described does not.
3. `:24` lists four `cli.py` subcommands; there are **20**.
4. `:63` says "the top `max(4k, k)` fused ids are re-fetched" — the code re-fetches the **entire**
   fused pool (`recall.py:92-102`) and truncates *after* (`:104`). The AUDIT-I09 comment at `:91` is
   right and the prose is stale. `max(k*4, k)` is itself a no-op for `k ≥ 1`.
5. `:105-107` describes the authorizer as reads-plus-one-pragma without noting that
   **`SQLITE_FUNCTION` is allowed unconditionally** (`store.py:753`) — the actual residual risk of
   `search`, and the one allowlist item a reviewer must see. (Cleared elsewhere: it cannot reach
   `load_extension`, because Python's sqlite3 disables extension loading.)
6. **`corpus_mode` is documented nowhere in `ARCHITECTURE.md`** despite zeroing *both* rerank weights
   (`recall.py:110-111`, justified by a measured 40–56% nDCG loss at `config.py:66-71`). It is the
   single most behavior-changing flag in the config and it is invisible in the architecture doc.

### DOC-01 — Four notes give four test counts; three give three size counts [V]

`Implementation Status.md` mutates 32 → 126 → 203 in place; `Enhancements…:8` says 228; its own
status block says 244; `Post-Publish Analysis:15` says **44 `.py` / ~1,985 LOC / 127 tests**;
`Issues and Required Fixes:8` says **21 files / ~3,300 LOC**. On disk: **22 files / 4,603 LOC** [M].
A reader cannot tell which number is current. Severity Medium because the counts are load-bearing for
the coverage and ceiling claims below.

### DOC-02 — The stated LOC ceiling has been passed 3.8× by the project's own predicted failure mode [V]

`Comprehensive Analysis (Aug 2026).md:111` — *"Honest ceiling: ~1,200 lines doing one job well.
Biggest risk is re-inflation into the original ambitions; keep cutting."* Actual 4,603 [M]. The
from-scratch database design was abandoned **partly because it outgrew a solo maintainer's honest
budget** (`ADR-1`, "honest production-grade scope is roughly 18-36 months"). Code at 3.8× the
re-ratified ceiling, with `store.py` +37% *after* its split was deprioritised, is the named risk
materialising. Either re-ratify the ceiling with reasons, or cut. Leaving the doc and the repo telling
two stories is the outcome that killed plan one.

### DOC-04 — ADR-4's table-valued functions were never built, and three notes still assert they exist [A][V-doc]

`ADR-4` decides *"Expose plain SQLite SQL plus TVFs `VECTOR_SIM(col, $q, k=10)` and
`MEMORY_RECALL` … hybrid SQL with zero planner code."* `Implementation Status.md:29` defers them to a
Go/Rust port that has not happened. The shipped equivalent is Python-side fusion. **`Glossary.md:32-33`
is written in the present tense** — `VECTOR_SIM`/`MEMORY_RECALL` "are exposed this way", crediting
eponymous virtual tables as "the mechanism that makes TVFs work" — and a code-wide grep confirms those
three identifiers appear **zero times** in any `.py` file, with no `create_function` or
`create_module` anywhere. `ARCHITECTURE.md:142` gets it right. Three of the four `02 Design Spec/`
engine files additionally describe a B+tree / MVCC / LSM / SQL++ system that was rejected, and
`Graph Engine` / `Storage and Persistence` / `Transactions and Concurrency` each carry internal
callouts admitting corruption or inconsistency (`:29`, `:26`, `:7`) — yet `00 Home.md` still lists
them flatly under "The v1 Design Spec". Banner them at the note level, not just in Home.

Two further `Glossary` entries are false of the shipped code: `:18` **Rerank** describes
float32-rescore-over-quantized-shortlist, which exists only in `bench/baseline.py`; `:42` **LSM** is
correctly obituarised but `:61` **Consolidation** is defined as a *"**Scheduled** background
process"* when **no scheduler exists** — `consolidate` is an opt-in tool plus a CLI subcommand, and
the only background loop is WAL-checkpoint-plus-flush (`server.py:284-291`). Same file, same defect:
`00 Home.md` links neither of the two Sep-2026 analysis notes nor `Post-Publish Analysis`, so three
of the four largest notes are unreachable from the index.

### DOC-05 — Stale claims surviving inside current documents [V]

`Issues and Required Fixes (Sep 2026).md` still asserts DEBT-02 (wheel ships `tests`/`docs`) although
`pyproject.toml:46` is now `include = ["omnidb_server*"]`, and still asserts QA-06 version drift
although `pyproject.toml:7` and `__init__.py:8` are both 0.1.3. Its header *"21 files, ~3,300 LOC"* is
stale by ~1,300. `Post-Publish Analysis:14` and `Implementation Status` both market *"no native
dependencies"*: `numpy>=1.26` is **mandatory** and native (`pyproject.toml:12`); `hnswlib` is optional
(`:30`). The true claim is "no native deps beyond numpy, hnswlib optional". The one item from 09-11
that I re-verified as **genuinely, well fixed** is BUG-07: `db.py:129-143` wraps DDL + postcondition +
version bump in one transaction with rollback, carries `AUDIT-I10` at `:135`, and adds
`_heal_torn_v2` (`:146-156`) to repair pre-atomic torn homes.

### DOC-07 — Every `ADR-n` citation points outside the repository [V]

`ARCHITECTURE.md:5-7` and `README.md:5` reference `../OmniDB Vault/`, and ADRs 2, 4, 7 are cited in
code-facing prose — but the vault is **not in the git repository**. So the published artifact contains
traceability links that resolve to nothing for anyone who clones it. Worse, the far more numerous
`AUDIT-*`, `PERF-*`, `ARCH-*`, `SEC-*`, `BUG-*`, `QA-*`, `J-*`, `DEBT-*` tags that pepper the code
comments are **not indexed in any top-level document**; `CHANGELOG.md:36` points at `docs/AUDIT.md` as
their home. Either vendor a condensed `docs/DECISIONS.md` into the repo, or stop citing what a clone
cannot open.

### DOC-08 — The design half of the vault is missing the entire tuning surface [A]

None of the 11 design specs or 8 research notes documents the ~30 config knobs an operator actually
needs: `k_fetch 50`, `max_search_rows 500`, `persist_every_ops 64 / _every_s 0.25 / _min_ops 8 /
_max_s 5.0`, `dead_compact_ratio 0.2`, `ef_search 64`, `agent_overfetch 3.0`, `max_content_chars
100_000`, `query_retention_rows 100_000`, `dense_weight 0.7 / sparse_weight 0.3`, `rrf_k 60`,
`consolidation_*`, `corpus_mode`, `ollama_autodetect` — nor the store internals (`max_query_steps
5_000_000`, 5.0 s wall clock, 32 MB cap, `_RECONCILE_BATCH_CAP 500`, batch 512, `LIMIT 200`, hops ≤8).
`README.md:112` documents 9 of ~28 derived `OMNIDB_*` names; `OMNIDB_DENSE_WEIGHT`,
`OMNIDB_CORPUS_MODE`, `OMNIDB_EF_SEARCH`, `OMNIDB_MAX_CONTENT_CHARS`, `OMNIDB_QUERY_LOG`,
`OMNIDB_VECTOR_INDEX`, `OMNIDB_COLD_START_BUDGET_MS` and ~10 more are undocumented. Two of the
specs' recommended values were also not honored: `ANN State of the Art.md:32` asked for `efS=50`, the
shipped default is `ef_search = 64` (`config.py:88-89`) — trivial, but it shows the specs are not a
reliable description of the configuration.

### DOC-09 — The audit's own acceptance evidence is unboundable and, for the two gates that matter, contradicted by the artifacts it cites [V][M]

**Severity:** **Critical** for the *audit trail*, Low for the *product* · **Priority:** P0 ·
**Effort:** 0.5 d (re-measure + rewrite), then DOC-16 makes it self-enforcing

`docs/AUDIT.md` records 22 findings as fixed and, at `:375-411`, gives its acceptance evidence. Two of
those passages cannot be reproduced from anything in the tree.

**(a) `AUDIT-I12` cites numbers that are not in the file it cites.** `:196` states the saved 60.1 s run
"records recall P95 increasing from **3.30 to 34.38 ms, 10.42 times**… Overall recall P95 is
**32.10 ms**. RSS rises **36.9 to 61.9 MB**; peak WAL is **1.88 MB**", sourcing
`bench/soak-results.txt:1-9`. That file's actual contents are `p95_first_third_ms: 4.22`,
`p95_last_third_ms: 17.31`, `recall_p95_ms: 16.03`, `rss: 37.3 -> 89.9 MB`, `wal_max_mb: 2.02` — a
**4.10×** creep ratio, not 10.42×. `docs/perf-report.md:48-52` describes the same saved run as a third
distinct set (21,280 ops / 2,660 cycles / p95 11.84 ms / 3.40→12.95 / RSS 37.1→68.8 / WAL 1.84) and
concludes it **"passes all three gates"**, while the file itself fails the 4× gate. Three documents,
three mutually exclusive descriptions of one 180-byte artifact, one of which reports the opposite
verdict.

**(b) The remediation section's post-fix numbers match no file either.** `:394` attributes to
`bench/soak-results-v2.txt` and `bench/growth-results.txt`, "both bound to `source_identity_sha256:
c1e17113…`", a 2-minute run at "p95 ratio **3.758** ≤ 4.0 … **381 ops/s**" and growth "**629 ops/s**",
recall p95 "≈1→**5.6** ms". On disk both files carry `source_identity_sha256: 52de74cb…`;
`growth-results.txt` reports **1,242.95 ops/s** with bucket p95 rising 1.919→3.761 ms. No file in the
tree contains `c1e17113`, 3.758, or 381. `docs/DATASTORE_SPIKE.md:28` reports the honest picture from
the same file — "p95 creep ratio **4.39 (gate fail)**" — so `AUDIT.md` + `perf-report.md` are in direct
verdict conflict with `DATASTORE_SPIKE.md` and with the artifacts.

**(c) I must disclose a fourth cause, which is mine.** `bench/soak-results-v2.txt` was overwritten at
**10:03 today** by the `pytest --cov` run I started in this session, because `test_j10`
(BUG-17) shells out to `python -m bench.soak --minutes 1` as a side effect. Its current contents —
`duration_s: 60.003`, `p95_ratio: 6.070`, `validation_gate_passed: False`, `ops_per_second: 619.24`,
`rss_start_mb: 39.871`, `rss_peak_mb: 71.961` — are **my run, not the auditors'**. Because
`.gitignore:29-31` excludes every one of these filenames, whatever the Sep-18 remediation pass wrote
there is unrecoverable, and my numbers are now the ones a reader will find and compare against
`AUDIT.md`. That is exactly the failure mode BUG-17 describes, promoted from hypothetical to actual.
It also means **(b) above is only partly attributable to the auditors** — one of the two candidate
explanations is my own overwrite. Both readings are unfavourable to the docs, and the fix is the same.

**The structural defect, which is the reportable finding:** the evidence for the project's two
availability claims lives in files that are gitignored, so it is unverifiable by construction, and the
test suite overwrites them as a side effect of running. `AUDIT.md:194` citing
`bench/soak-results.txt:1-9` is a tracked document citing an untracked, ignored, mutable file.

**Fix.** (1) Commit the named result files or stop citing them — an evidence artifact a test can
overwrite is not evidence. (2) Re-measure the soak and the growth run on the *final* runtime, record
`source_identity_sha256` in the doc alongside the numbers so a mismatch is self-announcing, and rewrite
`AUDIT.md:196`/`:394` and `perf-report.md:48-52` from that run. (3) State the verdict as it measures:
if the ratio is 4.4 or 6.1 against a 4.0 gate, the gate **fails**, and `I12` is not fixed.

### DOC-10 — `research-v3-results.md`'s cross-encoder column disagrees with its own JSON in all 16 cells, and three documents quote both generations as if they were one [V][A]

Severity High, P1, 2 h. The markdown's `omnidb + cross-encoder` means (nfcorpus/minilm 0.3396,
scifact/minilm 0.6875, fiqa/minilm 0.3881, scidocs/minilm 0.1872) do not equal
`research-v3-results.json`'s `omnidb-ce` per-query means (0.3134, 0.7068, 0.3732, 0.1755); the base
hybrid cells match exactly, so only the CE column drifted. The `rerun_reranker.py:4` provenance guard
exists *because* of a prior silent-clobber incident and never re-rendered the markdown. The model
changed too: the md header (`research-v3-results.md:9`) names `cross-encoder/ms-marco-MiniLM-L-6-v2`
while every JSON cell is tagged `BAAI/bge-reranker-base`. **Three cells flip sign** — scifact/minilm
md −0.0177 vs JSON **+0.0016**; nfcorpus/e5-small md **+0.0025** vs JSON **−0.0275**; fiqa/minilm
md **+0.0085** vs JSON **−0.0064** — which is the difference between "the reranker helped" and "hurt"
on each. Downstream, `rerank-window-results.md:115` and `research-v3-results.md:244-249` quote the
ms-marco-era numbers, `probe_rerank_window.py:4-5` quotes the bge-era numbers, and
`research_bench_v3.py:80-83` preserves a third set matching neither. The paper's headline negative
result is safe — it draws on `review_probe_deltas.json`, and `DATASTORE_SPIKE.md:31`'s "never helps:
best +0.0067, worst −0.0541" is sourced from the probe JSONs — but **the v3 CE grid must be treated as
void** until re-rendered, and the hand-written Conclusion §2, which reads "on scifact it was −0.0538…
on fiqa it was noise (−0.0099)", is computed from the stale column.

### DOC-11 — Measured-behaviour documents carry integrity hashes, and one of the two no longer matches [V]

Severity Medium, P2, 30 m. `docs/mutation-report.md` binds itself to `scoring.py 5d8f6e7f…` and
`fusion.py 5ae71f86…`. `scoring.py` still hashes to its recorded value; **current `fusion.py` hashes
`5d33d7e7…`** — the file changed after the report was written (the measured-behaviour docstring) and
the report was not updated. This is the one place in the repo where a document declares how it should
be invalidated, and the mechanism caught a real drift. It is also unenforced: nothing runs the check.
Fold it into DOC-16's gate rather than fixing the hash by hand.

### DOC-12 — `test_audit_remediation.py` and `test_enhancements_p1.py` document their provenance inside the repository, so the tracked-file citation gap reaches the tests too [V][A]

Severity Low, P2, 15 m. Two test modules open by citing vault notes that live one directory above the
repo root and are not versioned with it — `tests/test_audit_remediation.py:1-5` points at
"`../OmniDB Vault/06 Analysis/Issues and Required Fixes (Sep 2026).md`" and
`tests/test_enhancements_p1.py:1-4` at "`../OmniDB Vault/06 Analysis/Enhancements and Modifications
(Sep 2026).md`". Same shape as DOC-07 but worse: a test file is the one artifact a clone is guaranteed
to have, and its stated reason for existing is guaranteed not to be. Copy the finding registers into
`docs/` and repoint, or drop the citation to a bare prose note.

### DOC-13 — Test-count and baseline-figure drift is now a chain, not a straggler [V][M]

Severity Medium, P2, 1 h. `pytest_out.txt:5` records **271 passed / 4 skipped in 109.33 s** (Sep 20
12:20). `docs/AUDIT.md:379` and `docs/coverage-report.md:6` both claim **268 passed, 4 skipped**; the
vault claims 203, 228 and 244 in three places (DOC-01). Nothing was updated when the count moved.
Separately, the *vector* baselines are a full generation behind `bench/results.md`: `perf-report.md:31-38`
tabulates 0.873 / 0.383 / 0.823 / 0.359 at ef128 against the file's current **0.888 / 0.519 / 0.854 /
0.395**, `README.md:128` still says "HNSW at ef=128 (0.837)" against **0.854**, and `AUDIT.md:339`'s
100k row cites 0.281/0.240 with P95 0.628/0.825 against **0.378/0.278, P95 0.864/1.101**. `numpy`
moved 2.3.5 → 2.5.2 between the report and the artifact, which explains the deltas and does not excuse
them: three documents each carry a "measured locally" header and none matches the one file all three
claim to summarize.

### DOC-14 — Two attribution errors that already propagated into an argument [V][A]

Severity Medium, P2, 30 m.
1. **`scifact` → `scidocs`.** The finding "sparse-heavy fusion weights are safe up to ≤0.3" was
   measured on **scifact** (`research-results.md:47,64`, `fusion.py:22`).
   `docs/DATASTORE_SPIKE.md:32` and `docs/MULTI_TENANCY_THREAT_MODEL.md:193-194` both restate it as
   scidocs — and the spike then uses it to argue about per-tenant tuning burden (`:80`), so the corpus
   mislabel is load-bearing in a document whose purpose is to inform an architecture decision.
2. **"Same-dimension swaps are *currently* undetectable."** `DATASTORE_SPIKE.md:56-57` presents
   `AUDIT-I03`'s pre-fix finding as the state of the tree, which contradicts `store.py:200-206`'s
   name-mismatch hard refusal and `ARCHITECTURE.md:80-83`'s `reembed_pending` description "covering
   same-dimension model swaps". Three-way contradiction about current code, resolved against the spike.

### DOC-15 — `AUDIT.md`'s effort totals are wrong against its own table [V]

Severity Low, P3, 15 m. The register's ranks 1–20 sum to **136.5–241 h** and the full 22 items to
**143.5–254 h**; the document states 147.5–266 and 154.5–279, and derives "19–35 engineer-days" from
the inflated pair (correct: ~18–32). A planning number that does not add up to its own rows will be
quoted onward.

### DOC-16 — The vault's market-claims refutation is now *under*-credited, and its two surviving load-bearing claims are wrong for different reasons [V][A]

Severity Medium, P1, 1 h. This reverses the direction of DOC-01..DOC-15: here the knowledge base
**understates** what the project's own evidence shows, because the raw evidence was saved and the
conclusion was not re-read.

`_verify/` is a 2026-08-21 01:56–02:00 fact-check sweep. Four of six direct fetches failed (a Mordor
404, a pre-JS Google shell, a DuckDuckGo Instant-Answer stub returning `"id":"just_another_test"`, and
Oracle's perimeter firewall twice, incident numbers `0.465f2c31.1787257674.677f691` and
`…7850.6781199`). The three Brave SERP pages that succeeded are the ones nobody re-read:

- **`mordor_agentic_vec.html` corroborates the number and refutes the year.** The live report page
  states "Market Size (2026) USD 0.57 Billion | Market Size (**2031**) USD 1.73 Billion | Growth Rate
  (2026–2031) **24.86% CAGR**". The PDF's `$0.57B → $1.73B by **2030**` (`omni.txt:445-446`) is wrong
  by one year, and its "$0.57B in 2026" is right. `Market Claims Fact-Check.md` recorded Mordor as
  "could not be corroborated" — so the project's own differentiator narrative is more solid than its
  own audit note says, and the audit note is stale.
- **`brave2.html` dates the Oracle announcement.** `blogs.oracle.com/machinelearning/announcing-the-oracle-autonomous-ai-database-mcp-server`,
  "**24 December 2025**", GA for versions 19c and 26ai. The PDF's "**March 2026** announcement"
  (`omni.txt:427`) is wrong by three months, and the vault's "Unconfirmed… no trace of the specific
  March 2026 announcement" is superseded — the trace was in `_verify/` all along.
- **The two claims the vault rejects stay rejected.** "A taxonomy published on Reddit's r/Database"
  (`:424-425`) and "The StackOne 2026 agentic AI tools landscape report" (`:447`) have no supporting
  fetch; `brave3.html` shows TigerGraph shipped *an* AI product (Savanna), which does not evidence
  "TigerGraph, CockroachDB, and Snowflake have all announced comparable roadmaps".

**Why this is P1 and not a footnote:** `Comprehensive Analysis (Aug 2026).md:5` still asserts "The
whitespace is real (per Aug research): no shipping engine unifies local relational + vector + graph +
agent-memory behind MCP," and the strategy section cites the market sizing. The sizing is defensible
with the year corrected; the Reddit taxonomy and StackOne report are not, and `omni.txt` — the 73-page
PDF's own text — is the sole carrier of all three. Anyone who re-markets from the PDF will re-publish
one wrong year, one wrong date, and two unverifiable citations. Fix: update `Market Claims
Fact-Check.md` from `_verify/brave{1,2}.html` and `mordor_agentic_vec.html`, correct the year and date
wherever they are quoted, and mark the Reddit/StackOne citations **unusable**.

### DOC-17 — `omni.txt` is a lossy derivative of an AI-generated PDF, and its table corruption is not marked where it is read [V][A]

Severity Medium, P2, 2 h. `omni.txt` is the `pdftotext -layout` extraction of
`OmniDB_v1_Design_Document.pdf`, not an independent source: the title page's letter-spacing artifacts
(`:1-3` "V1.0 ─ DESIGN DOC / U NI F I ED DATAB ASE ENG I NE ─ JU LY 2026") and the embedded per-page
footers (`:77` "OmniDB ─ Page 1" … `:3194` "Page 72") are extraction residue, and the whitespace-
normalised text lengths match to within 0.06% (143,433 vs 143,522 chars). Three of its tables arrived
**rotated by one row** — `:779-790` puts "~200 (variable)" under "Leaf nodes" and "Doubly-linked list"
under "Split policy"; `:807-831` shifts every type-system size down one; `:994-1003` shifts the HNSW
storage breakdown — and the vault knows (`Relational Engine (v1 spec).md`: "Table 6 … arrived with
shifted columns; the values above were reconstructed from surrounding text"). The corruption is
documented **in the vault**, not **in `omni.txt`**, so any reader of the extraction gets the wrong
values with no warning. The PDF itself carries `Author: Z.ai`, `Creator: Z.ai`, `Producer: http://z.ai`
and a `NotoSerifSC-Regular` fallback in an all-English document — it is machine-generated — while its
title page stamps "CONFIDENTIAL ENGINEERING REVIEW". Both facts belong at the top of the file.
Fix: prepend a provenance header to `omni.txt` naming the PDF, the tool, the rotated tables and the
generation, or delete it in favour of the vault's `02 Design Spec/*` restatements, which already
reconstructed the values.

### DOC-18 — The paper ships as three mutually contradicting generations, and the published `.docx` has placeholder authors and retracted claims [V][A]

Severity High for the artifact, P1, 2 h. `paper/` holds three states of one paper. The oldest
(`paper.md` → `build_docx.js` → `omnidb_ieee_access.docx`, on disk since **Sep 15 06:15**) states
"three dense encoders from two model families" and Spearman **ρ=0.60 at n=16, p≈0.42**; the current
`.tex` uses **four** encoders, **eighteen** fused-hybrid window-cells, reports "**ρ=0.40 with only
n=4 cells and p=0.60**, so we report it as a descriptive hint" (`.tex:548-550`), and retracts the
"helps in three of four corpora" verdict to "robust on two of the four, a third only without
multiple-testing correction". **The DOCX was built from the stale source**, so the shareable Office
artifact on disk asserts claims the paper now explicitly withdraws. Its author block is template
fiction verbatim — `paper.md:3-6` `First A. Author, Student Member, IEEE` /
`author@example.org` / `supported in part by [sponsor and grant number to be added]` — which the current
`.tex:18-22` comment identifies as disqualifying ("The IEEEtran template's default placeholders …
must not appear in a submitted manuscript") while substituting a *different* fiction
(`A. Author` / `a.author@university.edu`). Reference list entries were rewritten between generations
rather than reflowed (Robertson & Walker 1994→1995; the bm25s URL; SQLite and BGE citations replaced
wholesale). Build hygiene: 9 underfull-box warnings survive in `omnidb_ieee_access.log`, and
`build/p1.log:46` still says "Label(s) may have changed. Rerun to get cross-references right". No DOI,
no arXiv ID, no venue footnote, no ORCID; venue inferable only from `\documentclass[journal]{IEEEtran}`
and `\markboth{IEEE Access, 2026}`. Fix: regenerate the DOCX from the current `.tex` or delete it, and
resolve authorship before anything leaves the machine.

---

## 6. Code quality

| ID | Item | Evidence | Sev | Pri | Effort |
|---|---|---|---|---|---|
| QA-12 | **Free-string config with no enum.** `embedder` accepts any string; `make_embedder` compares against exactly one value (`"fastembed"`) so `"hashing"`, `"auto"` **and every typo** fall through to hashing (`embeddings.py:80-100`) while `_check_consistency` happily pins `hashing-v1`. Same shape for `vector_index` (`:263-310`). `OMNIDB_EMBEDDER=clip` silently degrades retrieval with no signal. | [A] | Med | P1 | 1–2 h |
| QA-13 | **Three sources for one schema version.** `meta.schema_version` hardcoded `"3"` (`store.py:188`), `SCHEMA_VERSION = 3` (`db.py:6`), `PRAGMA user_version` (`:139`). Any migration touching one and not the three is a latent desync. | [A] | Low | P2 | 1 h |
| QA-14 | **"Half-life" is a time constant.** `scoring.py:26` computes `exp(-age/τ)`, whose actual half-life is `τ·ln2 ≈ 116 h` for the configured `half_life_hours = 168` — so `README.md:113` and the config name both misdescribe the tuning knob by ~44%. Faithful to the spec's error. | [A] | Low | P3 | 15 m |
| QA-15 | **`md5` for feature hashing** (`embeddings.py:30`) — not a crypto use, deterministic across runs (unlike `hash()` under `PYTHONHASHSEED`), and the right choice. Will trip SAST; document the intent at the call site so the next reviewer does not re-file it. | [A] | Low | P3 | 5 m |
| QA-16 | **Error-code gaps.** Codes 001, 007–010, 012, 014–019 are absent; `E_CONSOLIDATION_FAILED` (`store.py:47`) is **never referenced** — `consolidate.py:251` uses the literal `"OMN-013"`. `cli.py:293` passes a bare `E_CONSTRAINT`-shaped literal. Dead constant plus magic strings where a named code exists. | [A] | Low | P2 | 1 h |
| QA-17 | **`lru_cache(maxsize=1<<20)` on `_bucket`** (`embeddings.py:26`) is up to 1,048,576 `(str,int)->int` entries — justified as *"where the vectorization win lives"* on bulk import, but an unbounded-in-bytes memo and the main RSS-growth candidate on heterogeneous corpora. Needs a bytes bound, not an entry count. | [A] | Low | P3 | 2 h |
| QA-18 | `CHANGELOG.md` breaks its own Keep-a-Changelog shape: `### Fixed` appears **three times** inside `[0.1.3]` (`:10, :88, :209`), `### Added` three times, and `## [Unreleased]` (`:214`) sits **below** the released section, inverting the chronological order it claims at `:5-6`. | [A] | Low | P3 | 30 m |
| QA-19 | Duplicated `_TOKEN = re.compile(r"\w+")` in four modules; `doctor.py:80` does a function-local `import json as _json` to dodge shadowing. No unused imports anywhere — checked across all 22 files [A]. | [A] | Low | P3 | 30 m |
| QA-20 | `fusion.py:28` `zip(rankings, weights)` silently drops legs on length mismatch, contradicting `:14`'s "must match". `recall.py:204-205` assumes metadata is a dict, so a foreign home with a JSON-array metadata raises `TypeError` inside recall. `langgraph.py:107` round-trips the entire framework value into `metadata["_value"]`, storing content **twice** (2× amplification, doubled exposure surface). | [A] | Low-Med | P2 | 2 h |
| QA-21 | **The no-injection invariant has no test.** 30 interpolated-SQL sites stay safe only because every spliced value is a module constant or an internally-derived column name (see QA-07). Nothing prevents a fourth `extras` key from arriving as caller data. Enforce it mechanically: assert every SQL literal in `omnidb_server/` is parameter-free or interpolates allowlisted names. | [V] | Med | P2 | 2 h |
| QA-22 | **Local paths in tracked files.** `bench/research-v3-results.json:31100` commits a captured traceback containing `C:\Users\dev25\AppData\Roaming\Python\Python314\site-packages\qdrant_client\local\persistence.py` — the only tracked file in the repo carrying the operator username. Nine `bench/` scripts hardcode absolute paths (`research_bench_v3.py:455` `D:/OMNIDB/tools/jdk-21.0.5+11`, `:464` and `probe_rerank_window.py:49` `E:/omnidb-bench-tmp`, `sweep_rerank_nfcorpus.py:49` `D:/OMNIDB/omnidb-server/.hf-cache`, `compute_probe_27.py:8,13`, `compute_review_stats.py:34,38`, `sweep_rerank_scidocs.py:42,175`, `sweep_rerank_scifact.py:53`), so the benchmarks are not reproducible on any other machine and the scidocs section of `rerank-window-results.md` is not reproducible from tracked files at all. `examples/claude-desktop-config.example.json:15` ships `/Users/yourname/.omnidb` and `pyproject.toml:40` `github.com/yourname/`. | [V][A] | Med | P1 | 3–4 h |
| QA-23 | **TLS verification disabled in the v3 harness** — `bench/research_bench_v3.py:58` builds `httpx.Client(verify=False)` for Hugging Face downloads. Benchmark-only, not shipped code, but the automated review pass flags this pattern **HIGH** and it is exactly the line a copy-paste carries into a request path — the shipped equivalent (`llm.py:16` `httpx.get`) does verify, and that asymmetry needs to stay visible. Same file, same finding: `docs/DATASTORE_SPIKE.md` cites that harness's qdrant latencies where 2 of 5 runs errored. | [A] | Med | P1 | 15 m |
| QA-24 | **Dead and empty artifacts.** `paper/_wrap.js` reads `_real.js` and writes `_real_instr.js`; neither exists — dead script by construction. Four empty directories at `D:\OMNIDB`: `.claude/` (0 files), `.codebuddy/` (0 files, one empty `teams/`), `tt-nupkg/` (0), `.benchmarks/` (0). `docs/__init__.py`, `bench/__init__.py`, `scripts/__init__.py`, `examples/__init__.py` are all 0-byte package markers for directories that are not importable packages (ARCH-14). | [V][A] | Low | P3 | 30 m |
| QA-25 | **`.git` carries ~1.6 GB of orphaned loose objects while the two commits carry 7.16 MB** (`git ls-tree -r -l` sum over 118 files), including objects >50 MB each — the residue of the `git add` of model and cache trees that was staged and then un-indexed during this session's baseline work before any commit existed. Nothing unreachable is recoverable and nothing sensitive is committed, but the clone/pack payload is 200× the content. Fix is `git gc --prune=now` — **destructive to any genuinely unreferenced object, so not run without approval.** | [M] | Low | P2 | 10 m + approval |

**Test-quality note [M].** The suite is strong — property tests, chaos (`test_chaos.py`
`test_crash_window_chaos[0..7]`), BFS-oracle fuzz (`test_graph_fuzz.py`), a security matrix,
concurrent-serve fail-fast (`test_concurrent_serve.py:0.80s`). Costs: `server.py` at 63.5% (ARCH-13),
`doctor.py` 79.3%, `embeddings.py` 79.6%, `portability.py` 80.5% with 51 statements missed, and one
test at **60.35 s** (`test_post_publish.py::test_j10_soak_reports_psutil_status_honestly`) dominating
the run. `pytest_out.txt` is a committed transcript (271 passed / 4 skipped / 109.33 s) and is stale
by construction, not a live signal.

### CI and release-pipeline findings

Verified by reading both workflow files end to end and grepping for the environment variables the tests
require. These are the items where "CI is green" and "the gate is armed" are not the same statement.

| ID | Item | Evidence | Sev | Pri | Effort |
|---|---|---|---|---|---|
| CI-01 | **Both performance gates are permanently dead.** `test_perf_smoke.py:18` skips unless `OMNIDB_RUN_PERF == "1"`; `test_semantic.py:20` skips unless `OMNIDB_TEST_SEMANTIC == "1"`. Neither variable is set anywhere in `.github/workflows/` — grepping both files for `env:` returns **zero matches**. The throughput regression gate the roadmap relies on has never run in CI, so the "writes >= 300/s / recall p50 < 5 ms" plan from `Comprehensive Analysis` §6.4 is unenforced. `test_post_publish.py:257-267` then asserts only that the perf module *reports as skipped*, encoding the deadness as expected behaviour. | [V] | High | P1 | 1–2 h |
| CI-02 | **`ci.yml` has no `permissions:` block at all.** Every job inherits the repository default token — conventionally `contents: write` — across a 9-cell matrix running third-party `actions/checkout@v4`, `setup-python@v5` and `upload-artifact@v4` against pull-request code. `release.yml:33-38` scopes correctly to `id-token: write, contents: read`, so the gap is an oversight, not a policy. | [V] | Med | P1 | 10 m |
| CI-03 | **The action holding the PyPI credential is pinned to a floating branch.** `release.yml:55` uses `pypa/gh-action-pypi-publish@release/v1` — a mutable branch tag — in the one job that mints an OIDC upload token. Pin to a full commit SHA with a Dependabot/`renovate` rule; that is the standard control on a path that ends in an artifact repository. | [V] | Med | P1 | 30 m |
| CI-04 | **`mypy` is installed ad hoc and never sees the tests.** Absent from `[project.optional-dependencies].dev` (`pyproject.toml:30` lists only pytest, pytest-cov, ruff, hypothesis), so CI installs it out of band and unpinned — a new mypy release breaks the build with no version to bisect to. `tests/` is not type-checked at all. | [V] | Low | P2 | 1 h |
| CI-05 | **An ungated performance floor inside a functional test.** `test_remember_many.py:73-76` asserts >= 800 writes/s with no `OMNIDB_RUN_PERF` guard — a hard timing assertion on shared runners, the classic intermittent-red generator, and the mirror image of CI-01: a slow test that is always on next to a fast gate that is never on. | [V] | Med | P2 | 30 m |
| CI-06 | **The coverage floor is a command-line flag, not configuration.** `--cov-fail-under=80` lives only in the CI invocation while `pyproject.toml` deliberately omits `fail_under` (`coverage-report.md`'s threshold note). Measured today **86.8%** (2,433 stmts / 321 missed) with `server.py` at **63.5%** — the aggregate passes while the least-tested module is the one owning the trust boundary (ARCH-09/13). | [V][M] | Med | P2 | 1 h |
| CI-07 | **`hnsw-required` pins node identities, not behaviour.** Four hardcoded node ids instead of index invariants, so any legitimate change to the seeded vectors turns the one job guaranteeing hnswlib is exercised red for the wrong reason. Assert dimension, element count, and recall-vs-brute-force instead. | [V][A] | Low | P2 | 1 h |
| CI-08 | **Ambient `OMNIDB_*` leaks into every child process.** No workflow clears the environment and `config.py:128-173` reads `OMNIDB_*` unconditionally, so a runner-org-wide variable silently reconfigures the product under test — including, by the same mechanism, silently *enabling* CI-01's slow gates. Subprocess-spawning tests (`test_concurrent_serve.py`, `test_stdio_conformance.py`, `test_j10`) inherit it. Sanitize in a `conftest.py` autouse fixture. | [V][A] | Med | P2 | 1 h |
| CI-09 | **The ignore rules and the index disagree about what is evidence.** `.gitignore:28-31` declares `pytest_out.txt`, `soak_*.txt`, `bench/soak-results*.txt` and `bench/results.md` generated, yet the baseline commit tracks `bench/growth-results.txt`, all four `research-v*` result files and `research-v2-results.json.killed.bak`. Result: the *one* artifact `AUDIT.md:194` cites is ignored and mutable (DOC-09), while a file named `.killed.bak` is permanent history. Decide per file and make the ignore list match the decision. | [V][M] | Med | P2 | 1 h |

---

## 7. Technical debt

| ID | Item | Evidence | Sev | Pri | Effort |
|---|---|---|---|---|---|
| DEBT-07 | `supersede` / `link` unreachable outside Python (= ARCH-11; product-surface debt, not dead code — the plumbing and its tests are live) | [V] | High | P1 | 2–4 h |
| DEBT-05′ | Consolidation re-embeds its dedup window each run. Now **one pool of 500 per agent scope** (`consolidate.py:264-272`, `:146`) — improved since 09-11 — still no content cache | [V][A] | Low | P3 | 2–4 h |
| DEBT-08 | ~20 zero-call-site dead items catalogued by the module pass, incl. `Embedder.name/dim` base values, `_fit_episode_budget`'s never-passed `max_tokens`, `consolidate.py:198`'s no-op `[:500]` duplicating `_episodes(limit=500)`, `langgraph._value_of`'s unreachable fallback, `recall.py:104`'s no-op `max(k*4,k)`, `pyproject.toml:75`'s phantom exclusion | [A] | Low | P3 | 0.5 d |
| DEBT-09 | README `:168` "GIF coming soon" and `examples/demo.gif` are unshipped promises in user-facing docs; `:153-165` claims a SQLCipher swap "needs no code changes" when `db.py:87` and `server.py:179` both call `sqlite3.connect` directly (two edits, plus `_home()` derived from `PRAGMA database_list` at `store.py:1092` would break) | [A] | Low | P2 | 1 h |
| DEBT-10 | `Issues and Required Fixes (Sep 2026).md` DEBT-06 (CI omits 3.11) marked fixed by adding 3.11, yet classifiers also claim **3.13** and the matrix per 09-11 is 3.11/3.12/3.14 — an untested claim re-introduced one version over | [V-doc] | Low | P3 | 30 m |

| DEBT-11 | Two 4.7 GB vendored toolchains and 2.6 GB of model weights sit inside the project tree with no documented boundary: `texlive/` (9,112 files, 313 MB, TeX Live 2026), `tools/jdk-21.0.5+11/` (490 files, 331 MB, Temurin 21.0.5+11-LTS), `TinyTeX-1-windows-v2026.09.exe` (73.7 MB), `omnidb-server/.hf-cache/` (~2.6 GB, incl. a 1,112,206,140-byte `bge-reranker-base` blob), `bench/.cache3/` (2.8 GB). They are correctly untracked, but nothing in the repo states that they are not project code, so an exhaustive review — including this one — must re-derive that boundary by hand every time, and `bench/research_bench_v3.py:455` depends on the JDK being at exactly that path (QA-22). Move to a documented `third_party/` root or make the paths config. | [V][M] | Med | P2 | 2–4 h |
| DEBT-12 | The `paper/` tree is a second, ungoverned artifact pipeline with no gate: three generations of one document, a DOCX built from the stale one (DOC-18), a `.log` pointing at `d:/OMNIDB/texlive/…` for `IEEEtran.cls`, and 23 vendored npm packages (`docx@^9.7.1`) with a lockfile but no `package.json` scripts or CI. Either bring it under a build target with a regeneration check or move it out of the project root; the current state means a paper claim can be stale without any test noticing, which is the same failure DOC-09 documents for benchmarks. | [V][A] | Low | P3 | 0.5 d |

---

## 8. Closed since 2026-09-11 — do not re-open without new evidence

| Item | Status |
|---|---|
| **BUG-12** unbounded `k`/`k_fetch` | **Fixed [V].** `recall.py:12-15` `_MAX_RECALL_K = 1000` + `:57-58,60,66`, variable-limit-aware batching at `:93`. My first draft listed this open from the 09-11 doc; the code disagrees and the code wins. |
| **MT-05 / BUG-05** adapter `clear()` wipes the home | **Fixed [A][V].** `llamaindex.py:66-73` and `langgraph.py:47-52` are scoped to `agent_id` defaults `"llamaindex"` / `"langgraph"`, citing BUG-05; `agent_id=None` is a documented whole-home opt-out (`llamaindex.py:17-20`). Residual: the opt-out is the unauthenticated path MT-02 covers. |
| **BUG-07** migration atomicity | **Fixed, verified by reading [V].** `db.py:129-143` one transaction + postcondition + rollback, `AUDIT-I10` at `:135`, `_heal_torn_v2` at `:146-156`. Solid. |
| **DOC-03** the `soak-results.txt` evidence chain | **Closed, and my characterization was wrong — see §9.** `bench/soak.py:19` states the new intent (*"writes soak-results-v2.txt, leaving soak-results.txt historical evidence intact"*) and `:137-138` **refuses** to write the historical file: `parser.error("the historical soak-results.txt must not be overwritten")`. It is a deliberately frozen artifact, not an orphaned citation. Confirmed by timestamps: my run today wrote `soak-results-v2.txt` at 10:03 and left `soak-results.txt` at Sep 18. |
| ARCH-05 lock (half) | `serve`'s exit code propagates [V] `cli.py:129-131`. The `doctor` half never did — BUG-13. |
| DEBT-02 wheel packaging | Fixed [V] `pyproject.toml:46` + `:43-45` comment. |
| QA-06 version drift; BUG-04 `OMNIDB_INDEX`; SEC-01 (single-user form) provenance; SEC-04; QA-01, -02, -05; DEBT-01; PERF-01..05; ARCH-01..04; BUG-01, -02, -03, -06, -08, -09, -11 | Marked fixed across remediation passes 1–3 [V-doc]; independently corroborated for several by the module pass (`config.py:172` alias, `rerank.py:76`, `store.py:894-921`, `vector_index.py:289-296`). SEC-01 **re-opened** as SEC-05 for a different, more severe reason. |

---

## 9. Retractions

Recorded because an unrecorded error propagates.

1. **"There is no cap on trigger pattern length or count, and no runtime bound on matching."** False.
   `validation.py:14-15` caps at 32 patterns / 512 chars on creation *and* import;
   `recall.py:29-30` bounds the query at 4,096 chars (oversized queries skip matching entirely,
   `:141-143`) and evaluations at 2,048, with both loops breaking correctly (`:161,170`). My grep of
   `skills.py` alone missed the bounds because they live in two other modules. The real finding is
   narrower and still Critical: **input bounds do not bound backtracking** for a single pattern, and
   the store-wide `RLock` turns one wedged recall into total unavailability. See SEC-02.
2. **"BUG-12 unbounded `k`/`k_fetch` is still open."** False — fixed in code (`recall.py:15`) and I
   repeated a stale doc statement instead of checking. Moved to §8.
3. **"`bench/soak.py` never writes `soak-results.txt`, so a vault latency claim cites a file the
   tooling does not produce."** Mischaracterized. The write is **prohibited on purpose**
   (`soak.py:137-138`) so the historical evidence stays stable; the live artifact is
   `soak-results-v2.txt`. The frozen file is deliberately immutable, not a tooling gap. **But my
   framing here was too generous in the other direction** — I filed "a test asserts on it" as an
   efficiency question. It is a correctness question: the file is gitignored and untracked, so the
   assertion cannot hold in CI at all. See BUG-17, which supersedes this note's last sentence.
4. **Every CVSS score in §1, as first written.** The hand-derived arithmetic was wrong in three ways
   (`ISC(H)` taken as 0.50 instead of 0.56; the Scope-Unchanged `PR:L` coefficient 0.62 used in a
   Scope-Changed vector, where it is 0.66; `UI:N` taken as 1.0 instead of 0.85) and invented a
   `min(5.274, …)` term that is not in the specification. SEC-02's 8.1 was really 7.3, and SEC-05's 7.7
   was really 6.2 under the vector I wrote. Recomputed against the spec with the implementation
   validated on two published reference vectors — see §0. CV-01 is closed; CV-02 records what remains.
5. **"Exactly three SQL strings are built by interpolation package-wide."** False as stated — there
   are **30** interpolated-SQL call sites. The safety *conclusion* holds, but for a different and
   narrower reason than "only three strings are dynamic", and it needed the two column-name sites
   (`store.py:493`, `:587`) checked individually rather than swept up in a count. QA-07 rewritten;
   QA-21 added, because an untested invariant that depends on a caller never passing a field name is
   not a control.
6. **"The 60 s test is ~73% of the suite."** Unsupported figure, repeated from an earlier draft rather
   than measured. The defensible statement is a **54.9% floor** (60.35 s / 109.33 s) with a realistic
   55–65% band. Corrected in BUG-17 and §6.

**One disclosure that is not a retraction but belongs in the same register.** This session's
`pytest --cov` run overwrote `bench/soak-results-v2.txt` as an indirect side effect of `test_j10`,
destroying the artifact the Sep-18 remediation pass recorded its acceptance numbers against. Because
those filenames are gitignored, the loss is unrecoverable. DOC-09 states it there, in the first
person, because a catalog that reports evidence drift while hiding that it caused some of it is not
evidence.

---

## 10. Sequencing and effort

**Track A — unblocked correctness (≈7 dev-days, no owner decisions needed).**
1. `BUG-15` embedder lockout — a Critical that fires from a documented happy path.
2. `SEC-05` consolidation scope leak — 0.5 d, and it is the only cross-tenant leak reachable today
   **without** an authenticated boundary, so it is not gated on MT-02. Its regression test is the same
   test MT-06 asks for; do them together.
3. `BUG-17` `test_j10` — 2–3 h, and it must land **before** any other CI work: this test currently
   overwrites benchmark evidence as a side effect (DOC-09c) and will go red the first time CI runs.
4. `CI-01` + `CI-02` + `CI-03` — arm the perf gates, add `permissions:` to `ci.yml`, pin the publish
   action to a SHA. ~2.5 h, and `CI-01` is what makes `PERF-*` and Track C measurable at all.
5. `SEC-02` screening/RE2, `SEC-07` zip budget, `BUG-13` doctor exit, `BUG-16c` doctor false-clean.
6. `LEGAL-01` LICENSE (blocks publication outright), `QA-22` path scrubbing, `DOC-16` market-claim
   corrections, `QA-12` enum validation.
7. `ARCH-08` logging, **because every tenant-facing fix after it needs attribution to be debuggable.**

**Track B — blocked on the owner decisions in `MULTI_TENANCY_THREAT_MODEL.md` §9 (≈4–5 dev-weeks).**
`MT-06` (the four xfail tests) can start **now** and should, since it measures the size of this track
before it is committed to. Then `MT-02` → `MT-03` (OAuth/resource-server) → `MT-04`
(multi-writer/replica story, which `DATASTORE_SPIKE.md` is still inconclusive on) → `SEC-01`
consolidation partitioning as a policy, `OPS-01` encryption and retention, `MT-05`-class
destructive-path audit re-scoped to *authenticated* identity. Do not start MT-02's schema work before
§9 is answered; guessing here is the expensive rework.

**Track C — evidence integrity (≈4 dev-days, new this pass, and arguably the highest-value week).**
`DOC-09` re-measure the soak and growth gates on the final runtime, bind the numbers to
`source_identity_sha256` in prose, commit or stop citing the artifacts (pairs with `CI-09`);
`DOC-10` re-render the v3 CE grid from its JSON; `DOC-11` wire `mutation-report.md`'s integrity hashes
into a check instead of trusting them; `DOC-12`/`DOC-13`/`DOC-14` reconcile counts, baselines and the
scifact/scidocs attribution; `DOC-15` fix the effort arithmetic; `DOC-18` regenerate or delete the
paper's DOCX. **Rationale:** Tracks A and B both change measured behaviour, and every current baseline
for both is in documents that demonstrably do not match their own artifacts. Fixing the evidence layer
first is what makes the other two verifiable rather than merely plausible.

**Track D — quality and scale (≈2 dev-weeks).** `PERF-09` → `PERF-10` → `PERF-11` in that order (mmap
substrate first; it is the prerequisite for the f32 rerank window), `ARCH-07` split before Track B's
predicates land, `ARCH-12` gate repairs, `CI-04`…`CI-08`, `QA-21`, `DEBT-07`, `DEBT-11`.

Total: **~7 days** unblocked correctness, **~4 days** evidence integrity, **~4–5 weeks** contingent on
the tenancy decision, **~2 weeks** of scale work valuable under either target.

### Coverage attestation

The earlier draft of this section listed three reader passes as outstanding. All five returned, and
each one corrected the draft — three of them corrected *me*, which is the point of running them. What
was actually read:

| Scope | Coverage |
|---|---|
| `omnidb_server/` — all 22 modules | Every line, both by me and by a dedicated reader pass |
| `tests/` (39 files), `.github/workflows/` (2), `pyproject.toml`, `MANIFEST.in`, `README`, `ARCHITECTURE`, `CHANGELOG` | Every line; findings at BUG-17/18, CI-01…09 |
| `docs/` (9 authored + this catalog), `bench/` (15 scripts + 20 result artifacts), `scripts/`, `examples/` | Every authored file in full; generated JSON inspected structurally and numerically |
| `OmniDB Vault/` — 36 files | 16 read by me, 20 by a reader pass; `Architecture.canvas` and `.obsidian/` config included |
| `omni.txt`, `OmniDB_v1_Design_Document.pdf` (73 pp), `paper/` (28 authored), `.remember/` (113), `_verify/` (9), `_verify-nope/` (3), root strays | Every file; credential and path sweeps run over all of them |
| **Declared exclusions** | `texlive/` (9,112 files), `tools/jdk-21.0.5+11/` (490 files), `paper/node_modules/` (386), `.hf-cache/` and `bench/.cache3/` (~5.4 GB of model blobs), `build/`, `dist/`, `*.egg-info`, `__pycache__`, `.mypy_cache`. Vendored toolchains and build output, not project code — listed, not read. Say the word and I will sweep any of them. |

**Two things this attestation does not claim.** First, the PDF was read as *text* (via `omni.txt` and
independently via PyMuPDF) with its per-page figure/table/vector inventory extracted, but never
rendered as pixels — `pdftoppm` is not installed here, so Figures 1–7 have positional and caption
evidence, not visual review. Second, everything cross-referenced above was verified against the tree
**as it stands at `d92c4a1` plus this uncommitted document**; the three `lastfailed` entries in
`.pytest_cache` are from a run that pre-dates some of the fixes, and re-running the suite is the only
way to know whether the current tree is green. The suite was last run at 86.8% coverage, exit 0.

---

## Related

- `MULTI_TENANCY_THREAT_MODEL.md` — bypass inventory; §9 owner decisions gating Track B
- `DATASTORE_SPIKE.md` — MT-04's experiments and the negative result so far
- [[Final Build Verdict]], [[Architecture Decision Record]], [[Issues and Required Fixes (Sep 2026)]]
- [[Enhancements and Modifications (Sep 2026)]] (E-16 → SEC-02), [[Risk Register]], [[Implementation Status]]
