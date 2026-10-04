# 4 - New Additions

Source: **D1** (modules) - **D2** (integrations) - **D3** (tooling) - **D4** (packaging) - **D5** (schema).
D1 LOC verdicts are judged against the vault 1,200-line ceiling, already breached 3.5x.

---

## S1 - New modules (agent D1)

| Module | LOC | Why separate | Worth it |
|---|---|---|---|
| errors.py | 140 | The OMN-* table is a **wire contract**; 5 classes share no ancestor; the code is baked into str(exc) forcing bracket-parsing | **Yes - foundation** |
| predicates.py | 70 | SQL vocabulary + the **one** batching helper (3 divergent reserves today; forget_many/purge have none) | **Yes - smallest LOC-per-defect-fix** |
| ranking.py | 90 | RetrievalPlan + RetrievalContext breaks the store <-> recall cycle | **Yes - mechanical** |
| index_keeper.py | 180 | Owns the 5 markers + the ARCH-02 post-commit ordering in one place - **the prerequisite for splitting store.py at all** | **Yes** |
| querylog.py | 160 | Owns a table holding **verbatim tenant text** with no redaction/TTL/export control | **Yes - P1** |
| obs.py | 200 | Zero logging exists; prerequisite for all tenancy work | **Yes - P1 prereq** |
| introspect.py | 150 | stats/schema/check are already pure - cheapest extraction, testable without a store | Yes |
| metrics.py | 120 | Counters must be registry-shaped to be scrapeable | Yes, **counters only** |
| migrate_harness.py | 170 | Migration bugs are **silent data loss** | Yes - highest risk-reduction/LOC |

**Net: ~+1,280 new, ~-700 out of store.py.** All modules stay far under 1,200 lines.

### The interfaces that matter

    # errors.py - one base, one table
    class OmniDBError(RuntimeError):
        def __init__(self, msg, code=E_INTERNAL, **ctx): ...
    CODES: dict[str, str          # contiguous OMN-001..OMN-021
    def code_of(exc) -> str        # for legacy catch sites

    # index_keeper.py - ARCH-02 lives in exactly one function body
    def recover(conn, index, embedder) -> RecoveryReport
    def after_commit(mark_dirty, apply_index, n_ops, compaction=False) -> None

    # obs.py - redaction rules, allowlist not denylist
    never log: content, raw query (log sha256[:12] + length instead)
    always log: agent_id, omni_code; truncate any string >256; mask sk-*/Bearer *
    drop: absolute home paths -> basename

    # querylog.py - redact-then-insert, row cap AND age purge
    class QueryLog(conn, cfg):
        def record(kind, query, result_count, took_ms, agent_id) -> None

### Do NOT build
A config-reference **generator** (a side table is enough; docs drift is loud and cheap) - a
generic plugin/registry framework (YAGNI, two implementations exist) - tenancy.py before the
privacy defect is fixed - **any further store.py split before index_keeper.py exists** (that
forks the ARCH-02 crash window) - gauges beyond counters.

---

## S2 - Schema evolution (agent D5)

### NOW

| Change | Unlocks | Migration LOC |
|---|---|---|
| **tags(memory_id, key, value)** - one table fixes **3** call sites: skill lookup, LangGraph keys, list_skills - all LIKE-on-JSON **full scans** | index seeks instead of scans | ~45, **pure SQL** |
| fact_sources(fact_id, episode_id, weight) | reverse provenance: which facts derive from episode 42? Lands **only** on the opt-in consolidate path | ~20 |
| schema_migrations(version, name, checksum, applied_ms) | auditable trail; makes _migrate postconditions generic | ~12 |

The tags migration is pure SQL and forward-only:

    INSERT INTO tags SELECT id,\'key\',json_extract(metadata,\'$.key\') FROM memories
      WHERE json_valid(metadata) AND json_type(metadata,\'$.key\')=\'text\';

**Requires a JSON1 probe in connect()** next to the existing FTS5 probe (db.py:97-100) or the
backfill fails mid-migration. WITHOUT ROWID + that PK gives the seek.

### LATER
- **tenant_id** - FTS5 has no partition key, so isolation degrades to exactly today post-filter
  join; true FTS isolation needs one FTS5 table per tenant (runtime DDL the forward-only ledger
  cannot gap-check). **The cheaper answer is one home per tenant: 0 LOC, and it closes all 4
  bypasses at once** because the home lock already guarantees the dense leg, the FTS join,
  related(), stats() and the sidecar are per-tenant by construction. Ship tenancy as deployment
  topology; revisit the column only if single-process multi-tenant ever becomes a requirement.
- **valid_from_ms / valid_to_ms** - the orthogonal valid-time axis. Must **never** be wired into
  _SELECTIVE, or every live-set query doubles its logic for a consumer that does not exist yet.

### NEVER
- **A facts/claims table** - duplicates kind=semantic + metadata.origin, and forces changes in
  store/recall/consolidate. This is precisely the re-inflation risk the vault named.
- **Edge weight/confidence/source** - edges are only written by remember caller-supplied links
  and import; consolidation emits none. Weighting user-declared edges carries ~zero signal.
- **FTS over more fields** - memories_au is `AFTER UPDATE OF content`, so it would silently
  NOT fire on a metadata or agent_id change, desyncing the index with no error. Plus a full
  reindex and 3x write tokenization.
- **Per-tenant FTS / HNSW tables** - runtime DDL the forward-only ledger cannot represent.
- jsonb columns, a stored score/rank column, FKs on meta, a separate embeddings table
  (vectors are deliberately derivable).

---

## S3 - Integrations (agent D2)

**Zero framework imports preserved throughout.** Every proposal adds **no** framework import
and no runtime dep beyond anyio, already a hard transitive dep of mcp and imported at
server.py:10.

| ID | Integration | Fit | Effort |
|---|---|---|---|
| I-1 | list_namespaces() on OmniDBStore | **High** - LangGraph requires it | 2-3 h |
| I-2 | batch() | **High** - a checkpoint write | 3 h |
| I-3 | aput/aget/asearch/adelete/abatch | **High** | 6-8 h |
| I-5 | LlamaIndex async | High, ~1 h once I-3 lands | 1 h |
| I-10 | MCP resources: omnidb://schema, //config, //stats | **High** | 4-6 h |
| I-12 | **MCP sampling for consolidation** | **High** | 4-5 h |
| I-11 | omnidb://memory/{id} | **BLOCKED** - inherits the tenancy problem | - |
| I-13 | Streamable HTTP transport | **Not now** - home lock makes it single-client | 5-8 d |
| I-4 | A separate AsyncOmniDBStore class | **Rejected** | 0 |
| I-8 | Semantic Kernel MemoryStore | **Rejected** - needs collections OmniDB has no primitive for | 2-3 d |
| I-9 | CrewAI | **Rejected** - its default stack is hosted OpenAI + LanceDB, violating zero-telemetry | 0 |
| I-16 | Prometheus push | **Rejected** - pull is a listener, push is phone-home | 0 |

### I-3 - async on the SAME class, not a new one
OmniDBStore already imports anyio; aput = await to_thread.run_sync(partial(self.put, ...))
reuses the server.py:15-16 _call pattern verbatim. **A separate class duplicates
_find_row / _value_of / _require_*** and is the re-inflation DOC-01 warns about.

**A real signature break to fix in the same PR:** the real BaseStore.search is
(namespace_prefix, filter=None, limit=10, offset=0); OmniDB is (prefix, query, limit=5).
**LangGraph passes a filter dict positionally**, so the current adapter raises E_TYPE. Also
Item requires created_at/updated_at and search items omit id - supply from created_ms.

### I-12 - the one that structurally removes SEC-06
MCP sampling is server-initiated createMessage where **the client selects the model and already
holds it**. The episode transcript therefore crosses **no new boundary** - whereas SEC-06 POSTs
it plus a bearer token to an operator-typed URL. Blocker: run_consolidation has **no LLM
injection seam** (_llm_extract_facts is a module global). Add
extract_facts: Callable | None = None (~1 h) plus a sampling-backed callable (~3 h), falling
back to the offline stub when the capability is absent. The vault deferred sampling to dodge
churn, but that was a *scoping* call - sampling has been in the spec since 2024-11-05.

### Never propose again
LangSmith / LangGraph Platform - any hosted vector DB (Pinecone, Qdrant Cloud, Weaviate Cloud) -
OTLP / Honeycomb / Datadog / Langfuse Cloud / Phoenix Cloud - a Prometheus pull endpoint - a
Postgres/Lucene backend (violates ADR-1/ADR-2) - any hosted MCP gateway or registry - CrewAI
default OpenAI stack - usage or telemetry beacons of any kind.

---

## §4 · Tooling (agent D3)

| ID | Tool | Solves | Effort | Worth it |
|---|---|---|---|---|
| D3-1 | **Generated config reference** | 28 fields; README names ~10; the `init` template documents 4; **`corpus_mode`** — which zeroes both rerank weights for a measured 40-56% nDCG effect — appears in **none** of them | 1.5-2 d | **YES, highest ROI** |
| D3-5 | **`restore --dry-run` + verify** | DR is 5 manual prose steps. **`PRAGMA integrity_check` and `foreign_key_check` appear NOWHERE in the package** (0 matches) | 2-3 d | **YES** |
| D3-4 | `doctor` remediation + severity + `--brief` | 12 checks return bare `{ok, detail}` with no what-to-run. Worse, `_optional_dep` returns `ok:True` for a **missing** module | 1-1.5 d | **YES** |
| D3-2 | Shell completions (bash/zsh) | 22 leaf commands, zero completions | 0.5 d | **YES** |
| D3-8a | `devcheck` one-command pre-push | It is the harness every other check here rides on | 0.5 d | **pick** |
| D3-8b | `omnidb-server tools --json` manifest | The 10-tool list is asserted in ~5 places that disagree; the vault Implementation Status is already stale at "7 tools" | 0.5 d | **pick** |
| D3-3 | `config --explain` (layer/shadow trace) | CWD `omnidb.toml` is read first then silently overwritten by the home copy — **shadowing is invisible today** | 1 d | **PARTIAL** |
| D3-7 | `docs/DECISIONS.md` for the code tags | `AUDIT-*`/`PERF-*`/`ARCH-*`/`SEC-*`/`BUG-*` pepper the code and are indexed nowhere | 4-6 h | **PARTIAL — author the index, skip the generator** |
| D3-6 | Standalone `backup --verify` | Same code path as `restore --dry-run` | — | **FOLD into D3-5** |

### The insight that makes D3-1 cheap
**The config doc metadata already exists in-repo as 8 docstring comment blocks** — `corpus_mode`
(`config.py:79`), PERF-01:92, PERF-07:100, PERF-08:103, SEC-04:107, PERF-05:111, SEC-01:118,
E-12:123. Hoisting them into generated docs is mechanical, not authoring. **The comments prove
the maintainer already wants this.**

### D3-5 — the real payload is not the restore command
It is that **`integrity_check` + `foreign_key_check` become reachable for the first time in a
shipped path.** A dangling `superseded_by` (the FK has no cascade) currently hides a row from
every live-set query forever, and no tool in the box reports it.

### Skip these
Pre-commit config (hooks get bypassed on Windows and add latency to the loop that must stay
fast) · release automation checks (at v0.1.3 with no cadence the bottleneck is writing changelog
entries, not verifying them) · a fail-on-unindexed-tag CI gate (taxes the person mid-bugfix;
disabled within two commits) · PowerShell completion as a first-class target
(`Register-ArgumentCompleter` needs manual `$PROFILE` editing — emit the snippet, do not claim
parity).

---

## §5 · Packaging (agent D4)

### PKG-01 LICENSE — the only true blocker, 15 minutes gating a package already on PyPI
`pyproject.toml:10` declares Apache-2.0, `MANIFEST.in:3` includes LICENSE, and **no LICENSE file
exists** (verified by me: a glob over `d:/OMNIDB` returns 73 hits, all under `tools/jdk legal/`
and `texlive/`, **zero** in `omnidb-server/`).

**Consequence of shipping unlicensed:** All rights reserved by default — nobody may legally copy,
modify or redistribute the code, including in a corporate dependency review. Most org OSS
scanners (FOSSA, Black Duck, GitHub dependency graph) fail the build or require a legal
exception. **An MCP server a vendor must vendor is a hard adoption wall.**

1. Add the full Apache-2.0 text as LICENSE (and a NOTICE — cheap and expected).
2. `license = "Apache-2.0"` (PEP 639 SPDX string) + `license-files`. The table form is deprecated.
3. Add a `release.yml` assertion that the shipped sdist actually contains LICENSE.

**Caveat:** Apache-2.0 carries a patent grant and NOTICE-propagation clause. If third-party
benchmark corpora (NFCorpus, SciDocs, SciFact qrels) are ever vendored into `bench/`, their
terms may be incompatible. **A 0.1.4 sdist cannot retroactively fix 0.1.3.**

### The rest
| ID | Addition | Blocks adoption? | Effort |
|---|---|---|---|
| PKG-02 | Single version source (`importlib.metadata`) + CI drift gate | No (trust) | 45 m |
| PKG-03 | Drop the untested 3.13 classifier (ci.yml tests 3.11/3.12/3.14; pyproject claims 3.13) | No (honesty) | 10 m |
| PKG-04 | Wheel-contents manifest test (exact file list + <1 MB) | No (regression) | 1 h |
| PKG-05 | `py.typed` honesty: keep the marker, add one README line that annotations are best-effort | No (trust) | 20 m |
| PKG-07 | Offline model provisioning + a `doctor` model check | Partial (air-gapped) | 3-4 h |
| PKG-08 | Pin the build backend (`setuptools>=68,<81`) | No | 1 h |
| PKG-09 | CHANGELOG shape repair (three `### Fixed` under [0.1.3]; `[Unreleased]` below the release) | No (trust) | 30 m |
| PKG-11 | **`omnidb-bench` as a real distribution** — fixes 9 hardcoded paths incl. a JDK at `D:/OMNIDB/tools/jdk-21.0.5+11` | No, but it is **the moat** | 2-3 d |
| PKG-12 | Devcontainer (replaces the 331 MB vendored JDK, DEBT-11) | No (credibility) | 3 h |
| PKG-14 | Homebrew / apt / Debian | No | **not worth it** |

### PKG-06 — hnswlib should NOT become a hard dependency
It has manylinux + Windows wheels, but they **lag new CPython** (which is why 3.14 is marginal),
and `ci.yml:45` already treats the native backend as a *conditional* gate, not a baseline. Making
it hard converts "pip install works everywhere" into "pip install fails on the newest Python on
the newest release." Keep the extra; document the brute fallback (`doctor.py:168` reports it).

### Not worth it yet
Homebrew/apt (a formula for a 0.1.3 package with ~0 installs; Homebrew Core review requires a
stable cadence you do not have — revisit at v1.0) · a full sdist lockfile (disproportionate at
4,195 LOC; pin the backend instead) · publishing model weights as a distribution (legal surface,
no user value, and PKG-01 blocks redistribution anyway) · a `Framework::MCP::Server` classifier
(`pyproject.toml:24-26` records that this already broke a build once) · a containerised **server**
image (the product is local-first stdio; there is no daemon to containerise).

---

## §6 · Sequencing — the packaging and schema bundles

These cannot be split. Each bundle lands as one commit or not at all.

| Bundle | Items | Why together |
|---|---|---|
| **P1** | `LICENSE` + `NOTICE` + PEP 639 `license-files` + the `release.yml` sdist assertion | The declaration, the file, the metadata, and the check that it shipped. Shipping the file without the assertion is how it went missing. |
| **P2** | `tags` table + `schema_migrations` + the `SCHEMA_VERSION` bump + the JSON1 probe in `connect()` | The backfill fails mid-migration without the JSON1 probe; 6 test modules pin the version. **One commit.** |
| **P3** | `omnidb-bench` distribution + the 9 hardcoded-path fixes | The bench *is* the moat (C4). A moat that only runs on the author's machine is not one. |
| **P4** | `config --explain` + the generated config reference | Both read the same layer/shadow trace. Shipping the docs without the explain flag documents behaviour an operator cannot inspect. |
| **P5** | Devcontainer + the DEBT-11 boundary doc | Replaces the 331 MB vendored JDK *and* records where the boundary is, so the next clone does not re-derive it. |

### The order that matters
**P1 is a 15-minute prerequisite for everything else in packaging.** Until Apache-2.0 is
actually granted, P3 cannot ship a redistributable benchmark and P5 cannot claim a clean
dependency story. Everything else is freely reorderable.

### The one thing to not build
**`tenancy.py`.** D5's LATER analysis concluded the cheaper answer to multi-tenancy is
**one home per tenant** — 0 LOC, and it closes all four MT-02 bypasses at once, because the
home lock already guarantees the dense leg, the FTS join, `related()`, `stats()` and the sidecar
are per-tenant by construction. A `tenancy.py` module is the *expensive* answer to a question
that a deployment decision answers for free. **Ship tenancy as deployment topology; revisit
the column only if single-process multi-tenant ever becomes a real requirement.**
