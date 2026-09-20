# 2. Issues & Required Fixes — OmniDB

**Assessment date:** 2026-09-18  
**Scope:** `d:/OMNIDB/omnidb-server/`, primarily runtime modules, tests, CI, and recorded benchmarks. Bundled toolchains, the vault, and paper-generation tooling are outside this assessment.  
**Method:** Eight completed read-only subagent reviews across the conversation: five initial domain reviews, one test-evidence follow-up, and two final verification reviews. Findings below were cross-checked against source or repository records. No application tests, new benchmarks, exploit reproductions, or crash simulations were executed. No application code was changed.

**Evidence labels:** **Static** means supported by implementation inspection, not executed; **Recorded** means an existing benchmark artifact, not a fresh measurement. All regression procedures below are proposed, use disposable synthetic fixtures, and have not been run. Security validation is deliberately bounded and omits exploit payloads.

Locations below are relative to this project root. Bare runtime filenames refer to `omnidb_server/`. Estimates include implementation and focused regression tests by an engineer familiar with the code; they are planning estimates, not measured delivery times.

## Executive assessment

The most urgent risks are incorrect vector recovery, ineffective cross-process ownership, inconsistent backup/migration publication, and restore-contract violations. Two additional High security findings concern resource bounds on the local MCP channel. No Critical vulnerability, working SQL write bypass, or dependency CVE was established.

The local stdio threat model matters: `agent_id` is a retrieval label, not an authenticated tenant boundary. Security ratings assume an agent/client already able to call tools. Numerical CVSS scores are not assigned without a validated deployment-specific attack vector; qualitative severity and CWE classifications are provided instead.

### Classification

- **Critical:** demonstrated catastrophic business impact or broad compromise; none established here.
- **High:** substantial correctness, recovery, or session-availability risk.
- **Medium:** constrained correctness failures, scale limitations, or release-assurance gaps.
- **Low:** limited-exposure safeguards or localized efficiency/policy debt.
- **P1:** recommended release blocker; contain immediately and prioritize implementation.
- **P2:** next remediation sprint; some narrow correctness fixes can be pulled forward.
- **P3:** scheduled debt; do not confuse lower priority with absence of risk.

Target windows below are relative to approval of the remediation effort. They are prioritization targets, not a promise that two engineers can complete all work concurrently.

## Ranked remediation register

| Rank / ID | Finding | Severity | Priority | Effort | Target window | Dependencies |
|---|---|---|---|---|---|---|
| 1 / I02 | Ownership acquired after writable initialization; CLI bypass | High | P1 | 12–20 h | 7 days | None |
| 2 / I01 | Same-ID content updates escape vector recovery | High | P1 | 12–20 h | 7 days | Shared recovery design; no hard prerequisite |
| 3 / I04 | Import changes IDs and lacks failure atomicity | High | P1 | 12–20 h | 7 days | I01 for post-commit recovery |
| 4 / I05 | SQL work budget undercounts approximately 10,000× | High | P1 | 6–12 h | 7 days | None |
| 5 / I06 | Stored trigger matching lacks resource bounds | High | P1 | 12–24 h | 7 days | Trigger-language compatibility decision |
| 6 / I03 | Backup and re-embedding publish mixed generations | High | P1 | 20–32 h | 14 days | I01, I02 |
| 7 / I07 | Mixed explicit/automatic batch IDs misassociated | Medium | P2 | 3–5 h | 14 days | None |
| 8 / I08 | Recall filters differ across retrieval branches | Medium | P2 | 3–5 h | 14 days | Filter contract |
| 9 / I09 | Candidate window truncation precedes filtering | Medium | P2 | 4–8 h | 14 days | I08 |
| 10 / I10 | Torn-v2 repair can omit a required column | Medium | P2 | 2–4 h | 14 days | None |
| 11 / I11 | Consolidation loses provenance and scope | Medium | P2 | 8–14 h | 14 days | Scope/provenance policy |
| 12 / I13 | Recovery SQL and embedding batches are unbounded | Medium | P2 | 4–6 h | 14 days | Coordinate with I01 |
| 13 / I14 | CI and assertions fail to enforce key contracts | Medium | P2 | 12–20 h | Start now; 14 days | Regression cases from I01–I04 |
| 14 / I12 | Saved soak result fails gate; workload is not steady-state | Medium | P2 | 12–24 h | 14 days | Correct methodology before optimization |
| 15 / I20 | Lifecycle cleanup is not exception-safe | Medium | P3 | 4–8 h | 30 days; bundle with I02 | I02 |
| 16 / I15 | Config coercion silently accepts malformed values | Medium | P3 | 3–5 h | 30 days | Configuration compatibility policy |
| 17 / I16 | Idle sidecar persistence deadline is not enforced | Medium | P3 | 3–5 h | 30 days | Lifecycle/maintenance ownership |
| 18 / I17 | Newer schema versions are accepted | Low | P3 | 1–2 h | 30 days | None |
| 19 / I18 | Fixed-operation snapshot policy overrides adaptive cost control | Low | P3 | 3–6 h | 30 days | I12, recovery-work budget |
| 20 / I19 | Batch forgetting repeatedly constructs the same set | Low | P3 | 0.5–1 h | 30 days | None |
| 21 / I21 | Update and import omit creation-time content limits | Medium | P2 | 4–8 h | 14 days | Shared validation; coordinate I01/I04 |
| 22 / I22 | Documentation contradicts implemented contracts | Low | P3 | 3–5 h | 30 days | Final decisions from I01–I04, I16/I18 |

**Sizing:** The first 20 entries sum to **147.5–266 hours**; the full register sums to **154.5–279 hours**, approximately **19–35 eight-hour engineer-days** before contingency. These are gross sums: I02/I20 and I01/I03/I13 share implementation work, while compatibility decisions can increase effort. Two engineers have a theoretical floor of roughly 10–18 working days before coordination and contingency; plan approximately 3–5 calendar weeks, not a guaranteed three-week delivery. Narrow containment and selected P1 fixes can land earlier.

## Detailed catalog

### I01 — Content updates can leave permanently stale vectors

**Category:** Software bug / durability. **Evidence:** Static. **Location:** `store.py:199–229,479–521`.

Content is committed before encoding and before marking the index dirty. Recovery compares live and indexed IDs, so an unchanged ID with changed content is invisible to its missing/stale set comparison. Business impact: dense retrieval can disagree with SQLite and FTS even after reopening.

**Proposed regression:** In a disposable store, remember and flush one memory; update it using an encoder double that raises; reopen and compare stored content with the vector representation. Also simulate interruption after dirty marking but before vector persistence.

**Expected:** Failure leaves content unchanged, or recovery restores vector/content agreement. **Source-indicated actual:** New content can coexist indefinitely with the old vector.

**Remediation/acceptance:** Encode before SQL mutation; atomically record a content revision or pending index operation with the row update. Reconcile revisions as well as membership. Every tested interruption boundary must recover the committed revision. Co-design the manifest interface with I03, but do not wait for full backup redesign to fix update handling.

### I02 — Ownership is acquired after mutations and bypassed by CLI

**Category:** Architecture / concurrency bug. **Evidence:** Static. **Location:** `server.py:104–123,177–190`; `cli.py:183–254`; `store.py:987–991`.

The server builds a writable stack before acquiring the home lock. A rejected second server calls `close()`, which flushes its independently loaded index. Ordinary CLI commands open and close another writable store without acquiring that lock. Business impact: a routine diagnostic or conflicting opener can overwrite another process's sidecar state.

**Proposed regression:** Use recording lifecycle doubles to assert lock acquisition precedes any database/index mutation; reject a second opener and assert it performs no persistence. Add an isolated two-process smoke case after that unit test.

**Expected:** Conflicting access fails before mutable initialization or cleanup writes. **Source-indicated actual:** Initialization and rejection cleanup can write without ownership.

**Remediation/acceptance:** Centralize ownership around stack creation and all maintenance entry points. Define read-only diagnostics separately rather than treating a writable store as read-only because the selected CLI command looks observational. Bundle I20 exception-safe cleanup.

### I03 — Backup and re-embedding lack consistent generation publication

**Category:** Architecture / disaster recovery. **Evidence:** Static. **Location:** `portability.py:43–59,177–218`; `db.py:145–151`.

Backup locks the database snapshot but copies sidecars afterward, outside that lock. Re-embedding publishes a new index before updating three identity fields through separate commits. Same-dimension model changes cannot be detected by dimension checks alone. Business impact: a backup or interrupted migration can reopen with semantically incompatible vectors or fail to open.

**Proposed regression:** With a tiny fixture and synchronization hooks, change existing content between DB snapshot and sidecar copy. Separately interrupt publication between index persistence and identity commits, including a same-dimension encoder change.

**Expected:** An internally consistent generation opens, or deterministic recovery reconstructs it. **Source-indicated actual:** DB, vectors, and embedder identity can belong to different generations.

**Remediation/acceptance:** Stage immutable index generations; atomically select a generation and commit identity together. Keep the previous generation until publication succeeds. A simpler backup option is a DB-only snapshot explicitly requiring index reconstruction. Test every publication boundary; do not promise identical ANN ordering after a rebuild without deterministic configuration.

### I04 — JSONL import changes IDs and is not failure-atomic

**Category:** Software bug / restore integrity. **Evidence:** Static. **Location:** `portability.py:346–351,410–434,454–517`.

Ordinary exports enter a bulk path that omits original IDs from insertion. Later edge conversion occurs after inserts without a transaction rollback guard, and unresolved endpoints are silently skipped while the report counts input edges. Business impact: external references break and failed restores can leave partial state.

**Proposed regression:** Export surviving IDs 1 and 4 and restore to an empty home. Separately supply two tiny valid memories with one invalid edge endpoint type; inspect transaction state, then close/reopen and check no rows survived rejection.

**Expected:** Original IDs and valid edges survive; rejection leaves the home unchanged. **Source-indicated actual:** IDs are reassigned sequentially and some failures occur after writes; later commits can persist those writes.

**Remediation/acceptance:** Preserve IDs explicitly, validate every field and endpoint before insertion, use a rollback-protected transaction, and report actual applied counts. Keep index recovery coordinated with I01 and content validation with I21.

### I05 — SQL budget counts callbacks instead of instructions

**Category:** Security / CWE-400. **Evidence:** Static. **Location:** `store.py:65–79,153,679–705`.

The budget is 5,000,000 while each increment represents a callback after 10,000 VM instructions. The effective default permits approximately 50 billion instructions rather than five million. The store lock remains held, and a row limit does not constrain result bytes. Business impact: a client-controlled read can impair the entire local session.

**Safe validation:** Use a recording callback/connection double and a tiny bounded query fixture to verify accounting and cleanup, not an expensive exhaustion query.

**Expected:** Execution is bounded near the declared work limit, with bounded response size. **Source-indicated actual:** The instruction budget is approximately 10,000 times larger, and bytes/time lack equivalent application bounds.

**Remediation/acceptance:** Account for callback interval explicitly; add a monotonic deadline and SQLite/application input and result-size limits. Ensure authorizer/progress cleanup on all exits. Deadlines checked by a VM callback alone do not guarantee interruption inside a long-running scalar function; document this and isolate execution if hard wall-clock guarantees are required.

### I06 — Procedural trigger matching has no complexity bound

**Category:** Security / CWE-1333. **Evidence:** Static. **Location:** `store.py:375–379`; `recall.py:23–35,123–142`.

Stored patterns execute through Python's backtracking regex engine during serialized recall. A bounded compilation cache does not bound matching work. Business impact: a problematic persisted skill can repeatedly delay later ordinary recalls.

**Safe validation:** Test admission limits and cancellation contracts with a controlled matcher double. Do not execute pathological regexes against live stores.

**Expected:** Pattern count, input size, and execution work have enforceable bounds. **Source-indicated actual:** These limits are absent on this path.

**Remediation/acceptance:** Prefer a restricted trigger language or a compatible linear-time engine; bound pattern count/length and query size. Assess existing pattern compatibility before rollout. Apply policy consistently to creation and import. Do not rely on abandoning a worker thread to stop regex execution.

### I07 — Mixed explicit and automatic batch IDs are misassociated

**Category:** Software bug. **Evidence:** Static. **Location:** `store.py:433–475`.

The highest-ID readback heuristic assumes automatic IDs occupy the tail. A batch with an automatic row followed by explicit ID 100 produces SQLite IDs 1 and 100, but the heuristic reports 100 for both. Exposure is limited to programmatic/importer use of explicit IDs. Business impact: wrong returned references, vector associations, or link sources.

**Proposed regression:** Insert those two synthetic records using a recording index double and compare returned IDs to stored rows and vector labels.

**Expected:** Two correct, distinct IDs associated with their content. **Source-indicated actual:** Duplicate ID association.

**Remediation/acceptance:** Capture each insertion's actual ID inside the shared transaction; retain batched encoding and a single commit. Test explicit IDs before, after, and between automatic rows.

### I08 — Recall filters differ by retrieval branch

**Category:** Software bug. **Evidence:** Static. **Location:** `recall.py:55–85,121–172`.

The trigger branch always selects procedural rows, regardless of requested kind. Empty `agent_id` is treated as unscoped by truthiness in some branches but as a supplied value by sparse SQL. Business impact: unexpected result types and inconsistent scoped retrieval, not a proven authentication bypass.

**Proposed regression:** Seed three memory kinds and a harmless literal trigger. Request semantic-only results; then verify each branch's scope behavior for `None`, empty, and nonempty agent IDs.

**Expected:** Every result satisfies one documented filter contract. **Source-indicated actual:** Procedural triggers can bypass kind filtering, and empty scope semantics differ.

**Remediation/acceptance:** Centralize eligibility; gate triggers by kind and use explicit `is not None` checks, or reject empty agent IDs consistently. Preserve documented global retrieval when no scope is supplied.

### I09 — Fusion truncates before eligibility filtering

**Category:** Software bug / retrieval quality. **Evidence:** Static. **Location:** `recall.py:44–86`.

Only the first four times `k` fused IDs reach SQL eligibility filtering. An eligible candidate already retrieved below that window is lost even if more fetching was requested. Business impact: avoidable empty or short result sets.

**Proposed regression:** Mock five dense candidates: four wrong-kind then one right-kind; no sparse hits or triggers; request one result.

**Expected:** The already-fetched eligible fifth row is returned. **Source-indicated actual:** The first four are filtered out and the result is empty.

**Remediation/acceptance:** Apply eligibility before candidate-window truncation. Evaluate bounded adaptive retrieval separately when the global ANN fetch itself lacks eligible candidates. Maintain I08's unified filters.

### I10 — Torn-v2 repair can skip a missing column

**Category:** Software bug / migration. **Evidence:** Static. **Location:** `db.py:59–63,110–137`.

If `agent_id` already exists, repair removes all ALTER statements, including the missing `trigger_patterns` addition. The version can advance while the required column remains absent. Business impact: affected legacy homes appear repaired but fail on recall.

**Proposed regression:** Construct a tiny version-1 fixture with only `agent_id` added; run migration and inspect required columns.

**Expected:** Both version-2 columns exist before recording completion. **Source-indicated actual:** `trigger_patterns` can remain missing.

**Remediation/acceptance:** Check each required column independently and verify schema postconditions before the version bump. Include both single-column torn states and normal upgrades.

### I11 — Consolidation loses provenance and agent scope

**Category:** Software bug / data quality. **Evidence:** Static. **Location:** `consolidate.py:53–67,149–154,209–268`.

Facts receive the entire episode window as provenance, including failed chunks. Episode selection does not carry agent identity through extraction; deduplication is global and new facts omit `agent_id`. Business impact: misleading audit trails, loss of scoped discoverability, and unintended cross-scope deduplication.

**Proposed regression:** Six tiny episodes in two chunks; mock one extraction success and one ordinary failure. Assert each fact references only its successful source chunk. Separately check scoped extraction and deduplication under the agreed policy.

**Expected:** Accurate source association and explicitly defined scope propagation. **Source-indicated actual:** All-window provenance and unscoped outputs.

**Remediation/acceptance:** Carry fact, source IDs, and scope together; partition extraction/deduplication as required. Scope behavior needs a product decision; chunk provenance correctness does not.

### I12 — Saved soak fails its latency gate and is not steady-state

**Category:** Performance / benchmark quality. **Evidence:** Recorded plus static methodology. **Location:** `bench/soak-results.txt:1–9`; `bench/soak.py:31–33,116–126,149,161–163`.

The saved 60.1-second run records recall P95 increasing from 3.30 to 34.38 ms, **10.42 times**, against a four-times gate. Overall recall P95 is 32.10 ms. RSS rises 36.9 to 61.9 MB; peak WAL is 1.88 MB. These are historical measurements, with no verified source-revision binding.

The current workload adds 11 and forgets five memories per iteration, increasing live rows by six despite claiming a bounded live set. Its `cycles` field counts recall samples. Business impact: the benchmark cannot distinguish an accumulation-driven slowdown from a leak or support reliable release claims.

**Proposed validation:** Use a short deterministic workload fixture to assert live-row balance and counter semantics; subsequently run separate bounded-live and growth experiments. Do not infer current throughput from the historical operation counter without verifying workload revision and counting semantics.

**Expected:** A steady-state test keeps its intended state bounded and yields comparable gates. **Actual evidence:** The saved latency gate fails; current code describes a different state trajectory than its docstring promises.

**Remediation/acceptance:** Separate stability and scale tests. Record revision, backend, dimensions, content sizes, live/deleted counts, flush time, embedding time, completed operations/second, and peak resource use. Profile before assigning a root cause. The saved WAL value is below its 80 MB gate and does not establish a WAL-growth defect.

### I13 — Recovery materializes unbounded SQL and embedding batches

**Category:** Performance / reliability. **Evidence:** Static. **Location:** `store.py:212–226`.

All missing IDs enter one parameterized `IN`, and their contents and vectors are materialized together. Recovery can exceed the connection-specific SQLite variable limit. Business impact: sidecar loss can cause startup failure precisely when reconstruction is needed.

**Quantified impact:** For missing count M, dimension d, and total content size C, transient content/vector materialization scales with C plus M times d. At 100,000 rows and 768 float32 dimensions, the vector output alone is 307.2 MB decimal, excluding content, Python overhead, and the resident index. This is arithmetic, not measured RSS.

**Proposed regression:** Five tiny rows, injectable batch limit two, recording fetch/encoder doubles; verify all IDs are covered without exceeding batch limits.

**Expected:** Bounded batches respecting the active connection's variable limit. **Source-indicated actual:** One full-size query and encoding batch.

**Remediation/acceptance:** Chunk fetching, encoding, and index addition; preserve dirty state until publication succeeds. Live/indexed ID sets remain linear unless separately redesigned. Do not claim fixed-memory recovery merely because embedding is chunked.

### I14 — CI and assertions do not enforce key guarantees

**Category:** Code quality / testing technical debt. **Evidence:** Static. **Location:** `.github/workflows/ci.yml:12–44`; `pyproject.toml:27–30,75–88`; `tests/test_reconcile.py:45–54`; `tests/test_portability.py:94–99`; `tests/test_enhancements_p1.py:313–342`.

CI installs the development extra but not the separate HNSW extra; HNSW tests may skip. Coverage settings exist without collection or an enforced threshold in the inspected workflow. A portability assertion checks `transient beta`, while its deleted fixture contains `forgotten beta`, and its comments accept ID reassignment. Re-embedding tests cover encoding failure before publication, not publication interruption. Business impact: green CI is weaker than the documented recovery contract.

**Proposed validation:** Review CI job collection and skip counts; map each recovery guarantee to an assertion and failure boundary. No coverage percentage is claimed in this audit.

**Expected:** Supported backends execute and key contracts fail tests when violated. **Source-indicated actual:** Optional-backend skips and weak assertions leave gaps.

**Remediation/acceptance:** Add a required supported HNSW job, collect branch coverage and establish a justified baseline, correct assertions, and require update/backup/migration interruption cases. The existing suite has substantial useful coverage; this is not a claim that whole modules are untested. Estimate excludes regression tests already budgeted under functional fixes where they overlap.

### I15 — Configuration silently coerces malformed values

**Category:** Software bug / maintainability debt. **Evidence:** Static. **Location:** `config.py:20–29,179–206`; `scoring.py:26–29`.

Unknown boolean strings become false; integers are parsed through floating point; ranking weights lack finite-value validation. Business impact: configuration typos silently alter behavior and invalid scores become difficult to diagnose.

**Proposed regression:** Table-driven tests for misspelled booleans, fractional integers, large exact integers, and non-finite weights; no database required.

**Expected:** Explicit accepted forms or field-specific errors. **Source-indicated actual:** Silent false/truncation/rounding and unvalidated weights.

**Remediation/acceptance:** Strict boolean tokens, exact integer parsing, finite-weight validation, and typed field-specific errors. Decide weight sign/normalization policy separately rather than assuming negative values are forbidden.

### I16 — Idle persistence does not honor the stated maximum interval

**Category:** Reliability / operational debt. **Evidence:** Static. **Location:** `store.py:242–290`; `server.py:201–207`.

The deadline is evaluated only after indexed mutations; the background loop checkpoints SQLite but does not persist vectors. Business impact: dirty sidecars may remain indefinitely after a final operation, increasing recovery work. This is not evidence of loss of committed SQLite rows.

**Proposed regression:** Fake clock, one mutation below the operation threshold, advance beyond the configured interval, invoke a maintenance tick, and assert one flush without another write.

**Expected:** Pending state flushes within the documented interval, allowing bounded scheduling/lock delays. **Source-indicated actual:** No idle-time enforcement exists.

**Remediation/acceptance:** A lifecycle-owned serialized maintenance tick using monotonic elapsed time; clean shutdown and no flush on already-clean state. Coordinate worker termination with I20.

### I17 — Future schema versions are not rejected

**Category:** Forward-compatibility safeguard debt. **Evidence:** Static. **Location:** `db.py:101–109`.

A version marker newer than all known migrations causes them all to be skipped. Business impact: an older binary can operate against an unsupported future schema; actual incompatibility depends on that future schema and is not established here.

**Proposed regression:** A valid tiny schema with its version marker advanced above the supported version.

**Expected:** Clear unsupported-version failure before application initialization. **Source-indicated actual:** Opening proceeds if the remaining probes succeed.

**Remediation/acceptance:** Reject newer versions with upgrade guidance; test supported, older, and future versions without modifying unsupported homes.

### I18 — Fixed-operation persistence overrides adaptive cost control

**Category:** Performance / policy debt. **Evidence:** Static. **Location:** `config.py:61–69`; `store.py:245–257`; `vector_index.py:114–117,202–215`.

The 64-pending-operation branch independently triggers a full index snapshot irrespective of adaptive elapsed time. This is a legitimate recovery-work tradeoff, not automatically a correctness defect; it conflicts with an unconditional bounded-amortized-cost claim. Business impact: write latency and disk traffic grow with the index.

**Quantified impact:** A brute snapshot's raw arrays use approximately N times (4d + 8) bytes. At 100,000 vectors and 768 dimensions that is 308 MB decimal per snapshot, excluding archive overhead. Under single-row growth and a fixed threshold B, cumulative snapshot volume grows proportionally to d times N squared divided by B. These are derived payload bounds, not measured disk traffic or latency.

**Proposed validation:** No-I/O persistence spy and fake clock over 64 small mutation notifications; verify which condition fires. Then measure actual flush bytes/time under the corrected I12 workload.

**Expected:** Documented recovery and cost budgets agree. **Source-indicated actual:** The hard count threshold bypasses adaptive time control.

**Remediation/acceptance:** Explicitly document the override or make the threshold cost-aware. Preserve a measurable recovery-work budget; do not optimize persistence by silently weakening guarantees.

### I19 — Batch forgetting repeatedly constructs the same set

**Category:** Code quality / localized performance bug. **Evidence:** Static. **Location:** `store.py:527–536`.

`set(forgotten)` is constructed inside the missing-ID comprehension for every wanted ID. For W requested and F found IDs, hashing work is proportional to W times F instead of W plus F. At W=F=1,000, this means roughly one million set-item insertions versus roughly one thousand, ignoring constants. This is an operation-count estimate, not measured speedup. Business impact: unnecessary CPU and lock hold time for large forgetting batches.

**Proposed regression:** Small present/missing/duplicate fixtures plus structural or instrumentation verification that the membership set is built once.

**Expected:** One set construction and unchanged result ordering. **Source-indicated actual:** Repeated equivalent construction.

**Remediation/acceptance:** Hoist the set; preserve existing ID deduplication and output behavior. SQL batch-size limits are separate from this optimization.

### I20 — Lifecycle cleanup can skip lock release

**Category:** Reliability / resource-ownership debt. **Evidence:** Static. **Location:** `server.py:112–123,177–229`; `store.py:986–991`.

Construction lacks explicit connection cleanup on downstream failure; post-lock setup lies outside the main cleanup scope; if `store.close()` raises, the following lock release is skipped. Business impact: in a process that remains alive, resources/ownership can linger and block retry. OS process exit releases locks, so this is not a permanent post-crash lockfile claim.

**Proposed regression:** Recording doubles inject construction, setup, and flush failures; assert connection closure, maintenance-worker termination, and lock release on every acquired-resource path.

**Expected:** Each acquired resource is released regardless of later failures. **Source-indicated actual:** Some outer cleanup paths are bypassed.

**Remediation/acceptance:** Nested context-managed ownership, with lock release in its own `finally`. `MemoryStore.close()` already closes its connection in a `finally`; preserve that behavior and fix the outer lifecycle rather than rewriting it unnecessarily.

### I21 — Content validation is inconsistent across mutations

**Category:** Software bug / resource-control weakness (CWE-20/CWE-400). **Evidence:** Static. **Location:** `store.py:354–365,479–521`; `portability.py:377–394,437–475`.

Creation enforces `max_content_chars`, but update and bulk import do not apply the same bound. Business impact: operators cannot rely on the configured size policy, and updates can incur unexpectedly large embedding/storage work. Local tool access is required; no exhaustion attempt was performed.

**Safe proposed regression:** Configure a tiny limit such as 16 characters. Verify an exactly-at-limit value succeeds and a 17-character ordinary string is rejected by creation, update, and import before any database or index changes.

**Expected:** The same documented content policy applies to all write paths. **Source-indicated actual:** Update/import can accept content that creation rejects.

**Remediation/acceptance:** Shared type/length validation before encoding or SQL mutation; preserve prior content, metadata, and vectors on rejection. Decide separate batch, metadata, and total-result budgets rather than treating a per-content limit as complete resource protection.

### I22 — Documentation contains contradictory operational contracts

**Category:** Code quality / documentation technical debt. **Evidence:** Static. **Location:** `README.md:170–174`; `ARCHITECTURE.md:72–79,113–115`; `portability.py:103–111,346–351`; `tests/test_portability.py:94–99`.

The README calls startup reconciliation deferred, while architecture and implementation provide it. The architecture's extension guidance describes changing embedders as requiring a fresh home even though re-embedding exists. Import docstrings promise ID preservation while implementation and a test comment accept reassignment. Business impact: incorrect recovery procedures, integration assumptions, and support advice; contradictory contracts also weaken review quality.

**Proposed validation:** Build a documentation-to-behavior checklist for recovery, re-embedding, import identity, persistence intervals, and ownership. Verify examples against controlled regression fixtures after the corresponding fixes.

**Expected:** One explicit contract per operation, backed by executable assertions. **Source-indicated actual:** Multiple conflicting claims.

**Remediation/acceptance:** Update README, architecture, docstrings, and tests together after contract decisions. Distinguish deliberate limitations from incomplete guarantees. Remove obsolete explanations rather than accumulating contradictory historical comments.

## Metrics and uncertainty ledger

| Evidence | Value / conclusion | Interpretation |
|---|---|---|
| Saved soak latency | 3.30 → 34.38 ms P95 by thirds; 10.42 times | Historical gate failure, not current runtime verification or root-cause proof |
| Saved soak RSS | 36.9 → 61.9 MB | +25 MB; does not independently establish a leak |
| Saved soak WAL | 1.88 MB maximum | Below the recorded 80 MB gate; no confirmed defect |
| SQL callback arithmetic | 5 million callbacks at 10,000 instructions each | Approximately 50 billion allowed instructions; not a measured execution time |
| Brute snapshot arithmetic | 308 MB raw arrays at 100k × 768 | Derived payload, not measured process RSS or disk write volume |
| `bench/results.md:17–32` | At 100k/ef=64: recall@10 0.281 (384 dim), 0.240 (768 dim); P95 0.628/0.825 ms | Historical synthetic ANN-only results, not full tool latency or representative-corpus accuracy |
| Production throughput | Not measured by this audit | Rebaseline completed operations/second, queue delay, and per-operation latency under a revision-bound workload |

No unmeasured speedup claim is included. In particular, neither a ten-times embedding optimization nor a fixed query-log deletion count was established. Missing indexes, expensive graph walks, startup model loading, and LLM batching are profiling candidates, not additional confirmed timed bottlenecks here. Dependency version ranges are not evidence of known CVEs; no advisory scan was performed.

The project already includes health diagnostics, query logging, tests, shutdown handling, and startup recovery. Their existence is not disputed; this report identifies particular gaps. Source size alone is not treated as a coding-standard violation, and no lint pass or measured coverage percentage is claimed.

## Remediation roadmap and business decisions

### Immediate containment

- Keep one active writable process per home, including CLI maintenance. Stop serving before backup/re-embedding until ownership and generation guarantees are repaired.
- Preserve original databases and exports; verify restores before relying on them or deleting originals. Do not remove a sidecar from a large home merely to force recovery: I13 must be considered.
- Limit arbitrary SQL and procedural-pattern authoring to trusted local workflows until resource bounds are enforced. This is an interim restriction, not a substitute for fixes.
- Keep security ratings scoped to local stdio; do not present agent labels as tenant isolation.

### Delivery streams

1. **Storage owner:** I02 with I20 → I01 → I04 with I21 → I03. Design the generation/revision interface early to avoid rework without making I01 and I03 circular dependencies.
2. **Retrieval/security owner:** I05 and I06; then I08 → I09, I07, I10, I11.
3. **Quality/performance work alongside both streams:** Start I14 immediately, implement I13 with recovery changes, correct I12 before deciding I18. Schedule I15–I17, I19, and I22 within the 30-day target window.

With two engineers, stream 3 consumes capacity from streams 1 and 2; it is not free parallel capacity. Re-estimate after P1 regression cases and compatibility decisions are agreed. If a 7-day P1 target cannot be staffed, retain containment and defer release rather than asserting completion.

**Required product decisions:** scope semantics for consolidation, trigger-language compatibility, exact restore-ID contract, and the acceptable vector reconstruction-work/persistence-cost tradeoff. Defaults proposed in the issue entries favor existing documented contracts and recoverability.

## Release acceptance criteria

1. **Identity and restore:** Gapped-ID exports preserve documented IDs, fields, and edge endpoints; invalid imports leave the home unchanged after close/reopen.
2. **Recovery:** Controlled failures at SQL/index publication boundaries recover committed content/vector revisions and consistent embedder identity. Compare vectors or exact retrieval in deterministic fixtures; assess ANN rebuilds with a quality tolerance rather than requiring bit-identical rankings.
3. **Ownership:** Every conflicting opener is refused before any mutation, including cleanup writes; exceptions release all acquired resources.
4. **Bounds:** SQL and trigger operations honor documented enforceable work, size, and cancellation limits; explicitly document limits of VM-callback deadlines.
5. **Retrieval:** Every returned item satisfies the agreed kind/scope contract, and eligible already-fetched candidates are not discarded prematurely.
6. **CI:** A required job executes HNSW tests without optional-backend skips; branch coverage is collected with an agreed baseline; crash/restore contracts have meaningful assertions.
7. **Performance:** Revision-bound bounded-live and growth workloads report P50/P95, throughput, live/deleted counts, flush time, RSS, and WAL. The corrected stability gate passes or an explicit workload-specific threshold is approved with evidence.

## Remediation status (updated 2026-09-18, post-implementation)

All 22 catalogued issues were implemented by a senior-engineer team (four
parallel streams plus lead integration) with focused regression tests. The
full suite passes: **268 passed, 4 skipped** (3 opt-in perf gates, 1
model-download semantic test), ruff and mypy clean, native HNSW contracts
verified locally.

| ID | Outcome | Evidence |
|---|---|---|
| I01 | Fixed: encode-before-commit + durable `pending_reembed` revision queue reconciled on open | `tests/test_audit_fixes_lead.py` (encode-failure atomicity; torn-update recovery) |
| I02 | Fixed: lock acquired before writable init in server and every writable CLI command; rejected openers persist nothing; read-only diagnostics (`stats`/`schema`/`export`/`query-log`/`query-stats`) take no lock | `test_second_owned_opener_refused_without_writes`; `test_concurrent_serve.py` |
| I03 | Fixed: backups carry `index_dirty=1`; re-embed publishes identity atomically behind a durable `reembed_pending` marker that forces rebuild-on-open | `test_reembed_pending_marker_rebuilds_on_open` |
| I04 | Fixed: import preserves ids/endpoints, validates before writing, single rollback-protected transaction, reports applied counts | `test_portability.py::test_export_skips_dead_rows_and_their_edges`; `test_import_rejection_is_atomic` |
| I05 | Fixed: budget counts VM instructions (interval 1,000), monotonic deadline, result byte cap, handler cleanup on all exits | `test_tiny_instruction_budget_interrupts`; `test_search_result_byte_cap` |
| I06 | Fixed: admission bounds (count/length, creation + import) and matcher bounds (query size, per-recall evaluations) | `test_trigger_admission_limits`; `test_oversized_query_skips_trigger_matching` |
| I07 | Fixed: per-row `lastrowid` capture replaces the highest-ID heuristic | `tests/test_audit_batch_identity.py` (3 orderings; failed 2/3 pre-fix) |
| I08–I10 | Fixed: unified recall filter contract; eligibility before truncation; independent torn-column healing + postcondition check | `test_recall_split.py`; `test_audit_fixes_schema.py` |
| I11 | Fixed: per-chunk provenance and inherited agent scope; dedup partitioned by scope | `test_consolidation_provenance_and_scope` |
| I12 | Implemented: steady-state bounded-live workload, ops semantics, revision-bound metrics (source digest over harness + runtime). The pre-fix runtime failed the corrected gate (p95 ratio 5.14); on the **final runtime** (2026-09-19, after the re-embed lock-ordering fix) a 2-minute stability run passes: p95 ratio 3.758 ≤ 4.0, live set stable at 110, 381 ops/s. The separate growth experiment scales 110→4000 live rows: recall p95 ≈1→5.6 ms across size buckets, 629 ops/s | `bench/soak-results-v2.txt` + `bench/growth-results.txt` (both bound to `source_identity_sha256: c1e17113…`) |
| I13 | Fixed: batched fetch/encode/remove bounded by `_RECONCILE_BATCH_CAP` and the connection variable limit | `test_reconcile_batches_fetch` |
| I14 | Implemented: required HNSW job with skips-as-failures, branch coverage with enforced 80% floor (measured 85.3%), corrected portability assertion, publication-interruption test | `.github/workflows/ci.yml` |
| I15–I19, I21 | Fixed: strict config coercion; idle `maintenance_tick`; future-version rejection; I18 override documented; set-hoist; shared `validation.py` across create/update/import | `test_audit_fixes_lead.py`; `test_audit_fixes_schema.py` |
| I20 | Fixed: nested context-managed ownership; lock release in its own `finally` | `open_owned_store`; serve lifecycle |
| I22 | Implemented: README/ARCHITECTURE/CHANGELOG now state one contract per operation matching the code | This revision |

**Remaining limitations (explicitly not closed by this pass):** the
historical v1 soak gate remains non-comparable and is not re-evaluated;
VM-callback deadlines still cannot interrupt long scalar functions mid-call
(documented); ANN rebuild ordering is not bit-identical without deterministic
configuration; the 30-day P3 items landed early but their product decisions
(consolidation scope policy beyond the default, trigger-language migration)
remain open choices.

**Status:** Catalog complete. Remediation implemented and regression-tested;
release acceptance criteria 1–7 verified (criterion 7 by the final-runtime
stability run above).
