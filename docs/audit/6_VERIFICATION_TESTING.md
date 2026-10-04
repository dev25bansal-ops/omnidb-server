# 6 · Verification & Testing Strategy

Source: **E1** (unit+property) · **E2** (integration+contract) · **E3** (security) · **E4** (performance) · **E5** (UAT).

---

## §1 · Unit & property testing (agent E1)

The mutation report already names the root cause: *"neither pure function has a single direct
unit test."* Its own killing tests are written out — **extracting them takes the mutation score
33% → ~97% in ~4 h.**

| Mutants killed | Test |
|---|---|
| rrf `k+p+1`→`k+p`; `(k+p)**2`; offset `+1`→`+2`; default `k=60`→`1`; `rankings[:1]`; accumulate→overwrite | `test_rrf_exact_rank_weights` — `rrf([[10,20,30]])[10] == 1/61` (±1e-12) |
| fts `\w+`→`\w`; OR→AND; no `[:max_terms]`; no `.lower()`; default 24→1; empty→`['']` | `test_fts_*` against a reference implementation |
| scoring: drop `max(half_life,1e-3)`; drop `max(access_count,0)`; weights → 0; ignore weights; omit freq; floor 1e-3→1e6 | `test_exact_values_pin_weights_and_defaults` — `score == rrf + rw*recency + fw*frequency` **exactly** |

**Projected 29/30 (~97%).** Only #25 is equivalent (`\w+` cannot emit a quote).

### The mechanical reason 7 scoring mutants survived
Every current property is a **non-strict inequality**, trivially satisfied when a term vanishes.
Replace with **exact equality**. That one change kills 06/07/08/12.

### Five property designs
1. **Fusion exactness + monotone shift.** For all rank lists R, k>0: `rrf(R,k)[id]` equals the
   closed form, and `k1<k2 ⟹ rrf(R,k1)[id] ≥ rrf(R,k2)[id]` elementwise. The existing test is
   directionally right but vacuous against mutants 15/16/20 — pair it with the ratio bound
   `(k2+p+1)/(k1+p+1)`.
2. **FTS injection safety, both halves.** Either the expression is empty, or every
   quote-doubled segment is well-formed and FTS5 accepts `MATCH` without raising. **Plus the
   equivalence half**: the output equals a reference join over the first 24 `\w+` tokens of the
   lowercased input. One differential property kills 24/26/27/28/29/30.
3. **Scoring totality + exact decomposition.** 3 finite floats, `recency ∈ (0,1]`,
   `frequency == log1p(max(access_count,0))`, and the score identity holds **exactly**.
4. **Trigger-admission bounds.** Validation returns the input iff `len ≤ 32` and every pattern
   is a str with `len ≤ 512`; else raises, and **no partial write survives**. Sweep `remember`,
   `remember_many` **and** `import_jsonl` — today only 2 hand-written cases exist.
5. **Graph BFS equivalence** — already implemented in `test_graph_fuzz.py`, the strongest
   property in the repo and the model for everything else. Widen `max_hops` from `{1,2,99}` to a
   generated range to actually exercise the clamp-to-8 boundary.

### Crash window: stateful model + metamorphic
**Gap:** the 5 durable markers are asserted at **7 hand-picked interruption points only.** No
test enumerates boundaries, and `test_issue_fixes.py:104-107` is the weak-assertion pattern —
a conditional `if … pass` that proves nothing.

**`RuleBasedStateMachine`** over commands: remember / update / forget / supersede / flush /
close / reopen / `kill_at(step)`. Invariant after every command and every simulated kill:
`indexed_ids ⊆ live_ids`, and within 2 opens `indexed_ids == live_ids`. Kill points: before
flush-persist · after `index_dirty=1` · after vector add before persist · after meta write ·
mid-`rebuild_index`.

**Metamorphic relations:** isolation (adding an unrelated memory leaves the recall id-set
identical) · no resurrection (supersede, then any sequence of writes+reopens) · backup/restore
id-preservation · reconcile idempotence (`reopen(reconcile) == reopen(reopen)`) · validation
round-trip.

### Coverage strategy
`server.py` 63.5% is **structural** (stdio tests are out-of-process). `cli.py` 0% is
**integration-only** — add exit-code tests only, do not chase coverage there. The 80% floor is
already met at 85.3%. Real ROI is `rerank.py` (the `REGISTRY` unknown-name branch is untested)
and `llm.py`.

### Two things 29 of 30 agents missed
- **`hypothesis` is imported at module level in 3 test files but lives only in `[dev]`** →
  collection error on a clean install. Fix: a `test` extra + a CI job + a `conftest`
  `collect_ignore` guard.
- **`mutation_check.py` exits 0 regardless of kill rate** — it cannot fail. Add `--min-score`,
  fail on `NOT_APPLICABLE` (4/30 anchors silently do not apply), wire it path-filtered and
  blocking for `fusion.py`/`scoring.py` only. Realistic target ≥90% in 2 weeks; **never gate
  at 33%.**

### AUDIT-I spot-check (3 of 22)
- **I01 holds** — `store.py:595-603` writes `pending_reembed` before commit; `:261` unions it
  into `missing`; `test_pending_revision_recovered_on_reopen` asserts the vector after reopen.
- **I06 holds for admission, not for time** — the bounds exist and are tested, but bounding
  input does not bound backtracking. The AUDIT.md "Fixed" label **overstates**.
- **I13 holds** — `store.py:263` batches on `SQLITE_LIMIT_VARIABLE_NUMBER - 2`;
  `test_reconcile_batches_fetch` monkeypatches the cap to 2 and records.

**Overall:** "all 22 implemented" is **directionally true but overstated for I12 and I14**,
whose cited evidence contains failing gate values under gitignored paths.

---

## §2 · Integration & contract testing (agent E2)

### Verified gaps
- The tool list is asserted in **5 places that disagree** (10/10/10/7/7).
- `test_stdio_conformance.py:51` asserts `EXPECTED_TOOLS <= names` — a **subset** check over **7**
  tools. It therefore passes today while **never touching `schema`, `update_memory`,
  `forget_many`**, and it is itself a 4th disagreeing tool count.
- The **AUDIT-I02 "rejected second server persists nothing" claim is untested** —
  `test_concurrent_serve.py:139` asserts only `returncode != 0`; `_wait_for_lock` proves the
  *first* server wrote `.lock`, nothing about the second writing nothing.
- Export is **not** byte-identical across platforms: `portability.py:320` opens in text mode
  with `newline=None`, so Windows emits `\r\n`.
- `server.py:221-228` and `doctor.py:139-148` are two hand-duplicated copies of the
  msvcrt/fcntl branch.

| ID | Test | Proves | Effort |
|---|---|---|---|
| T1 | `test_tool_manifest_matches_server` (golden JSON) | Names, descriptions, input+output schemas of all 10 tools pinned; kills all 5 divergent literals | 2-4 h |
| T2 | `test_mcp_call_tool_all_10_in_process` (real store, valid + invalid args) | Every tool body incl. `_call`/`anyio.to_thread` runs under coverage → `server.py` 63.5%→~90% | 3-5 h |
| T3 | `test_server_initialize_capabilities` (in-memory FastMCP client) | protocolVersion, serverInfo, `capabilities.tools` | 2-3 h |
| T4 | `test_error_envelope_3_classes` | `isError=True` + text content, no exception on the wire | 2-3 h |
| T5 | `test_export_import_export_byte_identical` | Byte-stability of the portability format (**unmeasured today**) | 2-3 h |
| T6 | `test_reject_path_writes_nothing` | AUDIT-I02 invariant | 1-2 h |
| T7 | `test_readonly_cli_during_serve` | Read-only diagnostics survive WAL while locked | 2-3 h |
| T8 | `test_writer_cli_refused_during_serve` | Second-writer refusal for **all** mutating commands, not just `serve` | 1-2 h |
| T9 | `test_second_serve_exit_code_is_3` | Pins the documented code | 15 min |
| T10 | `test_sigkill_then_reserve` | OS-released lock, no stale-lock brick | 2-3 h |
| T11 | `test_adapter_contract_spec` | Method names, arity, param names, return keys, `OMN-*` codes | 4-6 h |
| T12 | `test_adapter_contract_vs_live_framework` (`OMNIDB_TEST_FRAMEWORKS=1`) | We satisfy the real langgraph/llama_index duck types | 3-4 h |
| T13 | `test_llm_wire_contract` (recording `ThreadingHTTPServer`) | POST path, model, timeout, `Authorization` present iff key set | 3-4 h |
| T14 | `test_llm_malformed_response_never_corrupts` | Failed chunk stays retryable, no partial writes | 3-4 h |
| T15 | `test_replace_retry_under_windows_lock` | `_replace_with_retry` (`vector_index.py:237`) — **dead on POSIX CI today** | 2 h |
| T16 | `test_home_uri_with_special_chars` (`#`, space, unicode) | `server.py:178` `file:...?mode=ro` URI is not corrupted | 1 h |
| T17 | hnsw job on windows+macos; `-k requires_hnsw` instead of node ids | Backend builds natively per-platform | 2-3 h |
| T18 | nightly `fastembed-required` with a pre-warmed cache | The semantic extra is not silently broken | 2-3 h |

### T1 + T2 — the highest-value pair
Emit `docs/tool_manifest.json` from `create_mcp(_NoopStore())` with an `--update-manifest` flag;
assert equality. Then parametrise `call_tool` over `(tool, valid_args, expected_key)` against
a **real** store, and for each tool also call it with one deliberately wrong type asserting
`isError`. Replace `EXPECTED_TOOLS` with the manifest's name set and assert **equality**.

**Limits to state explicitly:** in-process `call_tool` skips stdio framing, JSON-RPC ids,
notification delivery, and `serve()` itself (lock, banner, maintenance thread). **It buys
coverage, not transport fidelity.** Keep `test_stdio_conformance.py`; do not let it become the
only test.

### What NOT to test
- **The official MCP SDK conformance suite** — a client/server *pair* fixture that mostly tests
  the SDK. Vendoring it adds a large dep for near-zero OmniDB-specific signal. Build T3/T4 from
  the spec instead: ~2 conformance cases' worth of value at 1/20 the cost.
- **Byte-identical export across OSes** — `portability.py:320` makes it false by construction.
  Assert semantic equality across platforms; bytes within a run.
- **Latency/throughput here** — that is `test_perf_smoke.py` + `bench/`. The round-trip property
  is about determinism, not speed.
- **`fastembed` model quality in the main matrix** — network-dependent; nightly only.
- **Property-fuzzing the tool surface** — a 10-tool schema is fully enumerable; a property test
  buys nothing.

---

## §3 · Security testing (agent E3)

**Correction to my own brief:** the VM-step and byte-cap budgets *are* tested
(`test_audit_fixes_lead.py:164-189`). The genuine gap is the **wall clock —
`_QUERY_WALL_CLOCK_LIMIT_S` appears in exactly two places (`store.py:32`, `store.py:773`) and
ZERO test files.** My brief's "entirely untested" was overstated.

| ID | Test | Threat | Effort | Safe? |
|---|---|---|---|---|
| T1 | Wall-clock deadline: monkeypatch `store_mod.time.monotonic` to a counter jumping +2 s on the 3rd call; assert `OMN-011` **and** that the sentinel row survives | slow-query DoS | 30 m | ✅ fake clock |
| T2 | Deadline already expired (`monotonic` returns `t+10`) on a trivial `SELECT 1` | proves the arm is live, not dead code | 20 m | ✅ |
| T3 | Byte-cap 8× undercharge: a 2 MiB BLOB cell with a 1 MiB cap **must NOT** truncate today — pins the gap | the cap's own flaw | 20 m | ✅ |
| T5 | Row-cap: `max_search_rows=3` over 5 rows → `truncated is True`, `len(rows)==3` | untested branch | 15 m | ✅ |
| T6 | `search("SELECT agent_id, content FROM memories")` returns **all** agents' rows | asserts the *current* MT-02 leak | 15 m | ✅ |
| T7 | Enumerate the `arg2` names the authorizer sees for `load_extension`/`readfile`/`writefile`/`edit` | SEC-09 | 1 h | ✅ |
| T8 | Assert `sqlite3.enable_extension == "false"` on a fresh connect; pin as `xfail` where it is on | the control is *inherited*, not ours | 20 m | ✅ |
| **T9** | **AST no-injection test** | SEC-10, 28-30 unenforced sites | 2-3 h | ✅ |
| T11 | Matcher eval cap: 40 rows × 32 patterns = 1280 candidates, cap 2048; assert the triggered set **equals** the full-scan result | SEC-05 | 1 h | ✅ |
| T13 | Nested-quantifier static screen (`(a+)+$`) on validation | SEC-05 | 1 h | ✅ |
| T14 | Zip size limit via a **declared-size** fixture (real `ZipInfo` claiming 2 GiB, 1-byte payload) | SEC-07 | 45 m | ✅ |
| T16 | `llm_base_url` = `http://internal-llm` + key set → `ConfigError`; loopback and https allowed | SEC-06 | 45 m | ✅ |
| T17 | Rewrite the 4 tenancy xfails to **assert the current leak** | MT-06 | 1 h | ✅ |
| T18 | Credential-literal scan (`sk-[A-Za-z0-9]{20,}`, `AKIA`, private-key header) | OPS-01 | 1 h | ✅ |
| T19 | `config` redaction via **TOML as well as env** (today only env is tested) | credential disclosure | 20 m | ✅ |
| T20 | Fuzz the JSONL importer, the config TOML loader, the skill-bundle parser, `_coerce` | parser robustness | 4 h | ✅ |

### T9 — the highest-value test, and it needs a negative control
`ast.parse` each module. Find every `Call` whose func attr ∈ `{execute, executemany,
executescript}`. Walk the SQL argument for `FormattedValue` / `%` / `.format` / `+`. Classify
each as **SAFE** only if it (a) resolves to a module constant via `import as`, (b) is a
`', '.join()` over names from a static list, or (c) is an int `*_CAP`. Assert `unsafe == []`
and **pin the sink count**. **Then inject a bad `execute` into a temp copy and assert the
classifier flags it** — without that, a classifier bug makes the test vacuously pass.

### T14 — never construct a real bomb
Build the zip with a `ZipInfo` whose `file_size` claims 2 GiB while the payload is one byte:
**decompression never runs**, so the test measures the declared-size gate, not memory. Assert
the store is untouched (sentinel row still recallable). The fix must check `zf.infolist()`
**before** `zf.read()`.

### T17 — make the current contract mechanically pinned
Replace each bare `xfail` with a test that **asserts the leak exists today** (the dense leg
returns beta's id to alpha; `stats()["agents"]` contains `"beta"`; `related()` walks the forged
cross-tenant edge), marked `xfail(strict=True, reason="MT-06: agent_id is a label, not a
boundary")`. A fix → XPASS → **strict xfail turns the suite red**, demanding the marker be
removed. A widening → the assertion fails → red. Either way the contract is pinned, which
§3.24 of my brief says is missing.

### Adversarial-fixture discipline (a rule, not a test)
No regex, recursive CTE, zip, or fuzz input may run against a **live** home. All adversarial
fixtures live in `tmp_path`, are **code-generated** (never a committed binary), and are capped:
≤64 KiB in, ≤1 MiB declared out, ≤2 s wall. Add a session fixture asserting
`Path(store.home).is_relative_to(tmp_path)`.

---

## §4 · Performance testing (agent E4)

### The two gates that cannot do their job
**1. The soak gate is a ratio of two p95s** — the *least* stable quantile — over a denominator
that `growth-results.txt:25-39` shows is **non-stationary by design** (brute p95 rises with
corpus size). That is why 3 runs with an **identical `source_identity_sha256`** gave
**4.156 (fail) / 2.379 / 2.632**.

**2. The baseline gate reads a maximum** — `bench/baseline.py:137-138`:
`best = max(hnsw_scores + [min(sq8_scores)])`, `gate = best >= 0.95`. It passes on the best
cell (1.0 at 384-d/1k) while the worst in the same sweep is **0.17** (`results.md:29`).

**3. There is no latency baseline at all.** `test_perf_smoke.py:24` uses 250 w/s (≈14× headroom)
and `test_remember_many.py:78` asserts 800 w/s with n=1 — effectively dead. And CI sets
`OMNIDB_RUN_PERF=1` repo-wide (`ci.yml:36`), so that 800 w/s assertion **executes in all 9 cells
on loaded runners**. This is CI-05.

### The replacement soak gate — exact formula
Seeds `s ∈ {1..5}` drive the corpus generator and forget schedule; each runs
`--minutes 1 --backend brute --iterations 0`; every recall latency is tagged with a live-size
bucket `L ∈ {<500, 500-1k, 1k-2k, 2k-4k}` (the buckets in `growth-results.txt`). R=5 runs
pooled. B=10,000 bootstrap resamples of the pooled vector.

```
budget_ms(L) = 2.0 / 3.0 / 4.0 / 5.0        # L ascending, committed as bench/soak-budget.json

PASS  ⟺  CI95_upper( p95*_pooled , bucket L )  ≤  budget_ms(L)          for all L
      ∧  CI95_upper( p95*_pooled , unstratified ) ≤ 5.0 ms
      ∧  β_lo  ≥ 0                            # Theil-Sen slope of log p95 per decile
      ∧  ops_per_second ≥ 0.80 × 440          # 0.8 × the measured 548.91
      ∧  rss_peak − rss_start < 300 MB  ∧  wal_max < 80 MB

CREEP ⟺  β_hi  >  log(1.25)                   # MDE = 25% per run, declared only
```

**The creep/noise separation is carried by the bootstrap CI, not by a ratio.** Run-to-run
variance inflates CI width, so 2.379 vs 4.156 can no longer flip a verdict — it can only fail
when the whole distribution shifts. `PASS ∧ CREEP` is impossible by construction, so creep is
reported, not asserted.

### The replacement vector gate — per-cell frontier
```
frontier(dim,n) = max{ recall@10 : ef ∈ {96,128} }     # per (dim,n), NOT globally
floor_hnsw(n)   = clamp(1 − 0.55·log10(n/1000), 0.30, 0.99)

PASS ⟺ ∀(dim,n): frontier ≥ floor_hnsw(n)              # 1k→0.99, 10k→0.44, 100k→0.30
     ∧ ∀(dim,n): recall@10 monotone non-decreasing in ef    # catches ef-order inversion
     ∧ min(sq8) ≥ 0.85  ∧  sq8_p50 ≤ 150 ms
```
Zero failing cells permitted. **At the shipped `ef_search=64` the 100k cells sit at 0.378/0.278
— this gate fails today, and that is the honest reading of `results.md`.**

### The latency regression gate
`bench/latency-baseline.json` (committed; keyed by python, os, dim, n, op, p50, p95, n_samples,
seed), built by 15 paired same-process runs; `σ = 1.4826·MAD`. Gate `p95_now > μ + 6σ` on
1,000 fixed-seed single-thread ops with a 200-op warm-up, hashing embedder, 1k-row fixed corpus.

### The missing measurements
| ID | What | Cost | Cadence |
|---|---|---|---|
| M-01 | Concurrency ceiling at 1/8/64 clients — measure the **lock-hold histogram**, not a QPS number | 9 min | nightly |
| M-02 | Backup/restore timing (export→import), bytes, RAM delta | 5 min | nightly |
| M-03 | Cold start on a **populated** home + crash recovery; time `_reconcile_index` | 3 min | nightly |
| M-04 | **hnsw backend soak — never run** (both soak and growth are `backend: brute`) | 5 min | nightly |
| M-05 | Peak RSS at open / steady / post-flush, 384- and 768-d × 1k/10k/100k | 2 min | nightly |
| M-06 | Graph latency (`related`) vs 1k/10k/100k | 3 min | nightly |
| M-07 | Per-tenant disk: home bytes per 1k memories × tenants | 1 min | nightly |
| M-08 | Scale >57,638 docs and >4,000 live memories | 20 min | `OMNIDB_RUN_PERF=1` |
| M-09 | Recall quality **through `recall()`** — the v3 bench substituted `BruteForceIndex` | 3 h | nightly |
| M-10 | Hostile query (k=1000, agent overfetch, 1 MB content) → worst-case p99 | 30 min | `OMNIDB_RUN_PERF=1` |
| M-11 | Write amplification: bytes-written/op vs N | 10 min | nightly |
| S-01 | ≥3 assertions: valid rows returned, `count>0`, no exception | <1 s | **9-cell matrix** |

### The concurrency ceiling, stated honestly
`_serialized` holds the RLock across the ONNX encode, FTS, ANN and commit, so
`ops/s ≤ 1/critical-section` **independently of client count**. One client: 548.9 ops/s
(measured). 8 and 64 clients: *the same*, plus queueing. State it as: **"throughput is
serialized; concurrency buys latency, not throughput. The 64-client ceiling equals the 1-client
ceiling ≈ 550 ops/s, and p99 latency at 64 clients ≈ 64 × critical-section."**

### Write amplification — the number that sizes the persist policy
`persist` is a full `np.savez` of `N×(4·dim+8)` bytes (`vector_index.py:114-125`), triggered
every `persist_every_ops=64`. At 512-d/4k that is 8.2 MB ÷ 64 = **128 KB/op**. At 768-d/100k it
is 308 MB ÷ 64 = **4.8 MB/op** — and soak shows 11.4 ms/flush × 23.9 ops/flush. The policy is
only tunable once this curve is measured; the target is ≤32 KB/op at the shipping corpus.

### The cadence rule
A perf assertion may enter the 9-cell matrix **only if** (a) it asserts a **floor** (never a
time bound), or (b) its threshold exceeds the measured **p99.9** noise floor, runs <20 s, and
uses a fixed seed with a warm-up. Anything with a non-zero false-positive rate is nightly-only
and **blocks the merge queue, never a PR.** Anything >5 min sits behind `OMNIDB_RUN_PERF=1`.

**Fix CI-05 concretely:** delete `test_remember_many.py:78`'s absolute 800 w/s and replace it
with a **self-calibrating paired assertion** (`t_batch ≤ 2×t_control`, both measured in the same
process) — immune to runner load by construction.

---

## §5 · UAT & operational readiness (agent E5)

`docs/uat.md` has 10 tasks and **all 27 checkboxes are unchecked** — it is an unexecuted
document, not evidence. `server.py:24-100` does expose exactly the 10 tools `uat.md:25-26` lists,
so *that* assertion is correct. But the CLI has **19 top-level + 3 `skill` leaves = 22**
commands and the trail touches ≤10.

| Tool / command | UAT | Error path | Verdict |
|---|---|---|---|
| remember | 2 happy | none | OK, weak |
| recall | 3 happy | none | **Unfalsifiable** — `:66-67` says a miss "is not a failure" |
| search | 8 | OMN-020 | **The only real negative test in the doc** |
| schema | listed only | none | **No task** |
| related | 7 happy | max_hops=0 / unknown id untested | Weak |
| update_memory | 6 happy | stale-id untested | Weak |
| forget | 5 happy | dead/unknown id untested | Weak |
| forget_many | listed only | none | **No task at all** |
| consolidate | 10 | 0-episode, bad `hours` untested | **Unfalsifiable** — conditional on `llm_used` |
| stats | 9 | n/a | OK |
| init · re-embed | none | — | **re-embed = 0.1.3's headline feature, 0 UAT** |
| backup / restore | none | — | **No `restore` command exists** |
| export / import | none | — | only in DR prose |
| doctor · config · skill · compact/purge/check/query-log | none | — | **No task** |

**8 of 10 tools have no error-path acceptance criterion.** And the only automatable part of the
checklist cannot detect a `forget_many`/`update_memory`/`schema` regression.

**`reembed_pending` durability is verified only by in-process unit tests** — **no CLI-level,
cross-process, or kill-mid-migration test exists.** Also: `doctor` has **12** checks, not 13.

### Prioritized backlog
| # | Item | Effort |
|---|---|---|
| 1 | `uat-ci.py`: exact-set `== {10 tools}` + an error case per tool in the stdio round-trip | 4 h |
| 2 | `re-embed` UAT: model refusal, SIGKILL mid-run → marker set → reopen heals → identity correct | 1 d |
| 3 | `dr_drill.py`: seed 100 → backup → truncate the db → restore → assert count and top-1 recall | 1 d |
| 4 | Upgrade matrix: v1→3, torn v2, `user_version=99` refuses, foreign embedder → OMN-006 | 6 h |
| 5 | First run: init → serve → recall asserts the banner + one self-probe | 4 h |
| 6 | doctor/config UAT incl. exit 4 and the provenance table | 3 h |
| 7 | Windows: `os.replace` retry (bare `OSError` at `portability.py:151-155`), clean SIGTERM flush | 1 d |
| 8 | skill export/import/list round-trip + zip-bomb rejection | 4 h |

### The UAT gate, stated precisely
`uat.md` has 10 tasks and **all 27 checkboxes are unchecked** — it is an unexecuted document,
not evidence. `server.py:24-100` does expose exactly the 10 tools `uat.md:25-26` lists, so *that*
assertion is correct. But the CLI has **19 top-level + 3 `skill` leaves = 22** commands and the
trail touches ≤10.

**8 of 10 tools have no error-path acceptance criterion.** And the only automatable part of the
checklist cannot detect a `forget_many` / `update_memory` / `schema` regression.

**The gate is items 1 and 2 above.** Until `uat-ci.py` runs in CI, UAT is a document that
describes intent. Everything else in this section is diagnostic depth; those two are the gate.

---

## §6 · What is already well-tested — do not re-audit it

A test-strategy document that only lists gaps misleads. These are genuinely strong:

- **`test_graph_fuzz.py`** — a real property-based BFS-equivalence check against a reference
  implementation, with generated `max_hops`. **This is the model for the five properties in §1.**
  Widen it, don't replace it.
- **`test_security_matrix.py`** — a 12-case deny matrix and a 5-case allow matrix, each
  asserting a specific `OMN-*` code and each leaving a sentinel row behind. It includes a
  **negative control** (`load_extension` asserted fail-closed, written to survive a build that
  omits the toggle) and asserts `conn.in_transaction is False` after four denials. **Most
  projects get the last one wrong.**
- **`test_tenancy_boundaries.py`** — the four MT-02 bypasses as `@pytest.mark.xfail(strict=True)`.
  That is the correct way to record a known-unfixed defect: it cannot be silently "fixed", and a
  widening fails the suite.
- **`test_embedder_identity.py`** — pins the `fastembed:<model>` round-trip and the reopen path.
  This is what **refuted** the P0 lockout claim in §0.
- **`test_coldstart.py` + `test_chaos.py` + `test_hardening.py`** — real crash-window coverage
  at 7 hand-picked interruption points. Not enough (see §1), but not decorative.

### The single highest-value test in the whole strategy
**T9 in §3 — the AST no-injection test with a negative control.** It costs 2-3 h, needs no
exploit, and it is the only mechanism that converts a *structural* safety property (~30
interpolated-SQL sites that are safe only because every spliced value happens to be a module
constant) from an unwritten assumption into an enforced invariant. A safety property that is
true by placement and unguarded by a test is one refactor away from a SQL injection.

### The one that most changes the project's honesty
**§4's soak-gate replacement.** The current gate compares two p95s over a denominator the repo's
own `growth-results.txt:25-39` shows is non-stationary by design — which is why three runs with an
**identical `source_identity_sha256`** produced **4.156 (fail) / 2.379 / 2.632**. A gate that
produces three different verdicts on the same bytes cannot be used to make a release decision.