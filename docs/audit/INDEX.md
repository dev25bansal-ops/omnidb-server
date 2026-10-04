# OmniDB 30-Agent Audit — Deliverable Index

**Status: COMPLETE.** 30/30 agents. 11 documents, ~265 KB.

## The one thing to know

The 30 agents were read-only (no write tool), so their reports came back to me as tool results.
Six category documents capture the synthesis of all 30. **Three agents (A1/A2/A3) were re-run a
second time** so their findings exist as full transcripts rather than as compressions — those are
the three long files below.

## Documents

| File | Contents | Source |
|---|---|---|
| [`MASTER_AUDIT.md`](MASTER_AUDIT.md) | The synthesis. Corrections first, then all 6 categories, then the ordered plan. | all 30 |
| [`1_ISSUES_REQUIRED_FIXES.md`](1_ISSUES_REQUIRED_FIXES.md) | 6 corrections · 15 bugs · 10 perf · 10 CVSS vulns + controls that passed · 23 quality · 8 architecture · 16 debt | A1–A5 |
| [`A1_SOFTWARE_BUGS.md`](A1_SOFTWARE_BUGS.md) | **Full bug transcript** — 16 bugs with runnable repros, expected-vs-actual, root cause, fix sketch + **controls that passed** | A1 |
| [`A2_PERFORMANCE.md`](A2_PERFORMANCE.md) | **Full perf transcript** — measured-detail table, 14 findings with **[M]/[D]/[U]** tags on every number, 21 measurement gaps, 8 contradictions | A2 |
| [`A3_SECURITY.md`](A3_SECURITY.md) | **Full security transcript** — 13 vulns, **CVSS arithmetic shown for every score**, two threat models, controls that passed | A3 |
| [`2_ENHANCEMENTS.md`](2_ENHANCEMENTS.md) | Core engine · retrieval · interface · ops · memory layer — each with trade-offs, plus **what was rejected and why** | B1–B5 |
| [`3_ADVANCED_FEATURES.md`](3_ADVANCED_FEATURES.md) | Memory architecture · retrieval innovation · agent-native UX · competitive positioning · resilience | C1–C5 |
| [`4_NEW_ADDITIONS.md`](4_NEW_ADDITIONS.md) | 9 new modules w/ LOC + verdicts · schema evolution (NOW/LATER/NEVER) · integrations · tooling · packaging | D1–D5 |
| [`6_VERIFICATION_TESTING.md`](6_VERIFICATION_TESTING.md) | Unit/property · integration/contract · security · performance (incl. the **exact soak-gate formula**) · UAT | E1–E5 |
| [`7_CROSS_VERIFICATION.md`](7_CROSS_VERIFICATION.md) | The adversarial wave. 10 verdicts, 12 bug verdicts, the impossible set, dedup list, the plan, the risk register | F1–F4 |

Category **5** (the ordered plan) is intentionally inside `MASTER_AUDIT.md` §7 rather than
duplicated — F4 produced it and it is the plan you act on.

## Agent roster — all 30 ✅

| Wave | Agents | Domain |
|---|---|---|
| A | A1 A2 A3 A4 A5 | bugs · perf · security · code quality · architecture+debt |
| B | B1 B2 B3 B4 B5 | core engine · retrieval · interface · ops · memory layer |
| C | C1 C2 C3 C4 C5 | memory arch · retrieval innovation · agent UX · competitive · resilience |
| D | D1 D2 D3 D4 D5 | new modules · integrations · tooling · packaging · schema |
| E | E1 E2 E3 E4 E5 | unit+property · integration+contract · security · performance · UAT |
| F | F1 F2 F3 F4 | claim verification · bug verification · integration consistency · roadmap |

**One caveat, stated plainly:** 4 agents' verbatim reports (A1/A2/A3/C4) were truncated out of
my context after their wave completed. Their findings survive in `1_ISSUES_REQUIRED_FIXES.md`
§1–§3 and `3_ADVANCED_FEATURES.md` §3.4 — written from the agents' output while it was in hand
**The truncation caveat is now closed.** A1/A2/A3 were re-run; their full transcripts are the
three `A*_*.md` files. C4's re-run already landed in `3_ADVANCED_FEATURES.md` §3.4.

## Corrections the audit forced on itself

F1/F2 refuted 4 claims — **2 of them mine, in my own `GROUND_TRUTH.md` brief**. The A1/A2/A3
re-run found **6 more**. Full detail in `1_ISSUES_REQUIRED_FIXES.md` §0, `A2_PERFORMANCE.md` §3–4,
`A3_SECURITY.md` §12, and `7_CROSS_VERIFICATION.md` §1.

| Claim | Verdict |
|---|---|
| `re-embed --model X` bricks the home (mine, rated **P0**) | **REFUTED** — `embeddings.py:128-140` resolves the stored model; 2 tests pin it. Residual is a 2-4 h UX gap |
| Raise `k_fetch` 50→150 to lift the recall ceiling | **REFUTED by the project's own bench** — scidocs nDCG 0.2084@w25 → 0.2083@w100 |
| Skill-trigger re-fetch unfiltered → cross-tenant leak | **REFUTED** — `triggered ⊆ pat_rows` by construction |
| RRF fusion is 15-20× smaller than the modifiers | **numbers wrong** (13-22×), direction right |
| The Qdrant latency gap is "not fixable" | **RESOLVED by measurement** — the `BruteForceIndex` substitution is **CONFIRMED**, but switching to HNSW makes end-to-end recall **32% slower** at 25k. The real root cause is the **FTS5 lexical leg**: ~~209.8 ms p50 at 60k, 75% of the total, growing 53× from 5k~~ — **the leg is real, those numbers are NOT.** Wave 3 showed both lexical harnesses measure an **empty FTS index**. Corrected measurement: `ORDER BY bm25` is **99.6%** of the leg, α = **1.149**, and the JOIN hypothesis is refuted. |
| `.npz` `load()` makes three full copies | **it's two** — `ascontiguousarray` on contiguous f32 returns the same object |
| The adaptive persist control is "overridden" | **it's dead code** — cannot engage unless a flush exceeds 62.5 ms; measured flushes are 4.8–9.8 ms |
| `source_identity_sha256` is why the soak harness is slow | **REFUTED** — runs once, ~177 KB, <5 ms against a 60 s soak |
| The query budgets are "entirely untested" | **overstated** — VM-step and byte caps *are* tested; the real gap is the 5 s wall clock |
| SEC-02's 9.5 implies confidentiality loss | **not independent** — the same attacker already has full read via SEC-01; it is complete *integrity* loss |

---

## Integrity log

Kept because this project has already shipped numbers that do not exist in any artifact.

| Date | Event |
|---|---|
| Audit wave 1 | 30 agents returned analysis; **only `MASTER_AUDIT.md` was persisted.** The other 29 reports existed only as tool results in my context. |
| Wave 1 close-out | 8 category documents written from that context. **6 of them were silently truncated mid-sentence** by long `replace_in_file` calls — I reported them as complete before verifying the tails. |
| A1/A2/A3 re-run | Attempted to have the agents self-persist. **Discovered subagents in this environment have no write capability** (Glob/Grep/WebFetch/WebSearch only), so the content returned inline and I wrote it. Full transcripts now on disk. |
| PERF-A2-07 resolution | `bench/verify_qdrant_rootcause.py` written and run. The A2 hypothesis was half-confirmed, half-refuted; root cause is the FTS5 lexical leg. |
| Integrity sweep | Checked every file's last line. **5 of 11 flagged as truncated were false positives** (a table's final row legitimately ends in `\|`). All 11 documents verified complete. |
| Wave 1 implementation | 4 folder teams (F1/F2/F3/F5) applied. `LICENSE` still absent, `ef_search` 64→128, 12 fabricated doc numbers retracted, mutation gate made able to fail. **3 defects in the agents' own work found on review** — see `WAVE1_IMPLEMENTATION.md`. |
| Wave 2 implementation | 4 more teams died with `502 ENOTFOUND` — a **network fault at the return stage, not an execution failure**; the filesystem showed Team A complete and Team B 1-of-5 with a *broken* fix. Took over directly. `LICENSE` + PEP 639 landed, 5 core bugs fixed, mutation 33%→100%. **Second consecutive wave containing a guard that cannot fail** (a `len()` shape check on the wrong side of `len()`). |
| **Wave 3 — FTS leg settled** | **Two published lexical artifacts declared UNREPRODUCIBLE.** `db.py:96-103` (`_MIGRATION_5`) dropped the FTS triggers, so both lexical harnesses — which build corpora with raw `INSERT` into `memories` — measured an **empty index**. The `0.949` sub-linear exponent and the `209.8 ms` lexical figure are both artifacts of a MATCH that matches nothing, and the failure is silent because a zero-match query is fast. New `bench/lexical_cause.py` measures the cause: **α = 1.149**, `ORDER BY bm25` is **99.6%** of the leg, the JOIN hypothesis is **refuted**, and the root cause is `fusion.fts_match_query` OR-ing 24 tokens so the match set *is* the corpus. See `WAVE3_IMPLEMENTATION.md`. |
| Wave 3 — first guard that fires | `lexical_cause.py` **refused to publish (exit 3)** when the median match width was 0. That guard is the only reason the void was found rather than re-measured. Its second guard then caught **my own** first width-1 probe, which reported 52 instead of 1 because `"7"` also matches every row whose `i % 97` bucket is 7. **Third consecutive wave containing a check whose condition is never exercised.** |
| Wave 3 — shell failure mid-wave | The 60k band and the V6 fix measurement were **not completed**: the tool shell died and every subsequent command returned success with empty stdout, including `python --version`. Infrastructure, not execution. **The α verdict therefore rests on two corpus sizes, not three**, and no V6 speedup is claimed. |
| Wave 3 — audit assumption refuted | The security review established that an **auth layer does exist** (loopback default + startup fails closed without credentials), contradicting this audit's own working assumption. The real defects are narrower and worse: `consolidate` is a live cross-tenant read+write, and `server.py:19-36` fails **open** while the resolver it copies fails closed. |

**The lesson, recorded so it is not repeated:** long `replace_in_file` payloads fail silently on
this toolchain, and a successful tool result is not evidence that the file is complete. The only
reliable check is reading the tail back.
