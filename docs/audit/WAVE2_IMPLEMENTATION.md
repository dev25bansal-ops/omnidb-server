# Implementation Wave 2 — Licensing + Core Correctness

**Status: shipped and verified end to end.** 393 passed, lint clean, mypy clean, mutation
gate passing at 100%, and the sdist actually contains LICENSE and NOTICE.

| Metric | Wave 1 start | Now |
|---|---|---|
| Tests | 292 | **393** (+101) |
| `LICENSE` | **absent** | present, in the sdist |
| `ef_search` | 64 | 128 |
| Mutation score | 33% (unenforced) | 100% (gated) |
| Failures | 2 | **0** |

## Infrastructure failure, and what it cost

All four Wave-2 subagent teams died with `502 ENOTFOUND: getaddrinfo opencode.ai`. That was
a **network fault at the return stage, not an execution failure** — the filesystem shows
substantial work landed before the 502:

- **Team A completed fully** (LICENSE, NOTICE, PEP 639 metadata, `MANIFEST.in`,
  `release.yml`, `tests/test_license_metadata.py` — 11 tests, all passing).
- **Team B completed 1 of 5** and left a *broken* fix behind.

So "the agent failed" and "the agent did nothing" are different claims, and only the
filesystem distinguishes them. **I take over from here; I did not re-run the teams.**

## The broken thing Team B left, and why it matters

Team B's `doctor.py` fix for BUG-A1-09 read:

```python
n_pending = len(_json.loads(pending_reembed))
if not isinstance(n_pending, int) or n_pending < 0:   # <-- always False
```

`len()` **always** returns a non-negative int. This is *the same mistake as the original
bug, relocated*. The shape check has to run on the **decoded value**, before `len()`:

```python
decoded = _json.loads(pending_reembed)
if not isinstance(decoded, list):        # now reachable
    return {"ok": False, "detail": "pending_reembed marker is not a JSON list; ..."}
if not all(isinstance(item, (int, str)) for item in decoded):
    return {"ok": False, "detail": "... non-id entries; ..."}
n_pending = len(decoded)
```

The agent also **deleted the module docstring's** "informational, never exits non-zero"
semantics and a docstring line, both of which I restored — silently dropping a documented
contract is its own defect.

**This is the second time in two waves that a fail-closed guard was written in a way that
cannot fail.** The lesson generalises: a check whose condition is *statically* always-true
is worse than no check, because it reads as protection.

## What I fixed, with the regression each one earned

| Fix | What changed | Test |
|---|---|---|
| **BUG-A1-09** | doctor marker shape check on the decoded value, not its length | 3 new tests in `test_doctor_truthfulness.py`: JSON object, list of non-ids, and a control that the real format still passes |
| **BUG-A1-08** | one `_id_batch()` helper; `forget_many` **and** `purge` both batch | 10 tests in `tests/test_id_batching.py`, incl. the two a naive fix gets wrong: `missing` must span all batches, and `purged` must be a **sum** |
| **BUG-A1-04** | `_remove_rebuild_artifacts` moved to **after** the identity-publish commit | covered by the existing reembed crash tests, which now pass with the corrected ordering |
| **BUG-A1-12** | `rebuild_index` builds into a scratch path, `persist()`s, then `os.replace` | existing compaction tests; a failed rebuild is now a no-op instead of destroying the only vector copy |
| **BUG-A1-11** | `HnswlibIndex._ensure_capacity()` shared by `add` and `update` | existing vector_index tests |

### Two real regressions I introduced and caught
Worth recording, because both were silent:

1. **Merging two batching policies broke an existing test.** `test_reconcile_batches_fetch`
   monkeypatches `_RECONCILE_BATCH_CAP = 2` to prove AUDIT-I13's recovery batching. My new
   `_id_batch()` read a *different* constant, so the seam stopped working and the batch
   became 6. Fixed by making the ceiling `min(_ID_BATCH_CAP, _RECONCILE_BATCH_CAP, limit −
   reserve)` — **one policy, both seams intact.**
2. **`rebuild_index` renamed a file that did not exist yet.** An index object writes nothing
   to disk until `persist()`, so `os.replace` raised `FileNotFoundError` on 4 tests. Fixed by
   persisting to the scratch path **before** publishing.

Both were caught only because the suite was run. Neither would have been caught by reading
the diff.

## Verification, verbatim

```
393 passed, 5 skipped, 4 xfailed in 62.46s
All checks passed!                                    (ruff)
Success: no issues found in 22 source files         (mypy)
mutation score (equivalent excluded): 100.0% (29/29)
MUTATION GATE PASSED
```

sdist contents: `LICENSE` **OK**, `NOTICE` **OK**, 156 files.

## Still open

- **The FTS lexical-leg contradiction is unresolved.** Two harnesses disagree by more than an
  order of magnitude — my measurement implies α ≈ 1.60 (super-linear), the Wave-1 harness
  reported α ≈ 0.95 (sub-linear). At n=4000 they nearly agree, so the divergence is
  somewhere above it. **Neither number should be used until this is settled**, and the fix
  direction (is `ORDER BY bm25(...) LIMIT` the cost?) is still unmeasured.
- **`skills.py` was modified by Team A and never reviewed.** It is in the diff and I have
  not read what changed.
- **Two scratch files** (`_probe7.py`, `_tmp_gate_review.py`) are still in the repo root from
  Wave 1; deletion needs approval and it has twice timed out.
- **`_qcr/` still holds ~950 MB** of benchmark scratch on D:.
