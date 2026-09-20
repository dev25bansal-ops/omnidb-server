# Coverage Report

> **Status (2026-09-19):** the numbers below are the historical pre-remediation
> baseline. After the Sep 2026 audit fixes the suite measures **85.3% branch
> coverage** (268 passed, 4 skipped), and CI enforces a **`--cov-fail-under=80`
> floor** with branch collection (`.github/workflows/ci.yml`, AUDIT-I14). The
> "Threshold note" at the bottom no longer reflects the shipped configuration.

Generated with `python -m pytest tests -q --cov=omnidb_server --cov-report=term` on Python 3.14.4 / Windows (pytest-cov 7.0, coverage 7.13). Suite result: **63 passed, 1 skipped** (the opt-in `OMNIDB_TEST_SEMANTIC=1` fastembed test).

## Overall

**79.5%** line coverage (1006 statements, 206 missed) against an **85% target — gap of ~5.5 points (~55 statements)**.

## Per-module highlights

| Module | Stmts | Miss | Cover | Notes |
|---|---|---|---|---|
| `fusion.py`, `scoring.py`, `__init__.py` | 26 | 0 | 100% | Fully covered incl. hypothesis property tests (`test_scoring_properties.py`) |
| `consolidate.py` | 103 | 6 | 94.2% | Only defensive branches missed |
| `config.py` | 75 | 5 | 93.3% | Fallback/env-default branches |
| `db.py` | 40 | 3 | 92.5% | WAL/pragma error paths |
| `store.py` | 424 | 32 | 92.5% | Largest module; misses are edge branches in recall/reconcile/error paths |
| `vector_index.py` | 176 | 38 | 78.4% | Biggest absolute gap (38 lines) |
| `embeddings.py` | 54 | 14 | 74.1% | Mostly the optional `FastEmbedEmbedder` class |
| `cli.py` | 48 | 48 | **0%** | Entirely unmeasured |
| `server.py` | 60 | 60 | **0%** | Entirely unmeasured |
| `__main__.py` | — | — | omitted | Intentionally excluded via `[tool.coverage.run] omit` |

## Gap analysis vs 85% target

Biggest uncovered modules, and which existing or new tests close them:

1. **`server.py` + `cli.py` (108 lines, 0%)** — measured as uncovered because they execute only in a *child process*: `test_stdio_conformance.py::test_stdio_end_to_end` spawns `-m omnidb_server serve --home ...` via subprocess and drives it with the official MCP client over stdio, so coverage of that process is not collected into this run. The behavior IS tested; the lines are invisible to in-process measurement. Options: add `coverage run --parallel-mode`/`COVERAGE_PROCESS_START` subprocess collection, or accept them as integration-tested-only.
2. **`vector_index.py` (38 missing)** — concentrated in `HnswlibIndex.save/_atomic_write/_replace_with_retry` (lines ~192–249: Windows retry-on-PermissionError loop, tmp-file cleanup) and backend-selection guards in `open_vector_index` (kind mismatch errors, corrupt-sidecar fallback). Partially closed by `tests/test_issue_fixes.py` (I-01 compaction/scheduled rebuild), `tests/test_db.py::test_hnsw_roundtrip_and_reopen`, and `tests/test_reconcile.py::test_hnsw_reconciliation` — all gated by `pytest.importorskip("hnswlib")`. New unit tests that force a `PermissionError` on `os.replace` and exercise `open_vector_index(kind=...)` mismatch/corrupt-fallback branches would recover roughly half these lines.
3. **`embeddings.py` (14 missing)** — almost entirely the `FastEmbedEmbedder` class (init, encode) plus the `make_embedder` fastembed-unavailable `EmbeddingError` path. Closed by the opt-in `tests/test_semantic.py::test_fastembed_paraphrase_recall` when run with `OMNIDB_TEST_SEMANTIC=1`; the error branch needs a one-line test stubbing an import failure.

Closing items 2 and 3 (≈52 of the remaining 55 measurable missing lines outside server/cli) would lift measured coverage to ≈84–85%; item 1 is the deciding factor for crossing 85%, since its 108 lines are covered functionally but not counted.

## CI-ready command

```sh
python -m pytest tests -q --cov=omnidb_server --cov-report=term
```

## Threshold note

Coverage thresholds are **intentionally not enforced yet** (`fail_under` is deliberately absent from `[tool.coverage.report]`). Per §6.9 report-first policy: publish the number first, enforce once the suite stabilizes at/above target.
