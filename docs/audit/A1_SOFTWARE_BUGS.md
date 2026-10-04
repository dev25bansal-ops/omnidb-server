# A1 · Software Bug Catalog — OmniDB `omnidb-server`

**Agent:** A1 · **Scope:** correctness defects in runtime code
**Status:** re-derived in a second pass from source (not re-typed from the compressed
`1_ISSUES_REQUIRED_FIXES.md`). All four cross-verification verdicts are honoured.

**Evidence discipline:** every bug carries `file:line` and a mechanism. Anything that could not
be cited or explained was dropped. No style nits — agent A4 owns code quality.

**Counts:** Critical 2 · High 5 · Medium 6 · Low 3 = **16**
**P0:** BUG-A1-01, BUG-A1-02 · **P1:** BUG-A1-03 … BUG-A1-07

---

## BUG-A1-01 — `import_skill(overwrite=True)` is non-atomic; a mid-import failure destroys the skill

- **Location** `skills.py:175-183` (delete at `:176`, loop `:177-183`)
- **Severity** **Critical** — ships as a user command; loss of the original skill is unrecoverable (no backup written first).
- **Priority** P0
- **Reproduction**
```python
from omnidb_server.config import Config
from omnidb_server.db import connect
from omnidb_server.embeddings import HashingEmbedder
from omnidb_server.vector_index import BruteForceIndex
from omnidb_server.store import MemoryStore
from omnidb_server import skills

cfg = Config(); db = connect(cfg)
store = MemoryStore(db, HashingEmbedder(dim=64),
                    BruteForceIndex(cfg.home/"vectors.npz", 64), cfg)
# build a 2-row skill bundle, then make the 2nd remember() raise
# (e.g. content fails validate_content). Call import_skill(store, path, overwrite=True).
# Actual: skills.py:176 forget_many() COMMITS the delete; :178 remember() #1 commits;
#         #2 raises -> 0 live rows tagged "skill". Original gone.
```
- **Expected vs Actual** — Expected: either the whole import lands atomically or the prior rows are untouched. Actual: prior rows are committed-deleted first; the re-insert loop is **one transaction per `remember()` call**, so a partial failure leaves the skill destroyed.
- **Root cause** `skills.py:176` `store.forget_many([...])` commits. The loop `:177-183` calls `store.remember(...)` per record; each `remember` is `@_serialized` and commits independently. No `store.locked()` transaction spans delete+insert, and no rollback copy is taken.
- **Effort** 4-8 h · **Dependencies** none
- **Timeline** fix before any skill import is documented as user-facing.
- **Fix sketch** Wrap delete+re-insert in a single `with store.locked():` and re-insert via a store-level `remember_many` inside one SQLite transaction, or snapshot-and-restore the deleted rows on any exception. Repo pattern: a named constant for the max bundle size + a regression test that injects a failure on record *k* and asserts the original rows are still live.

---

## BUG-A1-02 — Consolidation retires episodes even when the chunk produced no usable fact (silent data loss)

- **Location** `consolidate.py:325-343` (gate `:325`, `consumed_ids` `:333`, mark `:343`); `consumed_source_ids` extended at `:254`
- **Severity** **Critical** — N source episodes are marked `consolidated` (permanently excluded from future windows) even if zero facts survived; source text becomes unreachable through consolidation.
- **Priority** P0
- **Reproduction**
```python
# LLM stub returns 2 valid facts for a 10-episode chunk (consolidation_chunk_episodes=10),
# both of which dedup or are dropped.
report = consolidate.run_consolidation(store, hours=24.0)
# consolidate.py:254  consumed_source_ids extended for the whole successful chunk
# consolidate.py:325  `if created or deduplicated:`  <-- gates on "any", not "all"
# consolidate.py:337  dry/off path maps consumed -> 10 episode ids
# consolidate.py:343  _mark_consolidated(all 10) -> metadata["consolidated"]=True
# Actual: 10 episodes retired for 2 (or 0) stored facts.
```
- **Expected vs Actual** — Expected: only episodes whose content is durably represented by a stored fact are retired. Actual: the gate is `created or deduplicated` (any), and `consumed_ids` is per-successful-chunk (appended at `:254` before fact-level validation). The `:326-331` comment documents this as intentional ("the LLM saw each transcript") — a real, design-acknowledged accounting gap.
- **Root cause** Retirement is driven by chunk success, not by fact→source provenance. `consumed_source_ids` is widened at chunk level.
- **Effort** 4-6 h · **Dependencies** couples to agent B5's `ConsolidationOutcome`.
- **Fix sketch** Accumulate `consumed_ids` only from facts actually created or deduplicated (via each fact's `source_episodes` provenance), and retire those specific episodes. A regression test where the stub returns facts covering 2 of 10 episodes, asserting the other 8 stay unmarked.

---

## BUG-A1-03 — Read-only open substitutes an *empty* index on any load failure (calls `__init__`, not `.load`)

- **Location** `server.py:192-193`
- **Severity** **High** — a read-only `search`/`recall` on a home whose index fails to load silently returns **zero dense results** while reporting healthy; `_reconcile_index` is not suppressed read-only, so nothing repairs it.
- **Priority** P1
- **Reproduction**
```python
from omnidb_server.server import open_read_only_store
# Corrupt/omit index.hnsw so open_vector_index raises, then:
s = open_read_only_store(cfg)
# server.py:190 open_vector_index(...) raises -> :192-193 except -> BruteForceIndex(home/vectors.npz, dim)
#              BruteForceIndex.__init__ (vector_index.py:26-31) does NOT load existing file.
# Actual: dense leg returns nothing; diagnostics look healthy.
```
- **Expected vs Actual** — Expected: a load failure surfaces as an error (or a correctly-`load`ed fallback), never as a silently-empty index. Actual: `server.py:193` constructs `BruteForceIndex(...)` directly, invoking `__init__` (empty buffers) rather than `.load(path, dim)`, so any populated `vectors.npz` is ignored.
- **Root cause** The fallback calls the constructor, not the classmethod loader; read-only reconcile is skipped so the gap is never self-healed.
- **Effort** 1-2 h · **Dependencies** none · **Timeline** P1
- **Fix sketch** In the `except` branch call `BruteForceIndex.load(path, dim)` and let a genuine failure raise, or propagate the original error.

---

## BUG-A1-04 — `reembed_home` deletes its rollback copies before the identity-publish commit

- **Location** `portability.py:271-273` (`_remove_rebuild_artifacts` at `:273`), the fallible commit at `:288-292`
- **Severity** **High** — if the final `set_meta`/`commit` publishing the new embedder identity fails, the staged `.rebuild-<pid>` backups are already removed → the home is left with vectors that may not match the stored rows, with no rollback.
- **Priority** P1
- **Reproduction**
```python
# Force a failure at the identity publish (portability.py:288-294), then call:
portability.reembed_home(cfg.home, new_embedder, assume_yes=True)
# :271 store.index = fresh; :272 store._persist_index(); :273 _remove_rebuild_artifacts(backups)  <-- gone
# :288-292 publish identity + DELETE reembed_pending + commit  <-- last fallible step, no backup
# Actual: on failure, no staged artifacts remain to restore.
```
- **Expected vs Actual** — Expected: rollback copies survive until the migration is durably committed. Actual: `_remove_rebuild_artifacts` (`:273`) runs **before** the identity-publish transaction (`:288-292`).
- **Root cause** Cleanup of the safety net precedes the last fallible operation. Note `:241-243` correctly restores on the hnsw-construction failure path — the terminal publish path lacks the same guarantee.
- **Effort** 1-2 h · **Dependencies** none · **Timeline** P1
- **Fix sketch** Move `_remove_rebuild_artifacts(backup_paths)` to **after** the successful `store.conn.commit()` at `:292`. Regression test: fail the publish commit, assert the `.rebuild-` artifacts still exist.

---

## BUG-A1-05 — `HnswlibIndex.persist` publishes graph and tombstones as two non-atomic replaces

- **Location** `vector_index.py:225-234` (`_replace_with_retry` at `:229` and `:230`)
- **Severity** **High** — a crash between the two `os.replace` calls leaves graph and tombstones at mismatched generations while the store already recorded `index_dirty=0` (`store.py:289`), so the open path considers the index clean.
- **Priority** P1
- **Reproduction**
```python
# Inject a failure (e.g. PermissionError exhausting retries) on the SECOND
# _replace_with_retry (vector_index.py:230), during store._persist_index():
store.remember("x"); store.flush()
# :229 graph replace succeeds; :230 tombstone replace raises
# On a hard process kill between the two replaces: graph is new, tombstones are old,
# and a prior clean state said index_dirty=0.
```
- **Expected vs Actual** — Expected: graph + tombstones published as one consistent generation. Actual: two independent `os.replace` calls.
- **Root cause** Two-file publish with no generation counter or single-bundle manifest (matches ARCH-A6/A7: 5 markers, no generation).
- **Effort** 4-8 h · **Dependencies** the generation-number consolidation (ARCH-A6) · **Timeline** P1; natural pairing with the `index_generation` fix.
- **Fix sketch** Publish under a single generation, or bump an `index_generation` and treat a sidecar as valid only if its generation matches `meta`. Named constant `_INDEX_GENERATION` + a crash-injection test asserting a torn pair is detected and reconciled, not trusted.

---

## BUG-A1-06 — `_recover_index_state` clears `reembed_pending` after an *unverified* rebuild

- **Location** `store.py:230-236` (`rebuild_index` at `:231`, unconditional `DELETE ... 'reembed_pending'` + `commit` at `:234-235`)
- **Severity** **High** — the marker is deleted after calling `rebuild_index`, but the rebuild's success is not verified; if it silently produced an incomplete index, the marker that would have forced a retry is gone.
- **Priority** P1
- **Reproduction**
```python
# Home left with meta['reembed_pending'] set (e.g. an interrupted migration).
store = MemoryStore(conn, emb, idx, cfg)   # not read_only
# store.py:230 get_meta(reembed_pending) is not None
# :231 rebuild_index(reason=...)   -- result not validated
# :234 DELETE FROM meta WHERE key='reembed_pending'; :235 commit
# Actual: marker cleared unconditionally. If the rebuild under-rebuilt, no marker
# remains to force reconciliation on the next open.
```
- **Expected vs Actual** — Expected: cleared only after the rebuilt index is verified to match the live row set. Actual: cleared unconditionally right after the rebuild call.
- **Root cause** No post-rebuild invariant check (`indexed_ids ⊇ live` / `active_count == _live_count`) gates the marker deletion; `:230-236` trusts the call.
- **Effort** 3-5 h · **Dependencies** reconciliation logic · **Timeline** P1
- **Fix sketch** After `rebuild_index(...)`, verify (e.g. `self.index.active_count() == self._live_count()`); only then delete the marker. Regression test: stub a failing/partial rebuild, assert the marker survives.

---

## BUG-A1-07 — `langgraph.put` upsert spans three lock acquisitions → duplicate live rows

- **Location** `adapters/langgraph.py:111-115`
- **Severity** **High** — a non-atomic check-then-act can leave two live rows for the same key; `get` returns the stale one.
- **Priority** P1
- **Reproduction**
```python
# See tests/test_adapters.py for construction. Drive two put() calls for the same
# namespace/key, interleaving the existence check between them.
# Expected: single live row, second put supersedes first.
# Actual (langgraph.py:111-115): existence checked, then written in separate
# store operations/lock acquisitions -> both rows live; get() returns the older.
```
- **Expected vs Actual** — Expected: upsert is atomic under the store lock. Actual: check and write are separate acquisitions.
- **Root cause** Check-then-act split across the `store.locked()` boundary.
- **Effort** 3-5 h · **Dependencies** none · **Timeline** P1
- **Fix sketch** Perform the existence check and the write inside a single `with store.locked():`. Concurrency test asserting one live row after simultaneous puts.

---

## BUG-A1-08 — `forget_many` / `purge` build unbounded `IN (...)` lists (one limit, not two)

- **Location** `store.py:613` (and the `purge` counterpart at `:904`); contrast the bounded reconcile at `store.py:263-267`
- **Severity** **Medium** — one MCP `forget_many` call with ~33k ids exceeds the connection's variable limit and raises. **Per cross-verification this is ONE limit, not two** — `_reconcile_index` batches off the *same* limit.
- **Priority** P2
- **Reproduction**
```python
ids = list(range(1, 40000))
store.forget_many(ids)   # store.py:613 ",".join("?"*len(wanted)) -> >SQLITE_LIMIT_VARIABLE_NUMBER
# Expected: batched delete that succeeds. Actual: sqlite3.ProgrammingError.
```
- **Expected vs Actual** — Expected: chunked `IN` execution. Actual: a single unbounded placeholder list.
- **Root cause** No batching; contrast `_RECONCILE_BATCH_CAP` (`:37`) and the reconcile loop (`:265-275`).
- **Effort** 2-4 h · **Dependencies** none · **Timeline** P2
- **Fix sketch** Reuse the existing idiom: `_SQL_VAR_BATCH = max(1, min(500, conn.getlimit(SQLITE_LIMIT_VARIABLE_NUMBER) - 2))` and loop. Regression test calling `forget_many(range(40000))`.

---

## BUG-A1-09 — `doctor._recovery_check` shape check is unreachable dead code

- **Location** `doctor.py:82-94`
- **Severity** **Medium** — a JSON *object* `pending_reembed` marker reports "1 content revisions" with `ok:True`.
- **Priority** P2
- **Reproduction**
```python
# Write meta['pending_reembed'] = '{"5": "x"}' (an object), then:
doctor.run_doctor(cfg)
# Expected: the shape check flags the malformed marker. Actual: the len()-based guard
# is unreachable, so the object marker is counted as one revision and reported ok.
```
- **Expected vs Actual** — Expected: a non-list `pending_reembed` is reported. Actual: the shape guard is dead code; a dict marker passes.
- **Root cause** The guard checks a condition that cannot be false for the parsed value; no `isinstance(..., list)` check.
- **Effort** 30-60 m · **Dependencies** none · **Timeline** P2
- **Fix sketch** Replace with an `isinstance(value, list)` assertion + a test feeding a dict marker.

---

## BUG-A1-10 — `FastEmbedEmbedder` does not L2-normalize, but the hnsw path assumes unit vectors

- **Location** `embeddings.py:82-83`; `vector_index.py:130-131` documents "IP on L2-normalized vectors == cosine" and `:205` reports `1.0 - dist` as similarity
- **Severity** **Medium** — IP ranking is still monotonic per-document but cross-document scores scale by ‖v‖, so longer documents win ties; only the `semantic` extra is affected. (Cross-verified: CONFIRMED, severity low.)
- **Priority** P2
- **Reproduction**
```python
# Use FastEmbedEmbedder (or any non-normalizing embedder) with backend=hnsw.
v = emb.encode(["short", "a much much longer document " * 20])
# Expected: similarity is cosine, comparable across documents.
# Actual: ip=1-dist scales with ||v||; the longer doc outranks despite lower cosine.
```
- **Root cause** Normalization is applied by some embedders but not enforced at the index boundary.
- **Effort** 2-4 h · **Dependencies** none · **Timeline** P2
- **Fix sketch** L2-normalize at the index-mutation boundary. Test asserting every persisted vector has ‖v‖=1±ε.

---

## BUG-A1-11 — `HnswlibIndex.update()` never grows `cap` (unlike `add()`)

- **Location** `vector_index.py:175-178` vs `:180-186`
- **Severity** **Medium** — an update-heavy workload growing the index purely via `update()` hits hnswlib's capacity error.
- **Priority** P2
- **Reproduction**
```python
idx = HnswlibIndex(path, dim=384, count_hint=0)   # cap = 64
for i in range(100):
    idx.update(i, vec)          # no resize; add_items past cap
# Expected: capacity grows as add() does. Actual: RuntimeError once count exceeds cap=64.
```
- **Root cause** The capacity check exists only in `add`, not `update`.
- **Effort** 2 h · **Dependencies** none · **Timeline** P2
- **Fix sketch** Extract grow-if-needed into `_ensure_capacity(n)` called by both.

---

## BUG-A1-12 — `rebuild_index` unlinks the sidecar before constructing the replacement

- **Location** `store.py:941-942`
- **Severity** **Medium** — if the replacement constructor raises, the original sidecar is already deleted → the only vector copy is gone.
- **Priority** P2
- **Reproduction**
```python
# Force the replacement index constructor to raise (dim mismatch / hnswlib error).
# Expected: original sidecar intact on failure. Actual: store.py:941-942 unlinks first.
```
- **Root cause** Delete-before-construct ordering — the mirror of BUG-A1-04's defect, on the store side.
- **Effort** 1 h · **Dependencies** none · **Timeline** P2
- **Fix sketch** Construct into a temp path first, then swap (mirror `portability.stage_rebuild`).

---

## BUG-A1-13 — `search()` logs `len(rows)` (the over-read batch), not the returned count

- **Location** `store.py:814`
- **Severity** **Low** — query-log `result_count` is inflated by the fetch batch, so analytics are systematically wrong.
- **Priority** P3
- **Reproduction**
```python
store.search("term", k=5)   # internally over-reads a batch for FTS
# store.py:814 records len(rows) rather than the truncated/returned count.
# Expected: result_count == 5. Actual: == batch size.
```
- **Effort** 15 m · **Dependencies** none · **Timeline** P3
- **Fix sketch** Pass the post-truncation length; assert `queries.result_count == len(results)`.

---

## BUG-A1-14 — `remember_many` rollback hole (latent, currently unreachable)

- **Location** `store.py:508-517` (raise at `:514-516` inside a `try` catching only `sqlite3.Error`)
- **Severity** **Low / latent** — a `StoreError` would bypass `_commit_or_rollback`. **Not currently live**: `lastrowid is None` is impossible for an `INTEGER PRIMARY KEY`. A structural trap, not an active bug.
- **Priority** P2 (fix the pattern, not a live incident)
- **Expected vs Actual** — Expected: any failure inside the transaction rolls back. Actual: only `sqlite3.Error` is caught; a non-sqlite exception raised mid-transaction escapes without rollback.
- **Effort** 1 h · **Dependencies** none
- **Fix sketch** Route through `_commit_or_rollback` / an explicit `rollback()` in a `finally`. Regression test injecting a non-sqlite failure, asserting no partial rows.

---

## BUG-A1-15 — Untyped `int()` coercion leaks bare `ValueError` off the store surface

- **Location** `store.py:610` (`forget_many`), `:640` (`link`), `:670` (`supersede`)
- **Severity** **Low** — a non-integer id raises a raw `ValueError` instead of the `StoreError`/OMN-* taxonomy the rest of the surface uses, so MCP callers get an inconsistent error shape.
- **Priority** P2
- **Reproduction**
```python
store.forget_many(["not-an-int"])
# store.py:610 [int(m) for m in ...] -> ValueError escapes
# store.py:640/670 link/supersede DO wrap int() in try/except -> StoreError.
# Expected: StoreError(E_TYPE) consistently. Actual: bare ValueError from forget_many.
```
- **Root cause** Missing coercion guard in `forget_many` relative to `link`/`supersede`.
- **Effort** 1-2 h · **Dependencies** none · **Timeline** P2
- **Fix sketch** Wrap the coercion to match `:641-642`; a test passing a non-integer to each of the three methods.

---

## BUG-A1-16 — `recall.py` silently discards scored candidates above `max(k*4, k)` with no `truncated` flag

- **Location** `recall.py:104` (`fused_rows = fused_rows[:max(k * 4, k)]`)
- **Severity** **Low/Medium** — **this is NOT a no-op.** With `k=1` the cap is 4, so a default `k_fetch=50` silently discards ~46 scored, filtered candidates *before* rerank/scoring, and unlike `search()` it sets **no `truncated` flag**.
- **Priority** P2
- **Reproduction**
```python
# insert >= 50 semantically similar memories, no reranker
res = recall.run_recall(store, query, k=1, kind=None, k_fetch=50,
                        agent_id=None, reranker=None)
# recall.py:103 sorts the fused pool; :104 truncates to max(1*4,1)=4 BEFORE the
# recency/frequency modifiers run. ~46 scored candidates dropped, unreported.
```
- **Expected vs Actual** — Expected: the candidate window reflects the fetched pool, and any discard is reported. Actual: hard truncation to `k*4` before modifiers, unreported.
- **Root cause** Pre-modifier truncation at `:104` with a small multiplier and no truncation signal.
- **Effort** 1-2 h · **Dependencies** none · **Timeline** P2
- **Fix sketch** Name the multiplier as a constant (`_RECALL_CANDIDATE_MULTIPLIER = 4`) and either truncate after modifiers/rerank or emit a `truncated` flag.

---

## Controls that passed (genuinely well-defended — a defect-only catalog misleads)

- **Write-path ordering (ARCH-02)** — `_after_commit` (`store.py:317-325`) centralizes the dirty-flag → index-mutation → compaction → debounced-persist order, and the dirty flag is set **first** so a crash mid-index-mutation forces reconciliation. This is the strongest correctness property in the codebase.
- **`_serialized` store lock** — the store-wide `RLock` is the *right* fix for the shared-connection concurrency bug; it is reentrant (so `remember→remember_many`, `flush→checkpoint` nest) and serializes **before** any mutation.
- **Reconciliation batching** — `_reconcile_index` correctly bounds fetch/encode/remove by `min(_RECONCILE_BATCH_CAP, SQLITE_LIMIT_VARIABLE_NUMBER - 2)`; it does not materialize a whole home in one query. **This is the pattern BUG-A1-08 should copy.**
- **Crash-window durability of content revisions** — `update_memory` records the revision in `pending_reembed` **in the same transaction as the content UPDATE**, and the queue clears only after vectors are durable. A committed revision can never coexist with a stale vector. Genuinely well-designed.
- **Windows-safe atomic file publish** — `_replace_with_retry` and `_atomic_write` correctly use temp+fsync+`os.replace` and handle AV/indexer `PermissionError` on Windows. (The *ordering* between the two replaces is BUG-A1-05, not the mechanism.)
- **ZIP/text writers clean up on all paths** — use `except BaseException: unlink(tmp); raise` — temp files never leak, even on `KeyboardInterrupt`.
- **Resource-release discipline** — `reembed_home` closes the connection on every pre-ownership failure path and releases the home lock; `backup_home` closes the backup destination in `finally`.
- **Home lock taken before mutation** — `_acquire_home_lock` takes the exclusive lock **before** `db.connect` would create the home, and refuses cleanly on contention.
- **Search authorizer** — genuinely default-deny with justified carve-outs; its data-modifying-CTE hole is closed **and tested**.
- **`load_extension` fails closed** — Python's `sqlite3` disables extension loading; no bypass.
- **AUDIT-I05 three-way query budget arithmetic is correct** — a real 10,000× error was found and fixed in an earlier pass.
- **Chunk partitioning never mixes tenant scopes** — `consolidate.py` builds chunks per-agent before chunking, and a chunk never combines scopes; a failed chunk's episodes stay unmarked and retryable.
- **Trigger re-fetch is NOT a leak** — `triggered` is built only from `pat_rows`, which already applies `_SELECTIVE` + kind + agent_id, so `triggered ⊆ pat_rows` by construction. *(Recorded here so it is not re-raised — this refuted an earlier claim of mine.)*
- **Bad importer regex is non-fatal** — `_compiled_pattern` caches invalid patterns as `None` (skipped, not fatal) in a bounded cache.
- **Embedded-prompt-injection defense** — the consolidation system prompt explicitly instructs the model to treat the transcript as data, never instructions. *(A mitigation, not a control — see A3's SEC-04.)*