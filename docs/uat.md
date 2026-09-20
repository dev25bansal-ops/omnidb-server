# Claude Desktop — user acceptance trail

Manual checklist for verifying the OmniDB MCP server end-to-end from Claude Desktop.
Automated coverage lives in `tests/`; this trail exercises what only a real client can:
tool discovery, restart persistence, and the conversational flow around recall.

## Prerequisites

1. Install the package into the Python that Claude Desktop will launch:

   ```bash
   pip install -e ".[hnsw,semantic]"
   ```

2. Copy [`examples/claude-desktop-config.example.json`](../examples/claude-desktop-config.example.json)
   over `%APPDATA%\Claude\claude_desktop_config.json` (merge the `omnidb` entry if you
   already have other servers). Adjust the absolute `command` path and `OMNIDB_HOME` to
   your machine. The example pins `OMNIDB_EMBEDDER=hashing` (offline-safe); to
   opt into real (fastembed) semantics change it to `fastembed` and set a local
   `HF_HOME` cache so the ~100MB model downloads once and never again.

3. **Fully quit and restart Claude Desktop** (tray icon → Quit, not just the window).

4. Confirm the server is up: the tools icon appears in the message input; opening
   it lists the `omnidb` server with tools `remember`, `recall`, `search`, `schema`,
   `related`, `update_memory`, `forget`, `forget_many`, `consolidate`, `stats`.
   If the icon's location or name has changed in a Claude Desktop update, the test
   still passes once the ten `omnidb` tools are visible — anchor on the tool list,
   not the icon.

Run each task below in a **new** conversation unless noted. Check every Pass criterion
before moving on. Ids are assigned sequentially from 1 in a fresh home directory; yours
will differ — substitute the id each step actually returns.

## Tasks

### 1. Handshake & tool listing

- Steps: Start a chat, open the tools (hammer) menu, expand `omnidb`.
- Expected: ten tools listed, each with its description (`Persist a memory…`,
  `Hybrid recall over memories…`, ...).
- Pass criteria:
  - [ ] server connects without error on launch
  - [ ] all ten tools are listed
  - [ ] tool descriptions render

### 2. remember returns an id and latency

- Steps: Ask Claude to "remember that the deploy window is Tuesday 09:00 UTC".
- Expected: one `remember` call; result JSON contains `"id": <n>` and `"took_ms"`.
- Pass criteria:
  - [ ] `remember` executes without error
  - [ ] response shows an integer id
  - [ ] response includes `took_ms`

### 3. Paraphrase recall ranks target in top-3

- Steps: In the same conversation ask "when do we ship?" (no shared keywords with the
  stored sentence beyond "deploy/ship" intent). With the hashing embedder this checks
  surface overlap; with fastembed it should match despite different wording.
- Expected: one `recall` call returning up to k results with `score` fields.
- Pass criteria:
  - [ ] the deploy-window memory appears in the returned results
  - [ ] it ranks within the top 3
  - [ ] results carry `rrf` / `recency` / `score` breakdowns
- Note: with `OMNIDB_EMBEDDER=auto` (hashing fallback) a pure paraphrase may miss —
  that is expected behavior of the offline embedder, not a failure of the trail.

### 4. Persistence across restart

- Steps: Fully quit Claude Desktop, reopen it, start a new conversation, ask "what is
  the deploy window?"
- Expected: `recall` hits the same memory from disk (`OMNIDB_HOME`).
- Pass criteria:
  - [ ] memory survives full client restart
  - [ ] content matches what was stored verbatim
  - [ ] no embedder/dimension mismatch error at startup (identity is pinned per DB)

### 5. forget hides from recall

- Steps: Ask Claude to forget the deploy-window memory using the id from task 2, then
  repeat the recall query from task 3.
- Expected: `forget` returns `{"forgotten": <id>, ...}`; the later `recall` no longer
  returns it.
- Pass criteria:
  - [ ] forget succeeds for a live id
  - [ ] post-forget recall omits the memory entirely
  - [ ] stats still counts it under `soft_deleted` (soft delete, not hard)

### 6. update_memory reindexes both directions

- Steps: Store "the API base URL is https://api.example.com/v1", then ask Claude to
  update that memory's content to point at `https://api.example.com/v2`. Recall both by
  the old phrasing ("base URL") and the new one ("/v2").
- Expected: content edit re-embeds and reindexes via the FTS trigger; both queries hit.
- Pass criteria:
  - [ ] update reports `"updated": ["content"]`
  - [ ] old-phrasing query finds the updated content
  - [ ] new-phrasing query finds the updated content
  - [ ] stale `/v1` text no longer appears anywhere in results

### 7. Link and graph walk

- Steps: Store two facts ("project Falcon uses Postgres", "Falcon's staging env is
  staging.falcon.dev"), link them when prompted (Claude may pass
  `links: [{"to": <first id>}]` on the second remember), then ask "what is related to
  the Falcon project fact?"
- Expected: `related` walks the edge undirected and returns the neighbor.
- Pass criteria:
  - [ ] second remember accepts the link to the first id
  - [ ] `related` returns the linked memory at depth 1
  - [ ] walking from either endpoint finds the other (undirected)

### 8. search enforces read-only (OMN-020)

- Steps: Ask Claude to run `search` with `DELETE FROM memories`.
- Expected: the tool result is an error carrying `[OMN-020]`; nothing is deleted.
- Pass criteria:
  - [ ] tool result surfaces as an error (isError) mentioning OMN-020
  - [ ] row count unchanged afterwards (verify via `stats` or
        `SELECT COUNT(*) FROM memories`)
  - [ ] a plain `SELECT id FROM memories LIMIT 3` still works through the same tool

### 9. stats fields present

- Steps: Ask Claude for database statistics.
- Expected: one `stats` call returning the documented payload shape.
- Pass criteria:
  - [ ] `live`, `by_kind`, `edges`, `embedder{name,dim}`, `index{kind,count}`,
        `wal_bytes`, `files` are all present
  - [ ] `by_kind` totals equal `live`
  - [ ] file sizes listed for `omnidb.db` and the vector sidecar

### 10. consolidate stub distills >= 3 episodes

- Steps: Store three or more short episodic memories (kind `episodic`), then ask Claude
  to consolidate the last 24 hours.
- Expected: `consolidate` runs the offline stub (`"llm_used": false`), creates one
  semantic copy per episode, marks episodes consolidated.
- Pass criteria:
  - [ ] report shows `episodes_scanned >= 3`
  - [ ] `facts_created == episodes_scanned` and `llm_used` is false
  - [ ] newly created semantic memories carry
        `metadata.source_episodes` referencing the episode ids
  - [ ] re-running immediately scans 0 fresh episodes (already marked)
- Note: if you configured `OMNIDB_LLM_BASE_URL` per README, expect `"llm_used": true`
  instead and distilled facts rather than verbatim copies.

## Bug report template

```
Title: <one-line summary>

Environment:
- OS / Claude Desktop version:
- omnidb-server version (pip show omnidb-server):
- OMNIDB_EMBEDDER / index kind (from `stats`):
- Home directory size (files table from `stats`):

Task #: <which numbered task>
Steps to reproduce:
1.

Observed:
<exact tool output / error text, including any OMN-* code>

Expected:
<the Pass criteria line that failed>
```
