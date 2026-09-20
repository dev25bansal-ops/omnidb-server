# Operations Guide

Running `omnidb-server` against a real home directory: backup, monitoring, log
handling, disaster recovery, upgrades, and capacity planning.

All CLI examples assume `omnidb-server` is on PATH and the data directory is
passed with `--home` (or `OMNIDB_HOME` is exported); the default is `~/.omnidb`.

## Data directory layout

One home directory is one copyable unit:

| File | Role |
|---|---|
| `omnidb.db` | SQLite main database (memories, FTS5, edges, meta) |
| `omnidb.db-wal` | WAL sidecar; uncommitted frames live here until checkpoint |
| `omnidb.db-shm` | WAL shared-memory index; transient |
| `vectors.npz` / `index.hnsw` | vector index sidecar (brute force / hnswlib) |
| `index.hnsw.deleted.json` | hnsw tombstone mirror |
| `.lock` | single-writer lock held by a running server |

## Backup

The DB is a directory of files, so a plain file copy is a valid backup — with
two caveats.

1. **Consistency**: stop the server first, or at minimum run
   `omnidb-server check --home D:/path/to/data` immediately before copying and
   confirm `fts_integrity` is `"ok"` and `index_drift` is `0`. A copy taken
   mid-write with live WAL frames can be torn; SQLite recovers from that on
   next open. The vector sidecar may lag the DB by at most `persist_max_s`
   (the serve maintenance loop flushes dirty vectors on that deadline), and
   startup reconciliation rebuilds any divergence from the rows.
2. **Hot physical backup**: `omnidb-server backup D:/backups --keep 3` takes a
   crash-consistent database snapshot via SQLite's online backup API and
   copies the vector sidecars alongside it into `D:/backups/snapshot-<UTC>/`.
   It acquires the home write lock, so it cannot run while a server holds the
   home — stop `serve` first (or schedule backups around it). Every snapshot
   carries `index_dirty=1`: on restore, reconciliation re-embeds vectors from
   the rows instead of trusting a sidecar copy that may predate the snapshot
   (AUDIT-I03), so a snapshot is always internally consistent and openable.
3. **Logical backups are preferred**: `omnidb-server export --home D:/data
   D:/backup/memories.jsonl` writes a validated, atomic JSONL dump of live
   memories and edges. It skips soft-deleted/superseded rows, preserves ids,
   and restores with `import` into an **empty** home. For archive purposes
   JSONL is also diffable and version-upgrade-safe.

WAL caveat: a file copy that captures `omnidb.db` but not a current
`omnidb.db-wal` silently loses recent commits. Either stop the server first
(checkpoint on close) or copy all three files (`-db`, `-wal`, `-shm`) together
and copy the vector sidecar alongside them. Copy the directory as a unit.

## Monitoring

`omnidb-server stats` and `omnidb-server check` both emit JSON; run them on a
schedule and alert on:

Locking (AUDIT-I02): `stats`, `schema`, `export`, `query-log` and `query-stats`
are read-only diagnostics — they take no lock and persist nothing, so they are
safe to schedule against a live server. Every other command (`check`, `compact`,
`purge`, `recall`, `remember`, `consolidate`, `backup`, `import`, `init`,
`re-embed`) acquires the home lock before opening and fails fast while a server
holds it.

- `index.dead_ratio > 0.5` — more than half of the hnsw index is tombstones;
  run `omnidb-server compact` (full rebuild). Compaction normally triggers
  automatically at `dead_compact_ratio = 0.2`.
- `index_drift != 0` after settle — the DB row count and the index diverge;
  a nonzero value right after a crash is expected until reconciliation, but a
  persistent nonzero value means the sidecar is stale. `compact` repairs it.
- `wal_bytes` growth trend — steady growth means long-running readers are
  holding the checkpoint back. The serve maintenance loop flushes dirty vectors
  and checkpoints PASSIVE every 1 s (AUDIT-I16); if `wal_bytes` still ratchets
  up between restarts, look for a leaked reader.
- `cold_start_ms` regression — open time is printed once per process and
  cached; set `OMNIDB_COLD_START_BUDGET_MS` to get a one-shot stderr warning
  when open exceeds budget. The fresh-home baseline is ~26 ms; budget 200 ms
  leaves headroom for a 100k-row index build.

Also worth watching: `pending_compaction: true`, `orphan_edges` (should be 0),
`fts_integrity` (should be `"ok"`), and `soft_deleted` as a purge-candidate
indicator (`purge --vacuum` reclaims the space).

`omnidb-server doctor` summarizes all of the above as a `recovery` check: it
peeks the recovery markers (`index_dirty`, `pending_reembed`,
`reembed_pending`, `compaction_due`) read-only and reports what the next open
will heal from the rows. Pending work is informational — doctor never exits
non-zero on it — but a home that never clears its markers between runs means
reconciliation keeps finding work to do.

## Logs

`serve` is an MCP stdio server: the protocol runs on stdout, so nothing
application-level goes to stdout and diagnostics (home-lock failures, cold
start budget warnings) go to stderr as plain text. Redirect both to a file or
the journal:

- systemd: run as a unit and let journald capture stderr;
  `journalctl -u omnidb-server -f` follows it.
- logrotate: `omnidb-server serve >> /var/log/omnidb/server.log 2>&1` with a
  `copytruncate` stanza (reopen-on-rotate does not exist here — the redirect
  is the shell's, not the server's).
- Windows: run under a service wrapper (NSSM, Task Scheduler) redirecting
  stdout/stderr to rotating files; there is no built-in rotation.

Do not parse stdout for operational signal — use `stats` / `check` snapshots.

## Disaster recovery

Scenario: the home directory is damaged (partial disk failure, botched rsync,
SQLite reports `database disk image is malformed`).

1. Stop the server; do not open the damaged home again with a writer.
2. Try the logical path: `omnidb-server export` from the damaged home. If the
   DB is readable enough to dump live rows, you get a clean JSONL.
3. If export fails (the common case — corruption inside `omnidb.db`), fall
   back to file copy: keep every file you can read (WAL, sidecars, any prior
   backup).
4. Recover into a fresh home: point `--home` at a new empty directory and
   `omnidb-server import D:/backup/memories.jsonl` (any surviving JSONL export,
   newest first). `import` validates the whole file before writing, so a
   truncated export cannot half-apply.
5. The vector index rebuilds from DB content on first open; if the DB is
   unrecoverable entirely, the index sidecar alone is not enough — text lives
   in the DB, so restore from the last JSONL export.

This is why the backup section says prefer export: JSONL is the only format
that survives total `omnidb.db` loss.

## Upgrades

- Schema migrations are forward-only and run automatically on first open:
  `PRAGMA user_version` tracks the applied head (currently 3) and each open
  applies pending migrations in order, with gap detection.
- Upgrade = install the new package and open the home; no manual step.
- **Downgrade is unsupported and now enforced**: a home whose schema version
  is newer than the binary supports aborts at open with upgrade guidance
  (AUDIT-I17) instead of silently operating against an unknown schema.
  Snapshot the home (or export JSONL) before upgrading if you may need to go
  back.
- The embedder identity is pinned per database, but switching embedders (e.g.
  hashing -> fastembed) no longer requires a fresh home:
  `omnidb-server re-embed --embedder fastembed --yes` migrates in place with
  atomic identity publication (AUDIT-I03).

## Capacity sizing

From `docs/perf-report.md` (Windows, hashing embedder, measured):

- Cold start ~26 ms fresh; recall p95 ~12 ms under a 20k-op soak.
- RAM: RSS 37 -> 69 MB over a 60 s / 20k-op soak — budget well under 300 MB
  for a single agent session.
- WAL stays ~2 MB under that soak (80 MB gate) thanks to the PASSIVE
  checkpoint loop (now every 1 s via the maintenance loop).

Disk per memory is roughly: content bytes + metadata JSON + one FTS5 posting +
one f32 vector (`dim * 4` bytes: ~1.5 KB at 384-d, ~3 KB at 768-d) plus the
sidecar's share. Rule of thumb at 768-d: ~4-6 KB per small memory before
deletion overhead, so 100k memories land in the 0.5-1 GB range including
index build scratch. At 100k rows prefer the hnsw index (`OMNIDB_INDEX=hnsw`):
the brute-force path still answers (p50 ~0.6 ms) but recall@10 decays and the
rebuild cost grows (~5.5 s at 384-d, 12-14 s at 768-d).

Gates target one agent session, not a fleet: one server process per home
directory, sized for interactive latency rather than QPS.
