from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .consolidate import run_consolidation
from .server import open_owned_store, open_read_only_store


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--version" in argv:
        from . import __version__
        print(f"omnidb-server {__version__}")
        return 0
    parser = argparse.ArgumentParser(prog="omnidb-server",
                                     description="MCP-native agent memory & retrieval server over SQLite")
    parser.add_argument("--home", help="data directory (default $OMNIDB_HOME or ~/.omnidb)")
    parser.add_argument("--version", action="store_true",
                        help="print version and exit")
    # accept --home after the subcommand too; SUPPRESS keeps the subparser
    # from overwriting a top-level value with its own default
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--home", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", parents=[common], help="run the MCP stdio server")
    p_http = sub.add_parser("serve-http", parents=[common],
                            help="run a multi-tenant streamable-HTTP MCP server "
                                 "(requires tenant credentials)")
    p_http.add_argument("--host", default="127.0.0.1",
                        help="bind address (default loopback; a wildcard bind is "
                             "an explicit, auditable choice)")
    p_http.add_argument("--port", type=int, default=8765)
    sub.add_parser("tenant-key", parents=[common],
                   help="generate a tenant API key and its config entry")
    sub.add_parser("stats", parents=[common], help="print database statistics as JSON")
    sub.add_parser("tenants", parents=[common],
                   help="list every tenant in the home with live/soft-deleted counts "
                        "(single-tenant operator view; read-only)")
    sub.add_parser("compact", parents=[common],
                   help="force a full vector-index rebuild (compaction / repair)")
    p_consol = sub.add_parser("consolidate", parents=[common],
                              help="distill episodic memories into semantic ones")
    p_consol.add_argument("--hours", type=float, default=24.0)
    p_consol.add_argument("--dry-run", action="store_true",
                          help="report what would be written without storing it "
                               "(still spends LLM calls when one is configured)")
    sub.add_parser("check", parents=[common],
                   help="run integrity checks (FTS, index drift, orphan edges)")
    sub.add_parser("schema", parents=[common],
                   help="print the search surface: tables, notes, example queries")
    p_recall = sub.add_parser("recall", parents=[common],
                              help="run a hybrid recall from the shell (JSON)")
    p_recall.add_argument("query")
    p_recall.add_argument("--k", type=int, default=5)
    p_recall.add_argument("--kind", choices=("episodic", "semantic", "procedural"),
                          default=None)
    p_recall.add_argument("--agent-id", default=None)
    p_recall.add_argument("--reranker", default=None,
                          help="omnidb_server.rerank REGISTRY name")
    p_remember = sub.add_parser("remember", parents=[common],
                                help="persist one memory from the shell (JSON)")
    p_remember.add_argument("content")
    p_remember.add_argument("--kind", choices=("episodic", "semantic", "procedural"),
                            default="semantic")
    p_remember.add_argument("--agent-id", default=None)
    p_remember.add_argument("--metadata", default=None,
                            help="metadata as a JSON object string")
    p_backup = sub.add_parser("backup", parents=[common],
                              help="hot backup: consistent DB snapshot + vector "
                                   "sidecars into a timestamped directory")
    p_backup.add_argument("dest")
    p_backup.add_argument("--keep", type=int, default=3,
                          help="snapshot directories to retain (default 3)")
    p_reembed = sub.add_parser("re-embed", parents=[common],
                               help="re-embed every memory with a different "
                                    "embedder (in-place migration)")
    p_reembed.add_argument("--embedder", required=True,
                           choices=("hashing", "fastembed"),
                           help="target embedder")
    p_reembed.add_argument("--model", default=None,
                           help="fastembed model name (default BAAI/bge-small-en-v1.5)")
    p_reembed.add_argument("--yes", action="store_true",
                           help="skip the confirmation prompt")
    sub.add_parser("config", parents=[common],
                              help="print the effective configuration and where "
                                   "each value came from")
    p_qlog = sub.add_parser("query-log", parents=[common],
                            help="dump recent query-log entries as JSON")
    p_qlog.add_argument("--limit", type=int, default=20,
                        help="max rows to return (most-recent first)")
    p_purge = sub.add_parser("purge", parents=[common],
                             help="hard-delete soft-deleted memories")
    p_purge.add_argument("--vacuum", action="store_true",
                         help="reclaim disk space with VACUUM afterwards")
    _add_tenant_flag(p_purge, "only purge this tenant's soft-deleted rows "
                              "(default: the single-tenant deployment)")
    p_export = sub.add_parser("export", parents=[common],
                              help="dump live memories and edges to a JSONL file")
    p_export.add_argument("path")
    _add_tenant_flag(p_export, "only export this tenant's rows "
                               "(default: the single-tenant deployment)")
    p_export.add_argument("--since-ms", type=int, default=None,
                          help="only export memories with created_ms >= this value")
    p_export.add_argument("--agent-id", default=None,
                          help="only export memories with this agent_id")
    p_export.add_argument("--kind", choices=("episodic", "semantic", "procedural"),
                          default=None, help="only export memories of this kind")
    p_import = sub.add_parser("import", parents=[common],
                              help="restore a JSONL export into an empty home")
    p_import.add_argument("path")
    _add_tenant_flag(p_import, "restore into this tenant "
                               "(default: the single-tenant deployment)")
    sub.add_parser("doctor", parents=[common],
                   help="print an environment health report as JSON (exit 4 on failure)")
    sub.add_parser("init", parents=[common],
                   help="create a home directory with a config template and a sample memory")
    p_skill = sub.add_parser("skill", parents=[common],
                             help="export/import/list skill bundles")
    skill_sub = p_skill.add_subparsers(dest="skill_command", required=True)
    # sub-subcommands take parents=[common] too so --home can trail them
    p_skill_export = skill_sub.add_parser("export", parents=[common],
                                          help="bundle memories tagged "
                                               "metadata[\"skill\"] = NAME into a zip")
    p_skill_export.add_argument("name")
    p_skill_export.add_argument("path")
    p_skill_export.add_argument("--tenant", default=None,
                                  help="tenant to export (default: the single tenant)")
    p_skill_import = skill_sub.add_parser("import", parents=[common],
                                          help="import a skill bundle zip")
    p_skill_import.add_argument("path")
    p_skill_import.add_argument("--overwrite", action="store_true",
                                help="replace memories already tagged with this skill")
    p_skill_import.add_argument("--tenant", default=None,
                                  help="tenant to import into (default: the single tenant)")
    p_skill_list = skill_sub.add_parser("list", parents=[common],
                                   help="list imported skills with memory counts")
    p_skill_list.add_argument("--tenant", default=None,
                              help="tenant to list (default: the single tenant)")
    p_qstats = sub.add_parser("query-stats", parents=[common],
                              help="aggregate the query log over a trailing window")
    p_qstats.add_argument("--hours", type=float, default=24.0,
                          help="window size in hours (default 24)")

    args = parser.parse_args(argv)
    home = getattr(args, "home", None)

    try:
        if args.command == "serve":
            from .server import serve

            return serve(home)  # propagate the lock-conflict exit code (3)

        if args.command == "tenant-key":
            return _cmd_tenant_key(home)

        if args.command == "serve-http":
            return _cmd_serve_http(home, args.host, args.port)

        if args.command == "doctor":
            from .doctor import run_doctor
            report = run_doctor(home)
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 4

        if args.command == "init":
            return _cmd_init(home)

        if args.command == "config":
            # pure config report: no database open, redacted secret (E-15)
            from dataclasses import asdict

            cfg, sources = Config.load_detailed(home)
            values = asdict(cfg)
            values["home"] = str(cfg.home)
            if values.get("llm_api_key"):
                values["llm_api_key"] = "***set***"
            print(json.dumps({
                "values": values,
                "sources": sources,
                "redacted": ["llm_api_key"] if values.get("llm_api_key") == "***set***" else [],
            }, indent=2))
            return 0

        if args.command == "re-embed":
            # manages its own store: opens with the stored embedder, then
            # migrates identity (E-01)
            from pathlib import Path as _Path

            from .embeddings import Embedder, EmbeddingError, FastEmbedEmbedder, HashingEmbedder
            from .portability import reembed_home

            target: Embedder
            if args.embedder == "fastembed":
                try:
                    target = FastEmbedEmbedder(model_name=args.model)
                except Exception as exc:
                    raise EmbeddingError(
                        f"cannot load fastembed model "
                        f"({args.model or 'default'}; check network / HF_HOME): {exc}"
                    ) from exc
            else:
                target = HashingEmbedder()
            resolved = _Path(Config.load(home).home)
            print(json.dumps(
                reembed_home(resolved, target, assume_yes=args.yes), indent=2,
            ))
            return 0

        cfg = Config.load(home)
        # AUDIT-I02: read-only diagnostics open the home without the write
        # lock and persist nothing (WAL gives them a consistent snapshot);
        # every command that can mutate — including recall's access bump —
        # goes through the owned-stack helper and is refused BEFORE any
        # mutable initialization when another process holds the home.
        if args.command in ("query-stats", "stats", "schema", "export",
                   "query-log", "tenants"):
            store = open_read_only_store(cfg)
            try:
                if args.command == "query-stats":
                    print(json.dumps(store.query_stats(args.hours), indent=2))
                elif args.command == "stats":
                    print(json.dumps(store.stats(), indent=2))
                elif args.command == "schema":
                    print(json.dumps(store.schema(), indent=2))
                elif args.command == "export":
                    from .portability import export_jsonl

                    print(json.dumps(export_jsonl(
                        store, args.path,
                        since_ms=args.since_ms,
                        agent_id=args.agent_id,
                        kind=args.kind,
                        tenant=_tenant_flag(args),
                    ), indent=2))
                elif args.command == "query-log":
                    rows = store.conn.execute(
                        "SELECT * FROM queries ORDER BY created_ms DESC LIMIT ?",
                        (args.limit,),
                    ).fetchall()
                    print(json.dumps([dict(r) for r in rows], indent=2))
                elif args.command == "tenants":
                    print(json.dumps(store.tenants(), indent=2))
            finally:
                store.close()
            return 0
        with open_owned_store(cfg) as store:
            if args.command == "skill":
                return _cmd_skill(store, args)
            if args.command == "consolidate":
                print(json.dumps(
                    run_consolidation(store, args.hours, dry_run=args.dry_run),
                    indent=2,
                ))
            elif args.command == "compact":
                print(json.dumps(store.rebuild_index(reason="manual"), indent=2))
            elif args.command == "check":
                print(json.dumps(store.check(), indent=2))
            elif args.command == "recall":
                print(json.dumps(store.recall(
                    args.query, k=args.k, kind=args.kind,
                    agent_id=args.agent_id, reranker=args.reranker,
                ), indent=2))
            elif args.command == "remember":
                metadata = None
                if args.metadata is not None:
                    try:
                        metadata = json.loads(args.metadata)
                    except ValueError as exc:
                        from .store import E_TYPE, StoreError

                        raise StoreError(
                            f"--metadata must be a valid JSON object: {exc}",
                            E_TYPE,
                        ) from exc
                print(json.dumps(store.remember(
                    args.content, kind=args.kind,
                    agent_id=args.agent_id, metadata=metadata,
                ), indent=2))
            elif args.command == "backup":
                from pathlib import Path as _Path

                from .portability import backup_home

                print(json.dumps(
                    backup_home(store, _Path(args.dest), keep=args.keep), indent=2,
                ))
            elif args.command == "purge":
                print(json.dumps(store.purge(do_vacuum=args.vacuum,
                                              tenant=_tenant_flag(args)), indent=2))
            elif args.command == "import":
                from .portability import import_jsonl

                print(json.dumps(import_jsonl(store, args.path,
                                               tenant=_tenant_flag(args)), indent=2))
    except Exception as exc:
        from .config import ConfigError
        from .db import DbError
        from .embeddings import EmbeddingError
        from .store import StoreError
        from .vector_index import VectorIndexError
        if isinstance(exc, (StoreError, DbError, EmbeddingError,
                            VectorIndexError, ConfigError)):
            print(f"omnidb-server: error: {exc}", file=sys.stderr)
            return 2
        raise
    return 0


def _cmd_tenant_key(home: str | None) -> int:
    """Generate one tenant API key and print the config the operator pastes in.

    The plaintext key is printed exactly once. Only its digest should be
    persisted (in `tenants` or `<home>/tenants.json`); the key itself is a
    bearer credential and must not be logged.
    """
    from .tenancy import hash_key, new_key

    key = new_key()
    print(json.dumps({
        "api_key": key,
        "warning": "print this once, store only the digest",
        "config_entry": f'tenants = {{ "{key}": "<tenant-name>" }}',
        "digest": hash_key(key),
    }, indent=2))
    return 0


def _cmd_serve_http(home: str | None, host: str, port: int) -> int:
    """Serve the multi-tenant HTTP transport.

    Owns the home like `serve` does, then resolves tenant credentials and runs
    the streamable-HTTP MCP server. Refuses to start without credentials. A
    wildcard bind is permitted, because the operator has to type it — the
    default is loopback.
    """
    from .config import Config
    from .http_server import build_registry, run_http
    from .server import _acquire_home_lock, _release_home_lock, build_stack

    cfg = Config.from_env(home)
    lock_fd = _acquire_home_lock(cfg.home)
    if isinstance(lock_fd, OSError):
        print(
            f"omnidb-server: error: another process is using {cfg.home}: {lock_fd}",
            file=sys.stderr,
        )
        return 3
    try:
        store = build_stack(cfg)
    except BaseException:
        _release_home_lock(lock_fd)
        raise
    try:
        # fails closed: no credentials is a refused start, not a warning
        registry = build_registry(cfg.home, getattr(cfg, "tenants", None))
        run_http(store, registry, host=host, port=port)
    finally:
        _release_home_lock(lock_fd)
    return 0


def _cmd_init(home: str | None) -> int:
    from pathlib import Path

    from .store import StoreError

    cfg = Config.load(home)
    target = Path(cfg.home)
    # AUDIT-I02: init writes too, so it must own the home before opening it.
    with open_owned_store(cfg) as store:
        live = int(store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_ms IS NULL"
        ).fetchone()[0])
        if live > 0:
            raise StoreError(
                f"home {target} already contains {live} memories; refusing to re-init",
                "OMN-005",
            )
        # template first: a failed sample-write below leaves a re-runnable init
        toml_path = target / "omnidb.toml"
        toml_path.write_text(_TOML_TEMPLATE, encoding="utf-8")
        sample = store.remember(
            "Welcome to OmniDB — this sample memory proves your store works.",
            kind="semantic",
            metadata={"sample": True},
        )
        print(json.dumps({"home": str(target), "toml": str(toml_path),
                          "sample_memory_id": sample["id"]}, indent=2))
    return 0


_TOML_TEMPLATE = """\
# omnidb-server configuration template (written by `omnidb-server init`).
# Keys mirror omnidb_server.Config; uncomment to override the defaults.

# Data directory. Defaults to $OMNIDB_HOME or ~/.omnidb.
# home = "~/.omnidb"

# Embedder backend: auto | hashing | fastembed.
# "auto" and "hashing" are offline-safe; "fastembed" downloads a model on
# first use (install the `semantic` extra).
# embedder = "auto"

# Record MCP/recall queries into the `queries` table (see `query-stats`).
# query_log = false

# Recency half-life for scoring, in hours.
# half_life_hours = 168.0
"""


def _add_tenant_flag(parser, help_text: str) -> None:
    """Give a subcommand the optional `--tenant` operator surface.

    Audit C13: `purge`, `export` and `import` all take a `tenant` in the store
    layer, but the CLI exposed no way to pass one, so an operator's commands
    acted on `default` and nothing else — there was no way to reclaim disk for,
    or export and restore, a named tenant. `skill --tenant` existed but only
    carries skill-tagged rows, which is not a backup.

    The default is `default` rather than required, because every existing
    single-tenant invocation must keep working unchanged.
    """
    parser.add_argument("--tenant", default=None, help=help_text)


def _tenant_flag(args) -> str:
    """Resolve a subcommand's `--tenant`, validating it.

    argv is exactly as untrusted as the store's own callers (audit C3), and the
    store validates too. Checking here as well means the operator gets the
    actionable message ("tenant id has characters outside ...") at the command
    line rather than a traceback from deeper in.
    """
    from .tenancy import DEFAULT_TENANT, validate_tenant_id

    return validate_tenant_id(args.tenant) if args.tenant else DEFAULT_TENANT




def _cmd_skill(store, args) -> int:
    from .skills import export_skill, import_skill, list_skills

    if args.skill_command == "export":
        print(json.dumps(export_skill(store, args.name, args.path,
                                  tenant=_tenant_flag(args)), indent=2))
    elif args.skill_command == "import":
        print(json.dumps(import_skill(store, args.path, overwrite=args.overwrite,
                                  tenant=_tenant_flag(args)), indent=2))
    elif args.skill_command == "list":
        print(json.dumps(list_skills(store, tenant=_tenant_flag(args)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
