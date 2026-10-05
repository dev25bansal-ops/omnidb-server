from __future__ import annotations

import ipaddress
import math
import os
import tomllib
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

_TOML_NAME = "omnidb.toml"
_EMBEDDER_SELECTORS = ("auto", "hashing", "fastembed")
_VECTOR_INDEX_SELECTORS = ("auto", "hnsw", "brute")


def validate_selector(name: str, value: object, accepted: tuple[str, ...]) -> str:
    from .store import E_TYPE, StoreError

    if not isinstance(value, str) or value not in accepted:
        raise StoreError(
            f"config value {name} must be one of {', '.join(accepted)} (got {value!r})",
            E_TYPE,
        )
    return value


class ConfigError(RuntimeError):
    pass


# SEC-A3-05: the LLM base URL is the only outbound data path in the product
# (consolidate.py sends the Authorization header plus the whole episode
# transcript to it), so its scheme is a security control, not a preference.
# https is required everywhere except loopback, which is what keeps a LAN or
# local Ollama working. A literal loopback *host* is the exemption, so a
# hostname that merely starts with "127.0.0.1" or "localhost" (an SSRF-shaped
# bypass such as 127.0.0.1.evil.com) is still rejected.
_LOOPBACK_HOSTNAMES = frozenset({"localhost"})


def _is_loopback_host(hostname: str | None) -> bool:
    """True when hostname is a loopback literal, not a name that looks like one."""
    if not hostname:
        return False
    if hostname.lower() in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_llm_base_url(value: object) -> str | None:
    """Return the URL unchanged, or raise ConfigError naming the reason.

    Loopback (127.0.0.1 / ::1 / localhost, including other addresses in the
    127.0.0.0/8 and ::1 ranges) may be plaintext http; every other host must
    be https. A URL with no scheme, or with any other scheme (file://, ftp://),
    is rejected -- a bare "host/v1" would otherwise be sent to whatever
    httpx defaults to.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"config value llm_base_url must be a non-empty string (got {value!r})")
    try:
        parts = urlsplit(value.strip())
        host = parts.hostname
    except ValueError as exc:  # malformed IPv6 literal, bad port, ...
        raise ConfigError(f"config value llm_base_url is not a valid URL: {exc} (got {value!r})") from exc
    scheme = (parts.scheme or "").lower()
    if scheme == "https":
        return value
    if scheme == "http" and _is_loopback_host(host):
        return value
    hint = "https" + (" (loopback such as http://127.0.0.1:11434/v1 is allowed)" if scheme else "")
    raise ConfigError(
        f"config value llm_base_url must use https, or http for a loopback host "
        f"(got {value!r}); refusing to send the LLM credential and transcript in "
        f"cleartext. Use {hint}"
    )


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _coerce(raw, template, name="value"):
    """AUDIT-I15: accept explicit scalar forms without lossy conversion."""
    try:
        # SEC-01: a field declared `dict | None` (or any other non-scalar type)
        # has default None, so `template is None` was true for it and every
        # value was str()-ed - which turned the documented inline `tenants =
        # {...}` map into a string and made `serve-http` refuse to start on
        # the very config `tenant-key` tells the operator to write. Pass
        # structured values through untouched; there is no scalar coercion to
        # apply to them, and the consuming code validates their shape.
        if isinstance(raw, (dict, list)):
            if template is not None and not isinstance(template, (dict, list)):
                raise ValueError(f"expected a scalar, got a {type(raw).__name__}")
            return raw
        if template is None or isinstance(template, str):
            return raw if isinstance(raw, str) else str(raw)
        if isinstance(template, bool):
            token = str(raw).strip().lower()
            if token in ("1", "true", "yes", "on"):
                return True
            if token in ("0", "false", "no", "off"):
                return False
            raise ValueError("must be an explicit boolean token")
        if isinstance(template, int):
            if isinstance(raw, bool):
                raise ValueError("must be an integer, not a boolean")
            if isinstance(raw, float) and (not math.isfinite(raw) or not raw.is_integer()):
                raise ValueError("must be an exact integer")
            return int(raw)
        if isinstance(template, float):
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("must be finite")
            return value
        return raw
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"config value {name}: {exc} (got {raw!r})") from exc


def _default_of(field):
    return None if field.default is MISSING else field.default


def _read_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


@dataclass
class Config:
    home: Path
    embedder: str = _EMBEDDER_SELECTORS[0]  # auto | hashing | fastembed
    vector_index: str = _VECTOR_INDEX_SELECTORS[0]  # auto | hnsw | brute
    half_life_hours: float = 168.0
    recency_weight: float = 0.15
    freq_weight: float = 0.05
    # corpus_mode: ranking profile for corpus retrieval (BEIR-style search).
    # Measured (bench/research-results.md): the memory rerank costs 40-56%
    # nDCG@10 on corpus workloads where all docs share near-identical
    # age/access, so corpus_mode zeroes both and defers to pure fusion.
    # Agent-memory workloads should keep it off.
    corpus_mode: bool = False
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    # --- graph leg (ADR-08 / W3-3) ---------------------------------------
    # RRF weight of the graph leg, and its only off-switch: 0.0 means the leg
    # is never computed -- not "computed then discarded". There is deliberately
    # no separate `graph_enabled` flag: a boolean and a weight are two sources
    # of truth that can disagree and produce silently dead config.
    #
    # Ships at 0.0 because the weight is UNMEASURED. `dense_weight=0.7` is
    # justified by a measured nfcorpus sweep, and rrf()'s own docstring says
    # to treat weights as a per-deployment surface with no universal default. The
    # graph leg's ranking is a function of the fused seeds, so an unmeasured
    # non-zero default would compound an unmeasured assumption into every
    # result. bench/graph_weight_sweep.py is what would earn a non-zero value.
    graph_weight: float = 0.0
    # Per-hop discount. NOT 1-d: with a hard pop budget the budget should be
    # the binding constraint, and at decay=0.15 a 3-hop node carries 0.0034
    # against a 1-hop node's 0.15 — a decorative tail. 0.5 keeps a 4-hop node
    # competitive at k=5..10. graph_decay and graph_expand_budget are one
    # tuning surface: changing the budget without the decay silently changes
    # what the leg means.
    graph_decay: float = 0.5
    # Pop cap. 0 derives it from k. Flat in graph size by construction, which
    # is what makes the leg affordable where whole-graph PPR was not.
    graph_expand_budget: int = 0
    k_fetch: int = 50
    max_search_rows: int = 500
    persist_every_ops: int = 64
    persist_every_s: float = 0.25
    # PERF-01: the sidecar persist is a full O(N) dump, so the time trigger
    # additionally requires this many pending ops, and never fires more
    # often than a multiple of the last flush's own duration. The ceiling
    # bounds how long dirty vectors sit (they are reconstructible from the
    # DB, so this limits post-crash re-embed work, not durability).
    persist_min_ops: int = 8
    persist_max_s: float = 5.0
    dead_compact_ratio: float = 0.2
    # hnsw ef_search (PERF-07, PERF-A2-02). This comment used to cite the
    # project's own sweep and then ship 64 -- the code was its own
    # counter-evidence. Measured on bench/results.md (clustered synthetic
    # cosine, hnswlib M=16/efC=200, recall@10 vs exact ground truth), 64 -> 128
    # buys +0.193 recall@10 at 10k/384-d for +0.027 ms p50, and +0.117 at
    # 100k/768-d for +0.208 ms p50 (0.278 -> 0.395, a 42% relative gain).
    # Every cell in the sweep improves and none regresses, so the default is
    # now the value the benchmark supports. The cost is bounded and measured:
    # worst-case p50 is 0.770 ms (768-d/100k), i.e. the densest, most
    # expensive cell in the sweep still answers in under a millisecond.
    # The value is honored down to a floor of 16 (vector_index.py HnswlibIndex
    # clamps with max(ef_search, 16)).
    # Coupling: this default and the per-cell frontier gate in
    # bench/baseline.py must ship together. The gate floors are set from the
    # measured 768-d/100k frontier, not from a wish -- see that file's
    # _recall_floor() for the derivation, and PERF-A2-10 for why a max(...)-
    # based gate cannot fail.
    ef_search: int = 128
    # PERF-08: when recall is scoped by agent_id, the dense leg over-fetches
    # the global ANN by this factor before the agent filter drops foreign
    # rows; raise it when many agents share one home.
    agent_overfetch: float = 3.0
    # SEC-04: upper bound on one memory's content, so a runaway agent cannot
    # bloat the DB, FTS, and the embedder in a single call.
    max_content_chars: int = 100_000
    query_log: bool = False
    # PERF-05: prune the queries table down to this many newest rows
    # occasionally while logging; 0 disables pruning.
    query_retention_rows: int = 100_000

    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    # ADR-08: inline map of api-key -> tenant-id for `serve-http`. Empty means
    # single-tenant stdio; `serve-http` refuses to start with an empty map.
    # Keys stored here are the PLAINTEXT keys in omnidb.toml; prefer a
    # <home>/tenants.json written by a secret manager, and treat inline keys as
    # a dev convenience, not a production secret store.
    tenants: dict | None = None
    # SEC-01: facts the LLM reports below this confidence are dropped instead
    # of written into long-term memory; 0.0 keeps every validated fact.
    consolidation_min_confidence: float = 0.0

    ollama_autodetect: bool = False
    # E-12: consolidation splits the episode window into LLM calls of this
    # many episodes; a failed chunk retries alone on the next run.
    consolidation_chunk_episodes: int = 10

    @staticmethod
    def load(home: str | Path | None = None) -> "Config":
        """Precedence: CLI arg > environment > omnidb.toml > built-in defaults.

        omnidb.toml is looked up in the CWD and in the resolved home directory
        (the home copy wins). Field names in the file mirror the dataclass.
        """
        return Config.load_detailed(home)[0]

    @staticmethod
    def load_detailed(
        home: str | Path | None = None,
    ) -> tuple["Config", dict[str, str]]:
        """Like `load`, but also returns each field's value source
        (`"default"`, a toml path, or `env:OMNIDB_*`) for the
        `omnidb-server config` report (E-15)."""
        arg_home = Path(home) if home is not None else None
        env_home_value = _env("OMNIDB_HOME")
        env_home = Path(env_home_value) if env_home_value else None
        resolved_home = arg_home or env_home or Path.home() / ".omnidb"

        merged: dict = {}
        source: dict[str, str] = {}
        cwd_toml = Path.cwd() / _TOML_NAME
        if cwd_toml.is_file():
            merged.update(_read_toml(cwd_toml))
            for key in merged:
                source[key] = str(cwd_toml)
        # a bare "home" key in the CWD file can redirect the home directory
        # when neither a CLI arg nor the environment pinned one
        if not arg_home and not env_home and isinstance(merged.get("home"), str):
            resolved_home = Path(merged["home"])
        home_toml = resolved_home / _TOML_NAME
        if home_toml != cwd_toml and home_toml.is_file():
            home_merged = _read_toml(home_toml)
            merged.update(home_merged)
            for key in home_merged:
                source[key] = str(home_toml)

        known = {f.name for f in fields(Config)}
        unknown = set(merged) - known
        if unknown:
            raise ConfigError(f"unknown keys in {_TOML_NAME}: {sorted(unknown)}")

        values: dict = {}
        for field in fields(Config):
            if field.name == "home":
                continue
            default = _default_of(field)
            if field.name in merged:
                values[field.name] = _coerce(merged[field.name], default, field.name)
            env_value = _env("OMNIDB_" + field.name.upper())
            env_name = "OMNIDB_" + field.name.upper()
            if env_value is None and field.name == "vector_index":
                # Legacy alias: the README, docs/operations.md, and the
                # VectorIndexError recovery messages all document OMNIDB_INDEX,
                # and users with a foreign sidecar can only act on that message.
                # The field-native OMNIDB_VECTOR_INDEX wins when both are set.
                env_value = _env("OMNIDB_INDEX")
                env_name = "OMNIDB_INDEX"
            if env_value is not None:
                values[field.name] = _coerce(env_value, default, field.name)
                source[field.name] = f"env:{env_name}"
            if field.name not in values:
                source.setdefault(field.name, "default")
            elif field.name not in source:
                # from the home toml: record the actual file for the report
                source[field.name] = str(home_toml) if home_toml.is_file() else "toml"

        values["home"] = arg_home or env_home or resolved_home
        source["home"] = ("cli" if arg_home
                          else "env:OMNIDB_HOME" if env_home
                          else "default")
        cfg = Config(**values)
        _validate_config(cfg)
        return cfg, source

    # backward-compatible alias used across tests/CLI
    from_env = load


def _validate_config(cfg: "Config") -> None:
    """Fail fast on obviously-broken values (E-15): a typo like
    `persist_every_ops = 0` otherwise turns into per-write sidecar dumps
    instead of a named configuration error."""
    # AUDIT-I15: direct Config construction must obey finite-value policy too.
    for field in fields(cfg):
        if isinstance(field.default, float):
            value = getattr(cfg, field.name)
            try:
                finite = math.isfinite(value)
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ConfigError(f"config value {field.name} must be finite (got {value!r})")
    for name, accepted in (
        ("embedder", _EMBEDDER_SELECTORS),
        ("vector_index", _VECTOR_INDEX_SELECTORS),
    ):
        validate_selector(name, getattr(cfg, name), accepted)
    checks: list[tuple[str, bool, str]] = [
        ("k_fetch", cfg.k_fetch >= 1, "must be >= 1"),
        ("rrf_k", cfg.rrf_k >= 1, "must be >= 1"),
        ("graph_weight", cfg.graph_weight >= 0.0, "must be >= 0"),
        ("graph_decay", 0.0 < cfg.graph_decay <= 1.0, "must be in (0, 1]"),
        # 0 carries a DIFFERENT meaning from small non-zero, so the message
        # names both: a typo should be diagnosable from the error alone.
        ("graph_expand_budget",
         cfg.graph_expand_budget == 0 or 32 <= cfg.graph_expand_budget <= 8192,
         "must be 0 (derive from k) or in [32, 8192]"),
        ("max_search_rows", cfg.max_search_rows >= 1, "must be >= 1"),
        ("persist_every_ops", cfg.persist_every_ops >= 1, "must be >= 1"),
        ("persist_every_s", cfg.persist_every_s > 0, "must be > 0"),
        ("persist_min_ops", cfg.persist_min_ops >= 1, "must be >= 1"),
        ("persist_max_s", cfg.persist_max_s > 0, "must be > 0"),
        ("dead_compact_ratio", 0 < cfg.dead_compact_ratio <= 1,
         "must be in (0, 1]"),
        ("half_life_hours", cfg.half_life_hours > 0, "must be > 0"),
        ("agent_overfetch", cfg.agent_overfetch >= 1.0, "must be >= 1.0"),
        ("ef_search", cfg.ef_search >= 16, "must be >= 16"),
        ("max_content_chars", cfg.max_content_chars >= 1, "must be >= 1"),
        ("query_retention_rows", cfg.query_retention_rows >= 0, "must be >= 0"),
        ("consolidation_chunk_episodes", cfg.consolidation_chunk_episodes >= 1,
         "must be >= 1"),
        ("consolidation_min_confidence",
         0.0 <= cfg.consolidation_min_confidence <= 1.0, "must be in [0, 1]"),
    ]
    bad = [(name, rule) for name, ok, rule in checks if not ok]
    if bad:
        name, rule = bad[0]
        raise ConfigError(f"config value {name} {rule} (got {getattr(cfg, name)!r})")
    # The graph leg must stay a structural minority vote. Its ranking is a
    # function of the fused seeds, so its own contribution to any row is
    # bounded by graph_weight/(rrf_k+1) while the two legs it is seeded from
    # contribute up to (dense_weight + sparse_weight)/(rrf_k+1). Capping it at
    # that sum means a graph-only row can never become the dominant term for
    # any row -- the leg can expand the candidate set, not outvote its own
    # seeds. Enforced here rather than left to the default, because the default
    # is exactly the kind of thing an operator overrides.
    if cfg.graph_weight > cfg.dense_weight + cfg.sparse_weight:
        raise ConfigError(
            "config value graph_weight must be <= dense_weight + sparse_weight "
            f"({cfg.graph_weight!r} > {cfg.dense_weight + cfg.sparse_weight!r}): "
            "a graph-only row would otherwise outrank the rows it was seeded "
            "from"
        )
    # SEC-A3-05: last, so a typo in a numeric field is still reported first.
    _validate_llm_base_url(cfg.llm_base_url)
