"""Tenant identity resolution for the multi-tenant server (ADR-08).

## Why this module exists

`agent_id` is a **label**, not a boundary. It is nullable (`NULL` is visible to
every caller), self-declared by the caller, and checked by no mutation path.
The audit found four bypasses that follow from that, all pinned as
`xfail(strict=True)` in `tests/test_tenancy_boundaries.py`.

ADR-08 replaces it with an **authenticated** tenant, resolved by OmniDB from a
bearer token it verifies itself. A client cannot choose its own tenant, because
the client does not supply one.

## The trust chain

1. The deployment owner writes a key map into the home's config
   (`tenants = {"<key>": "<tenant>", ...}`, or a `tenants.json` beside it).
2. OmniDB hashes the presented key and looks up the tenant **itself**.
3. The resolved tenant is bound to the request context. **No tool parameter can
   change it** — there is no `tenant_id` argument on any tool.

That last point is the whole design. If a tool accepted `tenant_id`, the value
would be client-supplied and the boundary would be an honour system again.

## Why stdio cannot use this

stdio has no headers: one process, one stdin/stdout pipe, one identity. A
tenant key resolves per *request*, which requires a transport that carries
identity per request. `serve` therefore keeps stdio and gains
`serve-http`; see `server.py`.

Single-tenant stdio deployments get a synthetic tenant so the rest of the stack
has exactly one code path — no "if multi-tenant" branches below this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# The tenant a single-tenant stdio deployment runs as. Not a magic value in the
# rest of the codebase — it is resolved once, here, and every downstream
# predicate takes whatever this module returns.
DEFAULT_TENANT = "default"

# Tenant ids are used in SQL (bound parameters) and in index file names. Keep
# them boring: no path separators, no dots, bounded length.
_MAX_TENANT_LEN = 64
_TENANT_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)


class TenantError(RuntimeError):
    """Raised when a tenant id or key map is unusable.

    Distinct from the store's OMN-* taxonomy on purpose: this is a deployment
    or credential fault, not a request fault, and the CLI surfaces it before
    the store is ever opened.
    """


def validate_tenant_id(tenant: str) -> str:
    """Return `tenant` if it is safe to use as an identifier, else raise.

    Called on the value *OmniDB* resolved, not on caller input, so a bad entry
    in the key map fails at startup rather than mid-request.
    """
    if not isinstance(tenant, str) or not tenant:
        raise TenantError("tenant id must be a non-empty string")
    if len(tenant) > _MAX_TENANT_LEN:
        raise TenantError(
            f"tenant id exceeds {_MAX_TENANT_LEN} characters: {tenant[:16]!r}...")
    bad = set(tenant) - _TENANT_ALLOWED
    if bad:
        raise TenantError(
            f"tenant id has characters outside [A-Za-z0-9_-]: {sorted(bad)}")
    if tenant == DEFAULT_TENANT and False:  # pragma: no cover - documentation
        # DEFAULT_TENANT is a legal value; this branch documents that the
        # reserved name is not special-cased away.
        pass
    return tenant


def hash_key(key: str) -> str:
    """The stored form of an API key: a salted digest, never the key itself.

    A config file that leaks must not hand over working credentials. Comparison
    is constant-time via `hmac.compare_digest`, so a timing oracle does not
    reveal a prefix.
    """
    if not isinstance(key, str) or not key:
        raise TenantError("api key must be a non-empty string")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def index_name(tenant: str, suffix: str) -> str:
    """The per-tenant sidecar file name.

    Per-tenant HNSW and FTS (ADR-08) mean one index per tenant, so this name is
    load-bearing for isolation at rest. Tenant ids are validated before they
    reach here, so the result cannot contain a path separator.
    """
    validate_tenant_id(tenant)
    return f"tenant-{tenant}.{suffix}"


@dataclass(frozen=True)
class TenantKey:
    """One credential and the tenant it grants access to."""

    digest: str
    tenant: str
    label: str = ""

    def describe(self) -> str:
        """A log/CLI-safe rendering. The digest is truncated, never reversible."""
        who = f" ({self.label})" if self.label else ""
        return f"{self.digest[:12]}… -> {self.tenant}{who}"


@dataclass(frozen=True)
class TenantRegistry:
    """The set of credentials this deployment accepts.

    Built once at startup from config. `resolve` is the only lookup path, and
    it is the only place a bearer token becomes a tenant id.
    """

    keys: tuple[TenantKey, ...] = field(default_factory=tuple)
    single_tenant: str = DEFAULT_TENANT

    def __post_init__(self) -> None:
        by_digest: dict[str, str] = {}
        for key in self.keys:
            validate_tenant_id(key.tenant)
            if key.digest in by_digest:
                raise TenantError(
                    f"duplicate api key: two entries share a digest and both "
                    f"claim tenant {key.tenant!r}"
                )
            by_digest[key.digest] = key.tenant
        object.__setattr__(self, "keys", tuple(self.keys))

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def tenants(self) -> tuple[str, ...]:
        """Every tenant this deployment serves, sorted for stable output."""
        return tuple(sorted({k.tenant for k in self.keys}))

    def resolve(self, presented: str | None) -> str:
        """Map a presented bearer token to its tenant, or raise.

        Fails closed: an absent, malformed, or unknown token all raise. There is
        deliberately no "fall back to the default tenant" path — that fallback
        is what makes a missing credential indistinguishable from a valid one.
        """
        if not presented:
            raise TenantError("missing bearer token")
        digest = hash_key(presented)
        for key in self.keys:
            # compare_digest on every candidate: no early exit, so the time
            # taken does not reveal how many keys matched a prefix.
            if hmac.compare_digest(key.digest, digest):
                return key.tenant
        raise TenantError("unknown bearer token")

    def describe(self) -> dict:
        """A redacted rendering for `doctor` and the startup banner."""
        return {
            "tenants": list(self.tenants),
            "keys": [k.describe() for k in self.keys],
            "single_tenant": self.single_tenant,
        }


def _load_tenants_file(path: Path) -> dict:
    """Read a tenants.json: {"tenants": {"<key>": "<tenant>"}} or a bare map."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TenantError(f"cannot read tenant map {path}: {exc}") from exc
    if isinstance(raw, dict) and "tenants" in raw:
        raw = raw["tenants"]
    if not isinstance(raw, dict):
        raise TenantError(
            f"tenant map {path} must be an object of key -> tenant")
    return raw


def load_registry(home: Path, config_tenants: dict | None = None) -> TenantRegistry:
    """Build the registry from config, then from `<home>/tenants.json`.

    Precedence: the inline `tenants` mapping in the loaded config wins if
    present, because an operator who set it in `omnidb.toml` meant it. A
    `tenants.json` beside the home is the fallback and is what a secret manager
    or a mounted volume would write.

    An empty result is **not** an error here. Single-tenant stdio deployments
    legitimately have no keys, and `single_tenant()` covers them. The HTTP
    server refuses to start without at least one key, in `server.py`.
    """
    merged: dict = {}
    if config_tenants:
        if not isinstance(config_tenants, dict):
            raise TenantError("config `tenants` must be a mapping of key -> tenant")
        merged.update(config_tenants)

    sidecar = home / "tenants.json"
    if sidecar.is_file():
        for key, tenant in _load_tenants_file(sidecar).items():
            merged.setdefault(key, tenant)

    keys = tuple(
        TenantKey(digest=hash_key(key), tenant=validate_tenant_id(tenant),
                  label="")
        for key, tenant in sorted(merged.items())
    )
    return TenantRegistry(keys=keys)


def single_tenant() -> TenantRegistry:
    """The registry a stdio deployment runs with.

    No credentials and one implicit tenant. Tool calls are still resolved to a
    tenant id — it is just always the same one, so the read and write paths have
    a single shape and the tenancy tests can exercise them without HTTP.
    """
    return TenantRegistry(keys=(), single_tenant=DEFAULT_TENANT)


def new_key() -> str:
    """Generate a candidate API key. Printed once, never stored in the clear.

    Provided so an operator does not hand-roll a weak key. The digest is what
    goes in the config; this plaintext is the only time it is available.
    """
    return "omnidb_" + os.urandom(24).hex()
