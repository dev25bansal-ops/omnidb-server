"""Streamable-HTTP transport with per-tenant bearer auth (ADR-08).

## Why this file exists

`create_mcp(store).run()` is stdio: one process, one pipe, one identity. A
tenant boundary needs per-*request* identity, which stdio cannot carry. This
module adds the HTTP transport and wires OmniDB's own credential check into
MCP's auth middleware.

## What the boundary actually is

The credential is verified **here**, by OmniDB, against a registry built from
the deployment's own config (`omnidb_server.tenancy`). The resolved tenant is
carried on the MCP access token and read back inside tool bodies.

**No tool takes a `tenant_id` argument.** That is the property that makes this a
boundary rather than a label: a caller cannot name its own tenant, because it
never supplies one.

## Fail-closed, at three levels

1. **Startup.** No keys configured → refuse to listen. A multi-tenant server
   with no credentials is an open server, and that is never intended.
2. **Per request.** Absent, malformed, or unknown bearer token → 401. There is
   no anonymous path and no default-tenant fallback.
3. **Per tool call.** A tool that cannot resolve a tenant raises rather than
   running unscoped.

## The lock is no longer a security boundary

`server.py:_acquire_home_lock` still serialises writers — that stays. But a
network listener means one process serves N tenants, so the lock guarantees
**ordering**, never **authorisation**. The audit's SEC-03 (ReDoS) is re-graded
from "self-inflicted" to a real cross-tenant DoS for exactly this reason.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .tenancy import TenantError, TenantRegistry, load_registry

# Where a resolved tenant is stashed on the MCP access token. A private claim
# name so it cannot collide with a standard OAuth claim.
_TENANT_CLAIM = "omnidb_tenant"


def _mcp_access_token_type():
    """Import the AccessToken type lazily, with a clear error if unavailable.

    It ships with `mcp`, but importing at module scope would make stdio-only
    deployments depend on the HTTP stack being importable.
    """
    try:
        from mcp.server.auth.middleware.auth_context import AccessToken

        return AccessToken
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "the installed mcp build has no server auth support; "
            "serve-http needs it, stdio does not"
        ) from exc


class TenantTokenVerifier:
    """MCP `TokenVerifier` that resolves an OmniDB key to a tenant.

    The MCP framework hands us the raw `Authorization` header value; we do the
    lookup ourselves rather than delegating to an external introspection
    endpoint, so the credential check is auditable inside this repository.
    """

    def __init__(self, registry: TenantRegistry) -> None:
        self._registry = registry

    async def verify_token(self, token: str):
        """Return an `AccessToken` carrying the tenant, or None to reject.

        Returning None is the framework's "unauthenticated" signal and produces
        a 401. Raising would surface as a 500, which would tell an attacker
        that the token was *recognised but rejected* — a different oracle.
        """
        _AccessToken = _mcp_access_token_type()
        try:
            tenant = self._registry.resolve(token)
        except TenantError:
            return None
        # expires_at=None: OmniDB keys do not expire on their own. Rotation is
        # the lifecycle. A fabricated far-future expiry would be a lie that
        # clients might cache against.
        # scopes=[]: no OAuth scopes are granted — a key grants whole-tenant
        # access, nothing finer, and an empty list is the honest rendering of
        # that.
        return _AccessToken(
            token=token,
            client_id=tenant,
            scopes=[],
            claims={_TENANT_CLAIM: tenant},
        )

    def describe(self) -> dict:
        return self._registry.describe()


def current_tenant(default: str | None = None) -> str:
    """The tenant for the in-flight request, or `default` outside HTTP.

    stdio tool calls have no access token, so they fall back to the
    single-tenant id. That keeps one code path in every tool body: they always
    receive a tenant, never "no tenant".
    """
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
    except ImportError:  # pragma: no cover - dependency guard
        if default is None:
            raise
        return default

    token = get_access_token()
    if token is None:
        if default is None:
            raise TenantError("no access token on this request")
        return default
    tenant = (getattr(token, "claims", None) or {}).get(_TENANT_CLAIM)
    if not tenant:
        # A token that authenticated but carries no tenant claim means the
        # verifier was bypassed or a different auth backend is wired in. Either
        # way this request has no tenant and must not run unscoped.
        raise TenantError("access token carries no tenant claim")
    return tenant


def build_auth(registry: TenantRegistry) -> tuple:
    """Build the `(token_verifier, auth_settings)` pair FastMCP expects.

    `token_verifier` does the credential check; `auth` carries the resource
    server metadata. They are separate arguments — passing a backend where
    `AuthSettings` is expected is a silent no-op at runtime and would leave the
    listener unauthenticated, so the pairing is asserted by a test.

    `AuthSettings` is required even for a purely local credential check,
    because the MCP spec models a protected server as a resource server. The
    issuer URL is informational here — there is no external issuer — but a
    syntactically valid placeholder keeps clients from refusing to connect.
    """
    from mcp.server.auth.settings import AuthSettings
    from pydantic import AnyHttpUrl

    settings = AuthSettings(
        issuer_url=AnyHttpUrl("https://omnidb.invalid/local-keys"),
        resource_server_url=AnyHttpUrl("https://omnidb.invalid"),
        required_scopes=[],
    )
    return TenantTokenVerifier(registry), settings


def build_registry(home, config_tenants: dict | None = None) -> TenantRegistry:
    """Load the registry, refusing an empty one for a multi-tenant listener.

    Called only by `serve-http`. stdio uses `tenancy.single_tenant()` and is
    unaffected.
    """
    registry = load_registry(home, config_tenants)
    if len(registry) == 0:
        raise TenantError(
            "serve-http requires at least one tenant credential. Set `tenants` "
            "in omnidb.toml, write <home>/tenants.json, or run "
            "`omnidb-server tenant-key` to generate one. Refusing to start an "
            "unauthenticated multi-tenant listener."
        )
    return registry


def run_http(
    store: Any,
    registry: TenantRegistry,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve `store` over streamable HTTP with per-tenant auth.

    Binds to loopback by default. A wildcard bind is a decision the operator
    makes explicitly, and the first thing to check when this is exposed.
    """
    from mcp.server.fastmcp import FastMCP

    token_verifier, auth_settings = build_auth(registry)
    mcp = FastMCP(
        "omnidb",
        host=host,
        port=port,
        token_verifier=token_verifier,
        auth=auth_settings,
    )
    # Same defect as the stdio transport: FastMCP's version fallback is the mcp
    # SDK's version, not ours (see `create_mcp`).
    mcp._mcp_server.version = __version__

    # Reuse the exact tool surface stdio exposes. Two hand-maintained copies of
    # ten tools would drift, and a drift here is a *security* drift: a tool
    # added to one and not the other is a tool with no tenant check.
    _register_tools(mcp, store)

    print(
        f"omnidb-server: serving http://{host}:{port} home={store._home()} "
        f"tenants={len(registry.tenants)} keys={len(registry)}",
        file=__import__("sys").stderr,
        flush=True,
    )
    mcp.run(transport="streamable-http")


def _register_tools(mcp, store) -> None:
    """Register the shared tool surface.

    Imported here rather than at module scope to keep the HTTP path free of a
    circular import with `server.py`.
    """

    # current_tenant raises when the token is absent or carries no tenant claim.
    # Passing it is what makes this transport fail closed; the default resolver
    # in attach_tools is the stdio one, which has no token to check.
    from .server import attach_tools

    attach_tools(mcp, store, tenant_resolver=current_tenant)
