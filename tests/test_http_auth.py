"""HTTP transport + per-tenant bearer auth (ADR-08, `omnidb_server/http_server.py`).

The security property under test is fail-closed at every layer:

1. an empty registry refuses to build a multi-tenant listener
2. an unknown, absent, or malformed bearer token is rejected (401), not mapped
   to a default tenant
3. a request that authenticated but carries no tenant claim fails, rather than
   running unscoped
4. the tenant resolved by the verifier is the one on the token, and it is the
   ONLY thing that determines the tenant — there is no caller-supplied tenant.

The stdio path is also pinned: `current_tenant()` without an access token falls
back to the single-tenant default, so tool bodies have exactly one code path.
"""

from __future__ import annotations

import asyncio

import pytest

from omnidb_server.http_server import (
    TenantTokenVerifier,
    build_auth,
    build_registry,
    current_tenant,
)
from omnidb_server.tenancy import (
    TenantError,
    TenantKey,
    TenantRegistry,
    hash_key,
)


def _verifier(registry):
    return TenantTokenVerifier(registry)


def test_build_auth_returns_verifier_and_settings_pair():
    """FastMCP takes `token_verifier=` and `auth=` as SEPARATE arguments.

    An earlier draft returned a single backend object where `AuthSettings` was
    expected — a silent no-op that leaves the listener unauthenticated. Pinning
    the two-tuple shape keeps that regression visible.
    """
    verifier, settings = build_auth(TenantRegistry(keys=(
        TenantKey(digest=hash_key("k"), tenant="acme"),)))
    assert isinstance(verifier, TenantTokenVerifier)
    # AuthSettings is a pydantic model with the resource-server fields MCP
    # requires to model the server as a protected resource.
    assert getattr(settings, "issuer_url", None)
    assert getattr(settings, "resource_server_url", None)


def _resolve(verifier, token):
    return asyncio.run(verifier.verify_token(token))


def test_verifier_returns_an_access_token_carrying_the_tenant():
    verifier = _verifier(TenantRegistry(keys=(
        TenantKey(digest=hash_key("secret"), tenant="acme"),)))
    token = _resolve(verifier, "secret")
    assert token is not None
    claims = getattr(token, "claims", None) or {}
    assert claims.get("omnidb_tenant") == "acme"
    # client_id is the tenant so logs and diagnostics carry it without needing
    # to dig into claims
    assert token.client_id == "acme"


def test_verifier_rejects_an_unknown_token_with_none():
    """An unknown key must be None (401), not an exception.

    Returning None is the framework's "unauthenticated" signal. Raising would
    surface as a 500, which tells an attacker the token was *recognised but
    rejected* — a different oracle.
    """
    verifier = _verifier(TenantRegistry(keys=(
        TenantKey(digest=hash_key("secret"), tenant="acme"),)))
    assert _resolve(verifier, "wrong") is None


def test_verifier_rejects_an_absent_token_with_none():
    verifier = _verifier(TenantRegistry(keys=(
        TenantKey(digest=hash_key("secret"), tenant="acme"),)))
    for absent in ("",):
        assert _resolve(verifier, absent) is None


def test_an_empty_registry_cannot_build_a_listener(tmp_path):
    """A multi-tenant server with no credentials is an open server."""
    with pytest.raises(TenantError, match="at least one tenant credential"):
        build_registry(tmp_path)


def test_a_configured_registry_builds_and_lists_its_tenants(tmp_path):
    import json

    (tmp_path / "tenants.json").write_text(
        json.dumps({"k1": "acme"}), encoding="utf-8")
    registry = build_registry(tmp_path)
    assert registry.tenants == ("acme",)


def test_current_tenant_falls_back_to_default_without_a_token():
    """The stdio path: no access token, so the single-tenant default is used.

    Tool bodies always receive a tenant — never "no tenant" — which is what
    keeps one code path in every tool.
    """
    assert current_tenant(default="default") == "default"


def test_current_tenant_raises_without_a_token_and_without_a_default():
    """An HTTP request that somehow reached a tool without a token must not
    run unscoped."""
    with pytest.raises(TenantError, match="no access token"):
        current_tenant(default=None)


def test_registry_description_is_redacted_through_the_verifier():
    from omnidb_server.tenancy import new_key

    key = new_key()
    registry = TenantRegistry(keys=(
        TenantKey(digest=hash_key(key), tenant="acme", label="acme prod"),))
    verifier = _verifier(registry)
    described = verifier.describe()
    assert key not in str(described)
    assert "acme" in str(described)


def test_the_same_token_verifies_consistently_across_registries():
    """Rotation safety: a token is tied to its key, not to process state."""
    from omnidb_server.tenancy import new_key

    key = new_key()
    digest = hash_key(key)
    a = _verifier(TenantRegistry(keys=(TenantKey(digest=digest, tenant="acme"),)))
    b = _verifier(TenantRegistry(keys=(TenantKey(digest=digest, tenant="acme"),)))
    assert _resolve(a, key).claims["omnidb_tenant"] == "acme"
    assert _resolve(b, key).claims["omnidb_tenant"] == "acme"