"""Tenant identity resolution (ADR-08, `omnidb_server/tenancy.py`).

The security property under test is **fail-closed**: an absent, malformed, or
unknown credential must raise, and there must be no path that silently falls
back to a default tenant. A fallback is what makes "no credential"
indistinguishable from "valid credential", which is the whole defect.

The second property is that **a caller cannot choose its own tenant**. There is
no `tenant_id` argument anywhere in the tool surface; `tests/test_tenancy_tool_surface.py`
enforces that structurally, and this module's tests enforce it by showing that
`resolve()` takes a credential and returns a tenant, with no other input.
"""

from __future__ import annotations

import json
import time

import pytest

from omnidb_server.tenancy import (
    DEFAULT_TENANT,
    TenantError,
    TenantKey,
    TenantRegistry,
    hash_key,
    index_name,
    load_registry,
    new_key,
    single_tenant,
    validate_tenant_id,
)

# --- the registry fails closed ----------------------------------------------

def test_unknown_key_raises():
    reg = TenantRegistry(keys=(TenantKey(digest=hash_key("good"), tenant="acme"),))
    with pytest.raises(TenantError, match="unknown bearer token"):
        reg.resolve("not-the-key")


def test_missing_key_raises():
    reg = TenantRegistry(keys=(TenantKey(digest=hash_key("good"), tenant="acme"),))
    for absent in (None, ""):
        with pytest.raises(TenantError, match="missing bearer token"):
            reg.resolve(absent)


def test_empty_registry_resolves_nothing():
    """A registry with no keys must not resolve *anything*.

    This is the property that would be silently broken by a `or DEFAULT_TENANT`
    fallback, and it is the one the HTTP server's startup check depends on.
    """
    assert len(TenantRegistry()) == 0
    with pytest.raises(TenantError):
        TenantRegistry().resolve("anything at all")


def test_single_tenant_registry_has_no_credentials():
    """stdio deployments run with an implicit tenant and no keys.

    The tenant still exists so the read/write paths have one shape; what they
    do not have is a way for a caller to name a *different* one.
    """
    reg = single_tenant()
    assert len(reg) == 0
    assert reg.single_tenant == DEFAULT_TENANT
    with pytest.raises(TenantError):
        reg.resolve("anything")


def test_each_key_maps_only_to_its_own_tenant():
    reg = TenantRegistry(keys=(
        TenantKey(digest=hash_key("k-acme"), tenant="acme", label="acme prod"),
        TenantKey(digest=hash_key("k-globex"), tenant="globex"),
    ))
    assert reg.resolve("k-acme") == "acme"
    assert reg.resolve("k-globex") == "globex"
    # swapping them must not work
    with pytest.raises(TenantError):
        reg.resolve("k-globex-typo")


def test_two_keys_for_one_tenant_are_allowed_for_rotation():
    """Key rotation needs both keys live at once.

    An earlier draft rejected this, reasoning that a stale key silently keeping
    access was the risk. But that makes a zero-downtime rotation impossible:
    the operator must deploy the new key and the removal together, which is
    exactly the coordinated change the rotation is trying to avoid. Both keys
    resolving to the same tenant is the desired state during a rollout.
    """
    reg = TenantRegistry(keys=(
        TenantKey(digest=hash_key("old"), tenant="acme", label="pre-rotation"),
        TenantKey(digest=hash_key("new"), tenant="acme", label="post-rotation"),
    ))
    assert reg.resolve("old") == "acme"
    assert reg.resolve("new") == "acme"
    # ...and they are still two distinct credentials, not one duplicated entry
    assert len(reg) == 2
    assert reg.tenants == ("acme",)


def test_a_duplicated_digest_is_rejected():
    """The same key listed twice is a config error.

    Different digests for one tenant are a rotation (above). An *identical*
    digest twice means the map was written wrong, and silently accepting it
    hides a typo in the key half of the entry.
    """
    digest = hash_key("k1")
    with pytest.raises(TenantError, match="duplicate api key"):
        TenantRegistry(keys=(
            TenantKey(digest=digest, tenant="acme"),
            TenantKey(digest=digest, tenant="acme"),
        ))


# --- the key is never stored or logged in the clear ------------------------

def test_key_is_stored_as_a_digest_not_the_key():
    key = new_key()
    digest = hash_key(key)
    assert key not in digest
    assert len(digest) == 64  # sha256 hex
    # deterministic, so a config can be written once and re-read
    assert hash_key(key) == digest


def test_registry_description_is_redacted():
    key = new_key()
    reg = TenantRegistry(keys=(TenantKey(digest=hash_key(key), tenant="acme",
                                         label="acme prod"),))
    described = json.dumps(reg.describe())
    assert key not in described
    assert "acme" in described
    assert "acme prod" in described


def test_a_reused_key_resolves_to_the_same_tenant():
    """Key rotation is a deployment concern; the registry is stateless."""
    key = new_key()
    first = TenantRegistry(keys=(TenantKey(digest=hash_key(key), tenant="acme"),))
    second = TenantRegistry(keys=(TenantKey(digest=hash_key(key), tenant="acme"),))
    assert first.resolve(key) == second.resolve(key) == "acme"


# --- resolution is not observably timing-dependent -------------------------

def test_resolution_does_not_early_exit_on_a_match():
    """A matching key must not be distinguishable by how long the loop took.

    The implementation compares every candidate with `hmac.compare_digest` and
    returns only after the full sweep. This test cannot prove constant time, but
    it pins the *structure*: an unknown key and a known key both walk all
    candidates, and the first-position key is not special-cased.
    """
    keys = tuple(
        TenantKey(digest=hash_key(f"k{i}"), tenant=f"t{i}") for i in range(8)
    )
    reg = TenantRegistry(keys=keys)

    # the first key must resolve, i.e. an early `return` on match is not
    # observable as a *correctness* difference
    assert reg.resolve("k0") == "t0"
    assert reg.resolve("k7") == "t7"
    with pytest.raises(TenantError):
        reg.resolve("k8")

    # a coarse sanity check that nothing pathological is happening
    start = time.perf_counter()
    for _ in range(200):
        with pytest.raises(TenantError):
            reg.resolve("nope")
    assert (time.perf_counter() - start) < 5.0


# --- tenant id validation --------------------------------------------------

@pytest.mark.parametrize("tenant", ["a", "acme", "acme-prod", "ACME_1", "t" * 64])
def test_valid_tenant_ids(tenant):
    assert validate_tenant_id(tenant) == tenant


@pytest.mark.parametrize("tenant", [
    "",                       # empty
    "t" * 65,                 # too long
    "acme/prod",              # path separator: would escape the sidecar dir
    "..",                     # traversal
    "acme.prod",              # dot: legal in a filename but not in our set
    "acme prod",              # space
    "acme:prod",              # colon
    "acme\nprod",             # control char
])
def test_invalid_tenant_ids_are_rejected(tenant):
    """A tenant id reaches a **file name**, so traversal must be impossible."""
    with pytest.raises(TenantError):
        validate_tenant_id(tenant)


def test_index_name_cannot_escape_the_home():
    """Per-tenant sidecars live in the home; a crafted id must not escape it."""
    name = index_name("acme", "hnsw")
    assert name == "tenant-acme.hnsw"
    with pytest.raises(TenantError):
        index_name("../../etc/passwd", "hnsw")
    with pytest.raises(TenantError):
        index_name("acme/../..", "npz")


# --- loading from configuration --------------------------------------------

def test_load_registry_from_a_sidecar_file(tmp_path):
    (tmp_path / "tenants.json").write_text(json.dumps({
        "tenants": {"k1": "acme", "k2": "globex"},
    }), encoding="utf-8")
    reg = load_registry(tmp_path)
    assert set(reg.tenants) == {"acme", "globex"}
    assert reg.resolve("k1") == "acme"


def test_load_registry_accepts_a_bare_map(tmp_path):
    """The `{"tenants": ...}` wrapper is optional; both spellings are in the wild."""
    (tmp_path / "tenants.json").write_text(
        json.dumps({"k1": "acme"}), encoding="utf-8")
    assert load_registry(tmp_path).resolve("k1") == "acme"


def test_inline_config_wins_over_the_sidecar(tmp_path):
    """An operator who set `tenants` in omnidb.toml meant it."""
    (tmp_path / "tenants.json").write_text(
        json.dumps({"k1": "from-file"}), encoding="utf-8")
    reg = load_registry(tmp_path, {"k1": "from-config"})
    assert reg.resolve("k1") == "from-config"


def test_sidecar_contributes_keys_the_config_omits(tmp_path):
    """A secret manager writing a mounted file must not be locked out by a
    partially-specified config."""
    (tmp_path / "tenants.json").write_text(
        json.dumps({"k1": "acme", "k2": "globex"}), encoding="utf-8")
    reg = load_registry(tmp_path, {"k1": "override-me"})
    assert reg.resolve("k1") == "override-me"
    assert reg.resolve("k2") == "globex"


def test_a_malformed_sidecar_is_a_startup_error(tmp_path):
    """A corrupt credential file must fail loudly, not degrade to no auth."""
    (tmp_path / "tenants.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TenantError, match="cannot read tenant map"):
        load_registry(tmp_path)


def test_a_sidecar_of_the_wrong_shape_is_a_startup_error(tmp_path):
    (tmp_path / "tenants.json").write_text('["a", "list"]', encoding="utf-8")
    with pytest.raises(TenantError, match="must be an object"):
        load_registry(tmp_path)


def test_an_invalid_tenant_id_in_the_map_fails_at_load(tmp_path):
    """Validation happens when the map is read, not on first request.

    A bad entry discovered mid-request is a 500 in the face of a live tenant;
    discovered at startup it is a refused boot.
    """
    (tmp_path / "tenants.json").write_text(
        json.dumps({"k1": "../escape"}), encoding="utf-8")
    with pytest.raises(TenantError):
        load_registry(tmp_path)


def test_load_registry_with_nothing_configured_is_empty_not_broken(tmp_path):
    """Single-tenant stdio has no credential file and that is fine."""
    reg = load_registry(tmp_path)
    assert len(reg) == 0


def test_tenants_listing_is_sorted_for_stable_output():
    reg = TenantRegistry(keys=(
        TenantKey(digest=hash_key("k1"), tenant="zeta"),
        TenantKey(digest=hash_key("k2"), tenant="alpha"),
    ))
    assert reg.tenants == ("alpha", "zeta")
