"""No command may print a live bearer key after the moment it is issued.

Security leads R2 and the `tenant-key` defects from the first audit, reproduced
before any change:

    $ omnidb-server config      # with tenants in omnidb.toml
    "tenants": {"omnidb_live_prod_KEY_AAAA": "acme-prod"}
    "redacted": []

`config` is documented as a "pure config report: no database open, redacted
secret", and it does redact `llm_api_key` — but `tenants` is a plain `dict` on
Config, so the whole key map goes out in the clear. Those keys authenticate
against `serve-http`, and this is exactly the output an operator pastes into an
issue or a CI log.

`tenant-key` has three separate problems, all from the same command:

1. It tells the operator to "store only the digest", but
   `TenantRegistry.resolve` hashes the *presented* value again, so a config
   entry holding a digest authenticates nothing. Following the instruction
   locks the operator out.
2. The `config_entry` it prints is `tenants = { "key": "tenant" }`, which is not
   valid TOML — `tomllib` rejects it. Pasting the line breaks `Config.load`.
3. Printing the digest next to the key makes the digest pointless as a stored
   form.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib

import pytest

KEY = "omnidb_live_prod_KEY_AAAA"
TENANT = "acme-prod"


def _cli(home, *argv):
    return subprocess.run(
        [sys.executable, "-m", "omnidb_server.cli", "--home", str(home), *argv],
        capture_output=True, text=True, timeout=180,
    )


@pytest.fixture()
def home_with_keys(tmp_path):
    """A home whose config carries a live credential."""
    (tmp_path / "omnidb.toml").write_text(
        f'tenants = {{ "{KEY}" = "{TENANT}" }}\n', encoding="utf-8")
    return tmp_path


class TestConfigRedaction:
    def test_config_does_not_print_a_bearer_key(self, home_with_keys):
        result = _cli(home_with_keys, "config")
        assert result.returncode == 0, result.stderr
        assert KEY not in result.stdout, (
            "omnidb-server config printed a live bearer key:\n" + result.stdout[:400]
        )

    def test_config_reports_that_a_key_is_set(self, home_with_keys):
        """Redaction must be visible, or an operator cannot tell a key is there."""
        payload = json.loads(_cli(home_with_keys, "config").stdout)
        assert payload["redacted"], payload
        tenants = payload["values"]["tenants"]
        rendered = json.dumps(tenants)
        assert TENANT in rendered, f"the tenant mapping itself is useful: {rendered}"

    def test_config_shows_no_key_material_at_all(self, home_with_keys):
        payload = json.loads(_cli(home_with_keys, "config").stdout)
        rendered = json.dumps(payload)
        assert KEY not in rendered
        # not even hashed: a digest in a config report is a credential-adjacent
        # value with no operational use
        from omnidb_server.tenancy import hash_key
        assert hash_key(KEY) not in rendered

    def test_config_still_redacts_the_llm_key(self, home_with_keys):
        """The existing control must not regress."""
        (home_with_keys / "omnidb.toml").write_text(
            f'llm_api_key = "sk-live-secret-value"\n'
            f'tenants = {{ "{KEY}" = "{TENANT}" }}\n', encoding="utf-8")
        result = _cli(home_with_keys, "config")
        assert "sk-live-secret-value" not in result.stdout

    def test_a_home_without_keys_is_unaffected(self, tmp_path):
        payload = json.loads(_cli(tmp_path, "config").stdout)
        assert payload["values"].get("tenants") in (None, {}, "")


class TestTenantKeyAdvice:
    """The instruction has to be followable, or operators route around it."""

    def test_generated_config_entry_is_valid_toml(self, tmp_path):
        payload = json.loads(_cli(tmp_path, "tenant-key").stdout)
        entry = payload["config_entry"]
        # Round-trip it the way Config.load would.
        parsed = tomllib.loads(entry)
        assert isinstance(parsed.get("tenants"), dict), parsed

    def test_the_generated_entry_authenticates(self, tmp_path):
        """End to end: the entry the command prints must actually work.

        This is the property that was broken. The command told the operator to
        store a digest, and `resolve` hashes the presented value again, so a
        digest entry authenticated nothing — the operator would discover this
        only when every request 401'd.
        """
        payload = json.loads(_cli(tmp_path, "tenant-key").stdout)
        key = payload["api_key"]

        from omnidb_server.config import Config
        from omnidb_server.tenancy import load_registry

        home = tmp_path / "auth"
        home.mkdir()
        (home / "omnidb.toml").write_text(payload["config_entry"], encoding="utf-8")

        cfg = Config.load(home)
        registry = load_registry(home, cfg.tenants)
        # The entry ships a placeholder tenant name; the operator replaces it.
        # What matters is that the key authenticates at all.
        resolved = registry.resolve(key)
        assert resolved == "acme", resolved
        assert "replace acme with the tenant name" in payload["note"].lower(), (
            "the operator has to be told which part of the entry to edit"
        )

    def test_the_digest_does_not_authenticate(self, tmp_path):
        """Pins *why*: a stored digest cannot be the credential.

        Documents the behaviour rather than the advice, so the warning text can
        never drift back into recommending a digest.
        """
        from omnidb_server.tenancy import TenantKey, TenantRegistry, hash_key

        registry = TenantRegistry(keys=(
            TenantKey(digest=hash_key("some-digest-text"), tenant=TENANT),))
        with pytest.raises(Exception):
            registry.resolve(hash_key("some-digest-text"))

    def test_the_key_is_printed_exactly_once(self, tmp_path):
        payload = json.loads(_cli(tmp_path, "tenant-key").stdout)
        assert payload["api_key"]
        # The digest used to be printed alongside the key, which made it
        # pointless as a stored form — and made operators store it anyway, which
        # does not authenticate. It is gone; the config entry is the credential.
        assert "digest" not in payload, (
            "printing the digest invites operators to store it, and a stored "
            "digest cannot authenticate (see the test above)"
        )
        assert payload["api_key"] in payload["config_entry"], (
            "the entry to paste must contain the key that was just issued"
        )