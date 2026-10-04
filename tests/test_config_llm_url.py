"""SEC-A3-05: llm_base_url must be https, except for loopback hosts.

consolidate.py POSTs the Authorization header and the entire episode
transcript to {llm_base_url}/chat/completions, so a plaintext scheme is a
credential + tenant-content leak to whatever host an operator (or a gateway
letting a tenant set the field) typed. Loopback stays exempt so a local or
LAN Ollama keeps working.
"""

import pytest

from omnidb_server.config import Config, ConfigError, _validate_llm_base_url


def _load_with_env(tmp_path, monkeypatch, value):
    """Load through the env layer, which is the production path into
    _validate_config (a direct Config(...) skips it)."""
    monkeypatch.setenv("OMNIDB_LLM_BASE_URL", value)
    return Config.load(home=tmp_path)


# --- rejected: plaintext or schemeless to a non-loopback host -----------------

@pytest.mark.parametrize("bad", [
    "http://api.example.com/v1",
    "http://internal-llm/v1",
    "http://10.0.0.5:8080/v1",                  # LAN host: not loopback
    "http://169.254.169.254/latest/meta-data",  # cloud metadata service
    "http://127.0.0.1@evil.com/v1",             # userinfo trick; real host is evil.com
    "http://127.0.0.1.evil.com/v1",             # prefix, not a loopback literal
    "http://localhost.evil.com/v1",
    "http://evil.com#127.0.0.1/v1",             # loopback hidden in the fragment
    "ftp://example.com/v1",
    "file:///etc/passwd",
    "ws://example.com/v1",
    "llm.local/v1",                             # no scheme: httpx would default it
    "example.com/v1",
    "//example.com/v1",                         # protocol-relative
])
def test_non_https_non_loopback_rejected(bad):
    with pytest.raises(ConfigError, match="llm_base_url"):
        _validate_llm_base_url(bad)


@pytest.mark.parametrize("bad", [
    "http://api.example.com/v1",
    "http://127.0.0.1@evil.com/v1",
    "ftp://example.com/v1",
    "example.com/v1",
])
def test_rejected_through_config_load(tmp_path, monkeypatch, bad):
    """The real path: OMNIDB_LLM_BASE_URL -> Config.load -> _validate_config."""
    monkeypatch.setenv("OMNIDB_LLM_BASE_URL", bad)
    with pytest.raises(ConfigError, match="llm_base_url"):
        Config.load(home=tmp_path)


def test_rejected_through_toml_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIDB_LLM_BASE_URL", raising=False)
    (tmp_path / "omnidb.toml").write_text('llm_base_url = "http://internal-llm/v1"\n')
    with pytest.raises(ConfigError, match="llm_base_url"):
        Config.load(home=tmp_path)


# --- allowed: https, and plaintext only on a loopback literal ------------------

@pytest.mark.parametrize("good", [
    "https://api.example.com/v1",
    "https://api.example.com:8443/v1",
    "HTTPS://api.example.com/v1",               # scheme is case-insensitive
    "https://user:pw@proxy.internal/v1",        # userinfo does not change the host
    "http://127.0.0.1:11434/v1",                # the documented Ollama default
    "http://localhost:11434/v1",
    "http://[::1]:11434/v1",
    "http://127.0.0.2:8080/v1",                 # rest of 127.0.0.0/8
])
def test_allowed_urls_pass(good):
    assert _validate_llm_base_url(good) == good


def test_none_is_allowed_and_means_stub():
    assert _validate_llm_base_url(None) is None
    assert Config(home=".", llm_base_url=None).llm_base_url is None


@pytest.mark.parametrize("good", [
    "https://api.example.com/v1",
    "http://127.0.0.1:11434/v1",
    "http://localhost:11434/v1",
])
def test_allowed_through_config_load(tmp_path, monkeypatch, good):
    monkeypatch.setenv("OMNIDB_LLM_BASE_URL", good)
    assert Config.load(home=tmp_path).llm_base_url == good


# --- the error must be actionable, and must not leak the credential -----------

def test_error_message_names_the_fix_and_the_value(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIDB_LLM_BASE_URL", "http://llm.local/v1")
    with pytest.raises(ConfigError) as exc:
        Config.load(home=tmp_path)
    message = str(exc.value)
    assert "https" in message
    assert "loopback" in message
    assert "http://llm.local/v1" in message  # names the offending value


def test_malformed_url_is_reported_not_crashed():
    # urlsplit raises ValueError on a broken IPv6 literal; the validator must
    # turn that into a ConfigError, never an unhandled ValueError.
    with pytest.raises(ConfigError, match="llm_base_url"):
        _validate_llm_base_url("http://[::1/v1")


def test_numeric_field_error_still_wins_over_url_error(tmp_path, monkeypatch):
    """The URL check runs last, so a broken number is still reported first."""
    monkeypatch.setenv("OMNIDB_LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("OMNIDB_K_FETCH", "0")
    with pytest.raises(ConfigError, match="k_fetch"):
        Config.load(home=tmp_path)
