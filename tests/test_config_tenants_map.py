"""SEC-01: a structured value in omnidb.toml must survive config loading.

`tenant-key` prints a `config_entry` telling the operator to paste an inline
`tenants = { "<api-key>" = "<tenant>" }` map into omnidb.toml, and
`tenancy.load_registry` documents that inline map as taking precedence. But
`Config.tenants` is declared `dict | None`, so `_coerce` saw `template is None`
and ran `str(raw)` on it — the map arrived as a string and `serve-http` refused
to start with "config `tenants` must be a mapping of key -> tenant", on exactly
the configuration the CLI documents. These tests pin the pass-through, and pin
that scalar coercion is unchanged.
"""

import pytest

from omnidb_server.config import Config, ConfigError, _coerce

KEY_A = "omnidb_" + "a" * 48
KEY_B = "omnidb_" + "b" * 48


def _write_toml(tmp_path, body):
    (tmp_path / "omnidb.toml").write_text(body, encoding="utf-8")


def test_inline_tenants_map_stays_a_dict(tmp_path):
    """The regression itself: the documented inline map must load as a dict."""
    _write_toml(tmp_path, f"tenants = {{ '{KEY_A}' = 'acme', '{KEY_B}' = 'globex' }}\n")
    cfg = Config.load(home=tmp_path)
    assert isinstance(cfg.tenants, dict)
    assert cfg.tenants == {KEY_A: "acme", KEY_B: "globex"}


def test_tenants_absent_still_loads_as_none(tmp_path):
    """The default is unchanged - a single-tenant stdio home has no keys."""
    _write_toml(tmp_path, "# no tenants here\n")
    assert Config.load(home=tmp_path).tenants is None


@pytest.mark.parametrize("raw,template,expected", [
    ("0.5", 0.15, 0.5),
    ("true", False, True),
    ("7", 60, 7),
    ("abc", "auto", "abc"),
])
def test_scalar_coercion_is_unchanged(raw, template, expected):
    """The structured-value pass-through must not disturb the scalar path."""
    assert _coerce(raw, template) == expected


def test_a_list_is_also_passed_through():
    assert _coerce(["a", "b"], None) == ["a", "b"]


def test_structured_value_for_a_scalar_field_is_rejected():
    """Guard the other direction: a table where a float is declared is a
    config error, not something to silently accept."""
    with pytest.raises(ConfigError):
        _coerce({"x": 1}, 0.15, "half_life_hours")