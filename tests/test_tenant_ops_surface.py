"""`purge`, `export` and `import` need a tenant surface (audit finding C13).

The store methods all take a `tenant` keyword and are tenant-scoped:

    purge(self, do_vacuum=False, tenant="default")
    export_jsonl(..., tenant=DEFAULT_TENANT)
    import_jsonl(..., tenant=DEFAULT_TENANT)

But the CLI defines no `--tenant` for any of the three and passes none, so an
operator's commands act on `default` and nothing else. Verified before the fix:
`omnidb-server purge --vacuum` reported `{"purged": 0}` while a soft-deleted
`acme` row survived, and `export` wrote only default-tenant rows with
`filters.tenant = 'default'`.

That leaves no operator path to reclaim disk for, or export and restore, a named
tenant — on a deployment that is explicitly multi-tenant. `skill export --tenant`
already existed, but it only carries skill-tagged rows, so it is not a
substitute for a full backup.

This file pins the CLI surface. The default stays `default`, so every
single-tenant deployment keeps working unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import make_store


@pytest.fixture()
def seeded(tmp_path):
    """One acme row (live), one acme row (soft-deleted), one default row."""
    st = make_store(tmp_path)
    try:
        st.remember("acme live note", tenant="acme")
        gone = st.remember("acme deleted note", tenant="acme")["id"]
        st.forget(gone, tenant="acme")
        st.remember("default live note", tenant="default")
        st.flush()
        yield st
    finally:
        st.close()


def _cli(tmp_path, *argv):
    return subprocess.run(
        [sys.executable, "-m", "omnidb_server.cli", "--home", str(tmp_path), *argv],
        capture_output=True, text=True, timeout=180,
    )


def _json(result):
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr}"
    return json.loads(result.stdout)


def _live_tenants(tmp_path):
    """Read the home directly, so the assertion does not depend on the CLI."""
    import sqlite3

    conn = sqlite3.connect(Path(tmp_path) / "omnidb.db")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT tenant_id, content FROM memories ORDER BY id")]
    finally:
        conn.close()


class TestPurgeTenantSurface:
    def test_purge_accepts_a_tenant_flag(self, tmp_path, seeded):
        result = _cli(tmp_path, "purge", "--tenant", "acme")
        assert result.returncode == 0, result.stderr

    def test_purge_actually_removes_the_named_tenants_rows(self, tmp_path, seeded):
        """The defect: purge reported 0 purged and the acme row survived."""
        payload = _json(_cli(tmp_path, "purge", "--tenant", "acme"))
        assert payload["purged"] == 1, payload
        assert "acme deleted note" not in _live_tenants(tmp_path)

    def test_purge_leaves_other_tenants_alone(self, tmp_path, seeded):
        _json(_cli(tmp_path, "purge", "--tenant", "acme"))
        rows = _live_tenants(tmp_path)
        assert any(r["tenant_id"] == "default" for r in rows), rows

    def test_purge_without_a_flag_keeps_its_single_tenant_behaviour(self, tmp_path, seeded):
        """Defaulting to `default` is what keeps existing deployments working."""
        payload = _json(_cli(tmp_path, "purge"))
        assert payload["purged"] == 0, "no default-tenant row was soft-deleted"

    def test_purge_vacuum_still_works_with_a_tenant(self, tmp_path, seeded):
        result = _cli(tmp_path, "purge", "--tenant", "acme", "--vacuum")
        assert result.returncode == 0, result.stderr


class TestExportTenantSurface:
    def test_export_accepts_a_tenant_flag(self, tmp_path, seeded):
        out = tmp_path / "acme.jsonl"
        result = _cli(tmp_path, "export", str(out), "--tenant", "acme")
        assert result.returncode == 0, result.stderr

    def test_export_writes_the_named_tenant_rows(self, tmp_path, seeded):
        out = tmp_path / "acme.jsonl"
        payload = _json(_cli(tmp_path, "export", str(out), "--tenant", "acme"))
        assert payload["filters"]["tenant"] == "acme", payload
        assert payload["memories"] == 1, payload
        assert "acme live note" in out.read_text(encoding="utf-8")

    def test_export_of_one_tenant_excludes_another(self, tmp_path, seeded):
        out = tmp_path / "acme.jsonl"
        _json(_cli(tmp_path, "export", str(out), "--tenant", "acme"))
        assert "default live note" not in out.read_text(encoding="utf-8")

    def test_export_default_keeps_its_single_tenant_behaviour(self, tmp_path, seeded):
        out = tmp_path / "all.jsonl"
        payload = _json(_cli(tmp_path, "export", str(out)))
        assert payload["filters"]["tenant"] == "default", payload

    def test_export_composes_with_the_existing_filters(self, tmp_path, seeded):
        out = tmp_path / "acme.jsonl"
        result = _cli(tmp_path, "export", str(out), "--tenant", "acme",
                      "--kind", "semantic")
        assert result.returncode == 0, result.stderr


class TestImportTenantSurface:
    def test_import_accepts_a_tenant_flag(self, tmp_path):
        """Restoring a named tenant into an empty home keeps its ownership."""
        from conftest import make_store as ms

        source = tmp_path / "src"
        source.mkdir()
        st = ms(source)
        st.remember("acme payload", tenant="acme", metadata={"skill": "acme-bundle"})
        st.flush()
        bundle = tmp_path / "bundle.jsonl"
        _json(_cli(source, "export", str(bundle), "--tenant", "acme"))
        st.close()

        target = tmp_path / "target"
        target.mkdir()
        payload = _json(_cli(target, "import", str(bundle), "--tenant", "acme"))
        assert payload["memories"] == 1, payload
        rows = _live_tenants(target)
        assert [r["tenant_id"] for r in rows] == ["acme"], rows

    def test_import_rejects_a_traversing_tenant(self, tmp_path):
        result = _cli(tmp_path, "import", str(tmp_path / "nope.jsonl"),
                      "--tenant", "../../evil")
        assert result.returncode != 0
        assert "characters outside" in (result.stderr + result.stdout)


class TestTenantFlagValidation:
    """The flag is argv, so it validates the same way every other tenant does."""

    @pytest.mark.parametrize("argv_tail", [
        ("purge", "--tenant", "../../evil"),
        ("export", "out.jsonl", "--tenant", "../../evil"),
        ("import", "in.jsonl", "--tenant", "../../evil"),
    ])
    def test_a_traversing_tenant_is_refused(self, tmp_path, seeded, argv_tail):
        result = _cli(tmp_path, *argv_tail)
        assert result.returncode != 0, f"accepted {argv_tail}"
        assert "characters outside" in (result.stderr + result.stdout)

    @pytest.mark.parametrize("command", ["purge", "export", "import"])
    def test_the_flag_is_optional_on_every_command(self, tmp_path, command):
        """Optional matters: argparse errors on an unknown flag, which would
        break every existing single-tenant invocation."""
        help_text = _cli(tmp_path, command, "--help")
        assert help_text.returncode == 0, help_text.stderr
        assert "--tenant" in help_text.stdout