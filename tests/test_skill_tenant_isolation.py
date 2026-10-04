"""F11 addendum - the skill bundle functions were never tenant-scoped.

skills.py carried zero tenant references. _skill_rows selected any tenant rows,
so:
  - export_skill wrote EVERY tenants skill content into one caller-controlled zip
  - list_skills enumerated every tenants skill names and row counts
  - import_skill wrote into the default tenant regardless of whose skill it was

store.remember and store.forget_many fence themselves, so the writes could not
cross into a foreign tenant; the damage was the two reads plus a mis-targeted
import. The surface is CLI-only (cli.py has no --tenant before this change), so
this is a data-separation bug on a shared home, not a remote attack path.
"""

from __future__ import annotations

import json
import zipfile

from conftest import make_store

from omnidb_server.skills import export_skill, import_skill, list_skills

MINE = "acme"
THEIRS = "globex"


def _tag(st, tenant, skill, content):
    st.remember(content, kind="semantic",
                metadata={"skill": skill}, tenant=tenant)


def _bundle_texts(path):
    with zipfile.ZipFile(path) as zf:
        return [json.loads(line)["content"]
                for line in zf.read("memories.jsonl").decode().splitlines() if line]


def test_export_takes_only_the_calling_tenants_rows(tmp_path):
    st = make_store(tmp_path)
    _tag(st, MINE, "deploy", "acme deploys on fridays")
    _tag(st, THEIRS, "deploy", "globex deploys on tuesdays")
    out = tmp_path / "deploy.zip"

    report = export_skill(st, "deploy", out, tenant=MINE)

    assert report["memories"] == 1
    texts = _bundle_texts(out)
    assert texts == ["acme deploys on fridays"]
    assert not any("globex" in t for t in texts)
    st.close()


def test_list_takes_only_the_calling_tenants_rows(tmp_path):
    st = make_store(tmp_path)
    _tag(st, MINE, "deploy", "acme deploys")
    _tag(st, THEIRS, "payroll", "globex payroll runs")
    _tag(st, THEIRS, "payroll", "globex payroll twice")

    assert list_skills(st, tenant=MINE) == [{"name": "deploy", "memories": 1}]
    assert list_skills(st, tenant=THEIRS) == [
        {"name": "payroll", "memories": 2},
    ]
    st.close()


def test_import_lands_in_the_named_tenant_not_the_default(tmp_path):
    st = make_store(tmp_path)
    src = tmp_path / "deploy.zip"
    _tag(st, MINE, "deploy", "deploys on fridays")
    export_skill(st, "deploy", src, tenant=MINE)

    import_skill(st, src, tenant=THEIRS)

    counts = {}
    for tenant, in st.conn.execute(
        "SELECT DISTINCT tenant_id FROM memories WHERE kind = 'semantic'"
    ):
        counts[tenant] = st.conn.execute(
            "SELECT count(*) FROM memories WHERE kind = 'semantic' "
            "AND tenant_id = ? AND deleted_ms IS NULL", (tenant,)
        ).fetchone()[0]
    assert counts.get(MINE) == 1
    assert counts.get(THEIRS) == 1
    assert "default" not in counts
    st.close()


def test_overwrite_does_not_touch_another_tenants_skill(tmp_path):
    """The dangerous one: overwrite soft-deletes the rows it found for that
    skill name, and it used to find every tenants rows."""
    st = make_store(tmp_path)
    _tag(st, THEIRS, "deploy", "globex deploys on tuesdays")
    src = tmp_path / "deploy.zip"
    _tag(st, MINE, "deploy", "acme deploys on fridays")
    export_skill(st, "deploy", src, tenant=MINE)

    import_skill(st, src, overwrite=True, tenant=MINE)

    survivor = st.conn.execute(
        "SELECT count(*) FROM memories WHERE tenant_id = ? AND kind = 'semantic' "
        "AND deleted_ms IS NULL", (THEIRS,)
    ).fetchone()[0]
    assert survivor == 1, "the overwrite tombstoned a foreign tenant's skill"
    st.close()
