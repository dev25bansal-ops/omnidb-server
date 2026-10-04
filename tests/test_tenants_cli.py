"""CLI wiring test: `omnidb-server tenants` prints the operator tenant roster."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import make_store

from omnidb_server.cli import main


def test_tenants_cli(tmp_path: Path, capsys):
    """`main([...,"tenants"])` exits 0 and emits valid JSON on stdout."""
    # populate via the writable path, then close so the read-only opener is free
    store = make_store(tmp_path)
    try:
        store.remember("acme fact", tenant="acme")
        store.remember("globex fact", tenant="globex")
        gone = store.remember("acme to forget", tenant="acme")["id"]
        store.forget(gone, tenant="acme")
    finally:
        store.close()

    exit_code = main(["--home", str(tmp_path), "tenants"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    by_tenant = {t["tenant"]: t for t in payload}
    assert set(by_tenant) == {"acme", "globex"}
    assert by_tenant["acme"]["live"] == 1
    assert by_tenant["acme"]["soft_deleted"] == 1
    assert by_tenant["globex"]["live"] == 1
    assert by_tenant["globex"]["soft_deleted"] == 0
