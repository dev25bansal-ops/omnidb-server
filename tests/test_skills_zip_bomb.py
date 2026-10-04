"""SEC-A3-06: a skill bundle is size-checked before anything is decompressed.

The gate reads ZipInfo.file_size out of the zip headers, so the test builds a
member whose HEADER DECLARES 2 GiB while its payload is one byte. Nothing is
ever inflated: the rejection happens on metadata alone, which is exactly the
property being tested, and the file on disk stays a few hundred bytes.
"""

import io
import json
import struct
import zipfile

import conftest
import pytest

from omnidb_server.skills import (
    MAX_BUNDLE_MEMBER_BYTES,
    MAX_BUNDLE_TOTAL_BYTES,
    export_skill,
    import_skill,
)
from omnidb_server.store import E_CONSTRAINT, StoreError

_DECLARED_2GIB = 2 * 1024 * 1024 * 1024


def _declared_size_zip(members: dict) -> bytes:
    """Build a real, openable zip whose members carry DECLARED sizes.

    `members` maps name -> (payload, declared_uncompressed_size). zipfile
    recomputes file_size on writestr, so the declared value is patched
    directly into both the local header (uncompressed size at +22) and the
    central-directory record (+24) -- the latter is what ZipInfo reads back.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, (payload, _declared) in members.items():
            zf.writestr(zipfile.ZipInfo(name), payload)
    data = bytearray(buf.getvalue())

    # Walk the central directory sequentially: each record's length depends on
    # the previous one's name/extra/comment, so find the first and step on.
    offset = data.find(b"PK\x01\x02")
    while offset != -1 and bytes(data[offset:offset + 4]) == b"PK\x01\x02":
        name_len = int.from_bytes(data[offset + 28:offset + 30], "little")
        extra_len = int.from_bytes(data[offset + 30:offset + 32], "little")
        comment_len = int.from_bytes(data[offset + 32:offset + 34], "little")
        name = bytes(data[offset + 46:offset + 46 + name_len]).decode()
        struct.pack_into("<I", data, offset + 24, members[name][1])
        local = int.from_bytes(data[offset + 42:offset + 46], "little")
        struct.pack_into("<I", data, local + 22, members[name][1])
        offset = offset + 46 + name_len + extra_len + comment_len
    return bytes(data)


def test_declared_2gib_member_is_rejected_without_decompressing(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        # a sentinel that must survive the rejected import untouched
        sentinel_id = st.remember("sentinel row survives a rejected bundle")["id"]

        manifest = json.dumps({"name": "bomb", "version": "1"}).encode()
        payload = _declared_size_zip({
            "manifest.json": (manifest, len(manifest)),
            "memories.jsonl": (b"x", _DECLARED_2GIB),
        })
        bundle = tmp_path / "bomb.zip"
        bundle.write_bytes(payload)
        # the file itself is tiny: this asserts the DECLARED size, not content
        assert bundle.stat().st_size < 4096
        with zipfile.ZipFile(bundle) as zf:
            assert max(i.file_size for i in zf.infolist()) == _DECLARED_2GIB

        with pytest.raises(StoreError) as exc:
            import_skill(st, bundle)
        assert exc.value.code == E_CONSTRAINT
        assert "memories.jsonl" in str(exc.value)

        # the store is untouched: the sentinel is still recallable
        assert st.get(sentinel_id) is not None
        assert st.recall("sentinel row survives", k=3)
        assert st.stats()["live"] == 1
    finally:
        st.close()


def test_per_member_budget_is_enforced(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        sentinel_id = st.remember("sentinel row for the per-member budget")["id"]
        manifest = json.dumps({"name": "big", "version": "1"}).encode()
        bundle = tmp_path / "big.zip"
        bundle.write_bytes(_declared_size_zip({
            "manifest.json": (manifest, len(manifest)),
            # just over the per-member cap, but under the aggregate cap
            "memories.jsonl": (b"x", MAX_BUNDLE_MEMBER_BYTES + 1),
        }))
        with pytest.raises(StoreError) as exc:
            import_skill(st, bundle)
        assert exc.value.code == E_CONSTRAINT
        assert "per-member" in str(exc.value)
        assert st.get(sentinel_id) is not None
    finally:
        st.close()


def test_aggregate_budget_catches_many_small_members(tmp_path):
    """Padding with members we never read must not bypass the total budget."""
    st = conftest.make_store(tmp_path)
    try:
        sentinel_id = st.remember("sentinel row for the aggregate budget")["id"]
        manifest = json.dumps({"name": "padded", "version": "1"}).encode()
        quarter = MAX_BUNDLE_TOTAL_BYTES // 4
        members = {
            "manifest.json": (manifest, len(manifest)),
            "memories.jsonl": (b"x", quarter),
        }
        # four declared-quarter members push the sum over the aggregate cap
        for i in range(3):
            members[f"pad{i}.bin"] = (b"y", quarter)
        bundle = tmp_path / "padded.zip"
        bundle.write_bytes(_declared_size_zip(members))
        with pytest.raises(StoreError) as exc:
            import_skill(st, bundle)
        assert exc.value.code == E_CONSTRAINT
        assert "total bundle limit" in str(exc.value)
        assert st.get(sentinel_id) is not None
    finally:
        st.close()


def test_ordinary_bundle_still_imports(tmp_path):
    """The gate must not break the happy path."""
    st = conftest.make_store(tmp_path)
    try:
        st.remember("deploy checklist: build, test, tag", kind="procedural",
                    metadata={"skill": "release"}, trigger_patterns=["deploy"])
        bundle = tmp_path / "release.zip"
        export_skill(st, "release", bundle)
        with zipfile.ZipFile(bundle) as zf:
            for info in zf.infolist():
                assert info.file_size <= MAX_BUNDLE_MEMBER_BYTES

        other = conftest.make_store(tmp_path / "copy")
        try:
            result = import_skill(other, bundle)
            assert result["name"] == "release"
            assert result["memories"] == 1
            assert other.recall("deploy checklist", k=3)
        finally:
            other.close()
    finally:
        st.close()
