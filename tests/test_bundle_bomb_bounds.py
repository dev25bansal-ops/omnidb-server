"""A bundle's declared size is not a bound on what reading it allocates.

Reproduced from first principles: a 306 KB zip whose central directory declares
`file_size: 10` passes `_enforce_bundle_size_budget` (it is a metadata check),
but `zf.read()` decompresses the full 300 MB before the CRC check rejects it —
peak allocation 631 MB. The CRC check fires *after* the allocation, so it is
not a defence.

`_enforce_bundle_size_budget`'s own docstring is honest that it reads declared
sizes; the gap is that nothing bounds the *actual* read. The fix is to cap
decompressed bytes as they stream, so a forged header cannot buy unbounded
memory regardless of what it claims.

The first audit framed this as "the gate is defeated". Refined: the gate is
passed *and then* the allocation happens. The outcome (memory exhaustion of the
shared process) is the same; the mechanism is what this file pins.

Run standalone for the raw numbers: `python tests/test_bundle_bomb_bounds.py`
"""

from __future__ import annotations

import io
import struct
import zipfile

import pytest

from omnidb_server.skills import (
    MAX_BUNDLE_MEMBER_BYTES,
    _enforce_bundle_size_budget,
    _read_member,
    import_skill,
)
from omnidb_server.store import StoreError


def _forged_zip(declared_size: int, payload: bytes) -> bytes:
    """A valid zip whose headers understate the real member size."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("memories.jsonl", payload)
    raw = bytearray(buf.getvalue())
    true_size = struct.pack("<I", len(payload))
    fake_size = struct.pack("<I", declared_size)
    start = 0
    while True:
        idx = raw.find(true_size, start)
        if idx == -1:
            break
        raw[idx:idx + 4] = fake_size
        start = idx + 4
    return bytes(raw)


def _bundle(memories: bytes, declared: int | None = None) -> bytes:
    """A complete, importable bundle whose memories member is `memories`."""
    import json

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"name": "bomb", "triggers": []}))
        zf.writestr("memories.jsonl", memories)
    raw = bytearray(buf.getvalue())
    if declared is not None:
        true_size = struct.pack("<I", len(memories))
        fake_size = struct.pack("<I", declared)
        start = 0
        while True:
            idx = raw.find(true_size, start)
            if idx == -1:
                break
            raw[idx:idx + 4] = fake_size
            start = idx + 4
    return bytes(raw)


def test_the_forgery_really_does_bypass_the_metadata_gate():
    """Establish the premise: the declared-size check alone lets this through."""
    payload = b"A" * (50 * 1024 * 1024)
    forged = _forged_zip(declared_size=10, payload=payload)
    with zipfile.ZipFile(io.BytesIO(forged)) as zf:
        # does NOT raise: the header says the member is 10 bytes
        _enforce_bundle_size_budget(zf)
        assert zf.infolist()[0].file_size == 10


def test_a_forged_header_cannot_force_an_unbounded_allocation():
    """The property that matters: bounded bytes allocated, whatever the header claims.

    Measured on a 300 MB member in a 306 KB zip declaring `file_size: 10`:
    `zf.read()` peaks at ~631 MB before the CRC check rejects it, while the
    streaming read peaks under 1 MB because it refuses the bundle while the
    member is still arriving.

    The error type is deliberately not asserted. zipfile raises BadZipFile on
    the first chunk when the header size is wrong, so the read is already
    bounded by the time that happens — what this pins is the *allocation*, not
    which exception wins the race.
    """
    import tracemalloc
    import zipfile as _zip

    payload = b"A" * (300 * 1024 * 1024)  # highly compressible
    forged = _bundle(memories=payload, declared=10)

    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    try:
        with _zip.ZipFile(io.BytesIO(forged)) as zf:
            _read_member(zf, "memories.jsonl")
    except StoreError:
        pass
    except Exception:
        pass  # zipfile's own CRC rejection is an acceptable outcome too
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()

    assert peak < 8 * 1024 * 1024, (
        f"a forged header still forced {peak / 1e6:.0f} MB of allocation; the "
        "streaming cap did not fire"
    )


def test_an_honest_oversized_member_is_rejected_by_the_streaming_cap():
    """A member over the cap is refused even when its header tells the truth."""
    big = b"A" * (MAX_BUNDLE_MEMBER_BYTES + 1024)
    raw = _bundle(memories=big)
    with pytest.raises(StoreError, match="bundle"):
        _import_bytes(raw)


def _import_bytes(raw: bytes):
    """Drive import_skill on an in-memory bundle via a temp file."""
    import tempfile
    from pathlib import Path

    from conftest import make_store

    with tempfile.TemporaryDirectory() as d:
        home = Path(d) / "home"
        bundle = Path(d) / "bundle.zip"
        bundle.write_bytes(raw)
        st = make_store(home)
        try:
            return import_skill(st, bundle)
        finally:
            st.close()


def test_the_member_budget_is_still_enforced_for_honest_bundles():
    """An honestly-declared oversized bundle is still rejected."""
    big = b"A" * (MAX_BUNDLE_MEMBER_BYTES + 1024)
    raw = _bundle(memories=big)
    with pytest.raises(StoreError, match="bundle limit"):
        _import_bytes(raw)


def test_the_total_budget_is_still_enforced():
    """Many members summing over the total budget are still rejected."""
    import json

    buf = io.BytesIO()
    chunk = b"A" * (8 * 1024 * 1024)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"name": "bomb", "triggers": []}))
        zf.writestr("memories.jsonl", chunk)
        for i in range(10):
            zf.writestr(f"pad{i}.txt", chunk)
    with pytest.raises(StoreError, match="total bundle limit"):
        _import_bytes(buf.getvalue())


def test_a_normal_bundle_still_imports():
    """The control: ordinary bundles are unaffected."""
    import json

    lines = "\n".join(json.dumps({
        "content": f"note {i}", "kind": "semantic", "metadata": {},
    }) for i in range(10))
    raw = _bundle(memories=lines.encode("utf-8"))
    result = _import_bytes(raw)
    assert result["memories"] == 10


def main() -> None:
    import tracemalloc

    payload = b"A" * (300 * 1024 * 1024)
    forged = _bundle(memories=payload, declared=10)
    print(f"zip on disk: {len(forged):,} bytes | real payload {len(payload):,} | declared 10")
    with zipfile.ZipFile(io.BytesIO(forged)) as zf:
        print("ZipInfo.file_size:", [i.file_size for i in zf.infolist()])
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    try:
        with zipfile.ZipFile(io.BytesIO(forged)) as zf:
            zf.read("memories.jsonl")
        peak = tracemalloc.get_traced_memory()[1] - base
        print(f"READ SUCCEEDED, peak {peak / 1e6:,.0f} MB -> gate bypassed")
    except Exception as exc:
        peak = tracemalloc.get_traced_memory()[1] - base
        print(f"READ RAISED {type(exc).__name__}: {exc}")
        print(f"peak {peak / 1e6:,.0f} MB -> allocation happened before the CRC check")


if __name__ == "__main__":
    main()
