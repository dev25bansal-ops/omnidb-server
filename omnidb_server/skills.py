from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from .store import _SELECTIVE, E_CONSTRAINT, E_NOT_FOUND, E_TYPE, StoreError
from .tenancy import DEFAULT_TENANT

_SKILL_KINDS = ("procedural", "semantic")
_MANIFEST = "manifest.json"
_MEMORIES = "memories.jsonl"
_MANIFEST_KEYS = {"name", "version", "description", "triggers"}

# SEC-A3-06: admission budgets for an imported bundle, checked against the
# ZipInfo headers BEFORE any read() so a deflate bomb is rejected without
# ever being decompressed. Named like the other module caps (validation.py's
# MAX_TRIGGER_PATTERNS, store.py's _RECONCILE_BATCH_CAP).
MAX_BUNDLE_MEMBER_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 64 * 1024 * 1024


def _enforce_bundle_size_budget(zf: zipfile.ZipFile) -> None:
    """Reject an over-large bundle from its declared sizes alone.

    ZipInfo.file_size is the uncompressed size recorded in the header, so
    this is a metadata check: a few-MB zip claiming gigabytes fails here and
    never allocates. Per-member and aggregate budgets both apply, and the
    sum covers every entry -- not just the two we read -- so padding the
    archive with extra members cannot slip past the total.
    """
    total = 0
    for info in zf.infolist():
        size = info.file_size
        if size > MAX_BUNDLE_MEMBER_BYTES:
            raise StoreError(
                f"{info.filename} declares {size} bytes, over the "
                f"{MAX_BUNDLE_MEMBER_BYTES}-byte per-member bundle limit",
                E_CONSTRAINT,
            )
        total += size
        if total > MAX_BUNDLE_TOTAL_BYTES:
            raise StoreError(
                f"bundle declares {total}+ bytes uncompressed, over the "
                f"{MAX_BUNDLE_TOTAL_BYTES}-byte total bundle limit",
                E_CONSTRAINT,
            )


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    """Read one member, refusing to buffer more than the budget allows.

    `_enforce_bundle_size_budget` reads `ZipInfo.file_size`, which is header
    metadata an attacker controls: a 306 KB zip declaring `file_size: 10` passes
    that check, and `zf.read()` then decompresses the real payload before the
    CRC check rejects it — measured at 631 MB peak for a 300 MB member. The CRC
    fires *after* the allocation, so it is not a defence.

    Streaming with a hard cap refuses the bundle while it is still arriving, so
    a forged header buys no memory. The cap is the per-member budget, which is
    the same limit the header check enforces — this does not lower it, it makes
    it real. Returns bytes so callers keep using `zf.read()`'s return shape.
    """
    limit = MAX_BUNDLE_MEMBER_BYTES
    with zf.open(name) as member:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = member.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise StoreError(
                    f"{name} expands past the {limit}-byte per-member bundle "
                    "limit; its declared size in the zip header is not trusted",
                    E_CONSTRAINT,
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _skill_rows(store, skill_name: str, tenant: str = DEFAULT_TENANT) -> list:
    """Live memory rows tagged metadata["skill"] == skill_name, ordered by id.

    The LIKE prefilter is a heuristic on the canonical json.dumps form
    ('"skill": "<name>"'); the parsed comparison below is the exact test, so
    over-matches are filtered out. Names containing quotes or LIKE wildcards
    can under-match at the SQL stage — a known limitation of the heuristic.
    """
    rows = store.conn.execute(
        f"SELECT id, kind, content, metadata, trigger_patterns FROM memories "
        f"WHERE metadata LIKE ? AND {_SELECTIVE} AND tenant_id = ? ORDER BY id",
        ('%"skill": "' + skill_name + '"%', tenant),
    ).fetchall()
    selected = []
    for row in rows:
        # metadata is a JSON column but the LIKE prefilter is a text heuristic,
        # so a selected row can hold a non-object JSON value or text that is not
        # one value at all. Skip those instead of failing the whole export.
        try:
            metadata = json.loads(row["metadata"])
        except ValueError:
            continue
        if isinstance(metadata, dict) and metadata.get("skill") == skill_name:
            selected.append(row)
    return selected


def _write_bundle(path: str | Path, manifest: dict, memory_lines: list[str]) -> None:
    """Write the zip atomically: sibling temp, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(_MANIFEST, json.dumps(manifest, indent=2))
                zf.writestr(_MEMORIES, "".join(line + "\n" for line in memory_lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def export_skill(store, skill_name: str, path: str | Path,
                 tenant: str = DEFAULT_TENANT) -> dict:
    """Bundle the live memories tagged metadata["skill"] == skill_name into a zip.

    Bundle layout: manifest.json (name/version/description/triggers) plus
    memories.jsonl (one JSON object per line). trigger_patterns are taken from
    the first matching procedural row that carries them; the bundle carries
    procedural/semantic rows only.
    """
    rows = _skill_rows(store, skill_name, tenant=tenant)
    triggers = None
    memory_lines: list[str] = []
    for row in rows:
        if row["kind"] not in _SKILL_KINDS:
            raise StoreError(
                f"memory {row['id']} has kind {row['kind']!r}; skill bundles carry "
                f"only {' and '.join(_SKILL_KINDS)} rows",
                E_TYPE,
            )
        if row["kind"] == "procedural" and triggers is None and row["trigger_patterns"] is not None:
            try:
                patterns = json.loads(row["trigger_patterns"])
            except (TypeError, ValueError):
                patterns = []
            if isinstance(patterns, list):
                triggers = patterns
        memory_lines.append(json.dumps({
            "content": row["content"],
            "kind": row["kind"],
            "metadata": json.loads(row["metadata"]),
        }))
    manifest = {
        "name": skill_name,
        "version": "1",
        "description": "",
        "triggers": triggers or [],
    }
    _write_bundle(path, manifest, memory_lines)
    return {"name": skill_name, "memories": len(memory_lines), "path": str(path)}


def _validate_manifest(manifest) -> None:
    if not isinstance(manifest, dict):
        raise StoreError("manifest.json must be a JSON object", E_TYPE)
    unknown = set(manifest) - _MANIFEST_KEYS
    if unknown:
        raise StoreError(f"unknown manifest keys: {sorted(unknown)}", E_TYPE)
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise StoreError("manifest.name must be a non-empty string", E_TYPE)
    for key in ("version", "description"):
        if key in manifest and not isinstance(manifest[key], str):
            raise StoreError(f"manifest.{key} must be a string", E_TYPE)
    triggers = manifest.get("triggers", [])
    if not isinstance(triggers, list) or not all(isinstance(t, str) for t in triggers):
        raise StoreError("manifest.triggers must be a list of regex strings", E_TYPE)
    for pattern in triggers:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise StoreError(f"manifest.triggers: invalid regex {pattern!r} ({exc})", E_TYPE) from exc


def _parse_memory_lines(text: str) -> list[dict]:
    records: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except ValueError as exc:
            raise StoreError(f"memories.jsonl line {lineno}: invalid JSON ({exc})", E_TYPE) from exc
        if not isinstance(record, dict) or not isinstance(record.get("content"), str):
            raise StoreError(
                f"memories.jsonl line {lineno}: record needs string content", E_TYPE
            )
        kind = record.get("kind", "semantic")
        if kind not in _SKILL_KINDS:
            raise StoreError(
                f"memories.jsonl line {lineno}: unknown kind {kind!r}; expected one of {_SKILL_KINDS}",
                E_TYPE,
            )
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise StoreError(f"memories.jsonl line {lineno}: metadata must be an object", E_TYPE)
        records.append({"content": record["content"], "kind": kind, "metadata": metadata})
    return records


def import_skill(store, path: str | Path, overwrite: bool = False,
                 tenant: str = DEFAULT_TENANT) -> dict:
    """Import a skill bundle; every memory is re-tagged with manifest.name.

    Duplicate-safety: refuses when live memories tagged with that skill already
    exist unless overwrite=True, which soft-deletes the existing rows first.
    Procedural rows receive the manifest triggers as trigger_patterns.
    """
    path = Path(path)
    if not path.is_file():
        raise StoreError(f"skill bundle not found: {path}", E_NOT_FOUND)
    try:
        with zipfile.ZipFile(path) as zf:
            missing = {_MANIFEST, _MEMORIES} - set(zf.namelist())
            if missing:
                raise StoreError(f"bundle is missing {sorted(missing)}", E_TYPE)
            # SEC-A3-06: size gate BEFORE the first read(); namelist() proves
            # presence, not size, so a bomb still reached zf.read() unbounded.
            _enforce_bundle_size_budget(zf)
            try:
                manifest = json.loads(_read_member(zf, _MANIFEST).decode("utf-8"))
            except ValueError as exc:
                raise StoreError(f"{_MANIFEST} is not valid JSON ({exc})", E_TYPE) from exc
            _validate_manifest(manifest)
            memories = _parse_memory_lines(_read_member(zf, _MEMORIES).decode("utf-8"))
    except zipfile.BadZipFile as exc:
        raise StoreError(f"not a valid skill bundle zip: {exc}", E_TYPE) from exc
    except OSError as exc:
        raise StoreError(f"cannot read skill bundle: {exc}", E_NOT_FOUND) from exc

    name = manifest["name"]
    triggers = manifest.get("triggers") or []
    existing = _skill_rows(store, name, tenant=tenant)
    if existing and not overwrite:
        raise StoreError(f"skill '{name}' already imported", E_CONSTRAINT)
    if existing:
        store.forget_many([int(row["id"]) for row in existing], tenant=tenant)
    for record in memories:
        store.remember(
            record["content"],
            kind=record["kind"],
            metadata={**record["metadata"], "skill": name},
            trigger_patterns=triggers if record["kind"] == "procedural" and triggers else None,
            tenant=tenant,
        )
    return {"name": name, "memories": len(memories), "triggers": len(triggers)}


def list_skills(store, tenant: str = DEFAULT_TENANT) -> list[dict]:
    """Group live memories by metadata["skill"]; counts sorted descending."""
    counts: dict[str, int] = {}
    for row in store.conn.execute(
        f"SELECT metadata FROM memories WHERE {_SELECTIVE} AND tenant_id = ?",
        (tenant,),
    ):
        try:
            metadata = json.loads(row["metadata"])
        except ValueError:
            continue
        if not isinstance(metadata, dict):
            continue
        name = metadata.get("skill")
        if isinstance(name, str) and name:
            counts[name] = counts.get(name, 0) + 1
    return [
        {"name": name, "memories": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
