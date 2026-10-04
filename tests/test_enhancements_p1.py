"""Tests for the section-3 P1 enhancements (E-01, E-06, E-07, E-12, E-13, E-15).

See `../OmniDB Vault/06 Analysis/Enhancements and Modifications (Sep 2026).md`.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import make_store

import omnidb_server.consolidate as consolidate
from omnidb_server import cli
from omnidb_server import db as dbmod
from omnidb_server.config import Config, ConfigError
from omnidb_server.db import get_meta
from omnidb_server.embeddings import Embedder, HashingEmbedder
from omnidb_server.portability import backup_home, reembed_home
from omnidb_server.store import MemoryStore, StoreError
from omnidb_server.vector_index import BruteForceIndex

# --- E-06: schema() introspection -------------------------------------------------

def test_schema_reports_tables_and_examples_run(store):
    # A tenant's FTS table is created on first write (ADR-08 / W3-3), so the
    # report has something to name only once there is a row to index.
    store.remember("a note so the lexical index exists")
    payload = store.schema()
    names = {t["name"] for t in payload["tables"]}
    # The shared `memory_fts` is gone; the lexical index is per-tenant, so the
    # report names the caller's own table instead.
    assert {"memories", "edges", "meta", "queries", "fts_default"} <= names
    assert "memory_fts" not in names, \
        "the shared table was dropped in v6; schema() must not advertise it"
    assert not any(name.startswith("fts_default_") for name in names), \
        "FTS shadow tables must not clutter the report"
    assert payload["schema_version"] == dbmod.SCHEMA_VERSION
    assert "fts_default" in payload["notes"], \
        "the per-tenant index must be described, or its example is unexplainable"
    # every advertised example must actually run under the read-only authorizer
    for sql in payload["examples"]:
        result = store.search(sql, unscoped=True)
        assert "rows" in result


# --- E-07: shell-facing recall/remember + serve startup line ----------------------

def test_cli_remember_then_recall(tmp_path, capsys):
    assert cli.main(["remember", "chai over coffee tonight",
                     "--home", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out["id"], int)
    assert out["kind"] == "semantic"

    assert cli.main(["recall", "chai coffee", "--home", str(tmp_path),
                     "--k", "3"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits and hits[0]["content"] == "chai over coffee tonight"

    # invalid metadata JSON is a typed error with exit code 2
    rc = cli.main(["remember", "broken", "--metadata", "{not json",
                   "--home", str(tmp_path)])
    assert rc == 2
    assert "--metadata" in capsys.readouterr().err


def test_cli_schema_command(tmp_path, capsys):
    assert cli.main(["remember", "schema probe row", "--home", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["schema", "--home", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {t["name"] for t in payload["tables"]} >= {"memories", "edges"}


# --- E-15: config validation + effective-config report ----------------------------

def test_config_validation_rejects_broken_values(tmp_path):
    (tmp_path / "omnidb.toml").write_text("persist_every_ops = 0\n")
    with pytest.raises(ConfigError, match="persist_every_ops"):
        Config.load(home=tmp_path)


def test_config_sources_report(tmp_path, monkeypatch):
    (tmp_path / "omnidb.toml").write_text("half_life_hours = 42.0\n")
    monkeypatch.setenv("OMNIDB_RRF_K", "7")
    monkeypatch.delenv("OMNIDB_LLM_API_KEY", raising=False)

    cfg, sources = Config.load_detailed(home=tmp_path)
    assert cfg.half_life_hours == 42.0
    assert sources["half_life_hours"] == str(tmp_path / "omnidb.toml")
    assert sources["rrf_k"] == "env:OMNIDB_RRF_K"
    assert sources["k_fetch"] == "default"
    assert sources["home"] == "cli"  # passed explicitly


def test_cli_config_command_redacts_api_key(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OMNIDB_LLM_API_KEY", "sk-super-secret")
    assert cli.main(["config", "--home", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["values"]["llm_api_key"] == "***set***"
    assert payload["sources"]["llm_api_key"] == "env:OMNIDB_LLM_API_KEY"
    assert "sk-super-secret" not in json.dumps(payload)


# --- E-12: chunked consolidation + dry-run -----------------------------------------

def _seed_episodes(st, n):
    for i in range(n):
        st.remember(f"episode {i}: user prefers dark mode {i}", kind="episodic")


def _llm_store(tmp_path, **extra):
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute",
                 llm_base_url="http://llm.local/v1", llm_model="m", **extra)
    conn = dbmod.connect(tmp_path)
    return MemoryStore(conn, HashingEmbedder(),
                       BruteForceIndex(tmp_path / "vectors.npz", 512), cfg)


def _fake_post(payload_json):
    class R:
        def raise_for_status(self):
            pass

        def json(self_inner):
            return {"choices": [{"message": {"content": payload_json}}]}
    return lambda *a, **k: R()


def test_consolidation_chunks_calls_and_isolates_failures(tmp_path, monkeypatch):
    st = _llm_store(tmp_path, consolidation_chunk_episodes=5)
    _seed_episodes(st, 12)

    calls = {"n": 0}

    def flaky_post(url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("chunk 2 exploded")
        return _fake_post(json_lib.dumps([
            {"fact": f"fact from chunk {calls['n']} about dark mode",
             "confidence": 0.9},
        ]))(url, json, headers, timeout)

    import json as json_lib
    monkeypatch.setattr(consolidate.httpx, "post", flaky_post)

    report = consolidate.run_consolidation(st, hours=24)
    assert calls["n"] == 3                      # 12 episodes / chunk of 5
    assert report["chunks_total"] == 3
    assert report["chunks_failed"] == 1
    assert report["facts_created"] == 2         # chunks 1 and 3 committed
    assert report["code"] is None               # not a total failure

    marked = st.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE kind = 'episodic' "
        "AND metadata LIKE '%consolidated%'"
    ).fetchone()[0]
    assert marked == 7                          # chunks 1+3 (5 + 2 episodes)

    # the retry picks up exactly the failed chunk's episodes
    report2 = consolidate.run_consolidation(st, hours=24)
    assert report2["chunks_total"] == 1
    assert report2["facts_created"] == 1
    st.close()


def test_consolidation_dry_run_writes_nothing(tmp_path):
    st = make_store(tmp_path)
    _seed_episodes(st, 5)

    report = consolidate.run_consolidation(st, hours=24, dry_run=True)
    assert report["dry_run"] is True
    assert report["would_create"] == 5
    assert report["facts_created"] == 0
    assert len(report["episode_ids"]) == 5      # preview of what would be marked
    assert "semantic" not in st.stats()["by_kind"]
    marked = st.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE metadata LIKE '%consolidated%'"
    ).fetchone()[0]
    assert marked == 0

    # the real run then distills everything the dry run previewed
    real = consolidate.run_consolidation(st, hours=24)
    assert real["facts_created"] == 5
    st.close()


def test_consolidation_dry_run_llm_path(tmp_path, monkeypatch):
    st = _llm_store(tmp_path)
    _seed_episodes(st, 3)
    monkeypatch.setattr(consolidate.httpx, "post", _fake_post(json.dumps([
        {"fact": "previewed fact about dark mode", "confidence": 0.9},
    ])))
    report = consolidate.run_consolidation(st, hours=24, dry_run=True)
    assert report["llm_used"] is True
    assert report["would_create"] == 1
    assert report["facts_created"] == 0
    assert "semantic" not in st.stats()["by_kind"]
    st.close()


# --- E-13: hot backup ----------------------------------------------------------------

def test_backup_snapshot_is_consistent_and_openable(tmp_path):
    src = tmp_path / "home"
    backups = tmp_path / "backups"
    st = make_store(src)
    st.remember("backup probe alpha")
    st.remember("backup probe beta")
    st.flush()
    report = backup_home(st, backups)
    snap = Path(report["dest"])
    assert (snap / "omnidb.db").is_file()
    assert "vectors.npz" in report["sidecars"]

    st.remember("post backup gamma")  # written after the snapshot

    # the snapshot opens as an independent home and shows only pre-backup rows
    cfg = Config(home=snap, embedder="hashing", vector_index="brute")
    conn = dbmod.connect(snap)
    emb = HashingEmbedder()
    st2 = MemoryStore(conn, emb,
                      BruteForceIndex.load(snap / "vectors.npz", emb.dim), cfg)
    assert st2.stats()["live"] == 2
    hits = st2.recall("backup probe", k=5)
    assert {h["content"] for h in hits} == {"backup probe alpha",
                                            "backup probe beta"}
    st2.close()
    st.close()


def test_backup_rotation_keeps_newest(tmp_path):
    st = make_store(tmp_path / "home")
    st.remember("rotation probe")
    backup_home(st, tmp_path / "b")
    backup_home(st, tmp_path / "b")
    report = backup_home(st, tmp_path / "b", keep=2)
    assert len(report["pruned"]) == 1
    snaps = sorted(p.name for p in (tmp_path / "b").iterdir()
                   if p.name.startswith("snapshot-"))
    assert len(snaps) == 2
    st.close()


def test_cli_backup_command(tmp_path, capsys):
    st = make_store(tmp_path / "home")
    st.remember("cli backup probe")
    st.close()
    assert cli.main(["backup", str(tmp_path / "out"), "--keep", "1",
                     "--home", str(tmp_path / "home")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (Path(payload["dest"]) / "omnidb.db").is_file()


# --- E-01: re-embed migration ---------------------------------------------------------

class _AltEmbedder(Embedder):
    """Deterministic non-network embedder with a different name and dim."""

    def __init__(self, dim: int = 64):
        self.name = "alt-v1"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
                out[i, bucket] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def test_reembed_migrates_identity_vectors_and_recall(tmp_path):
    st = make_store(tmp_path)
    for i in range(5):
        st.remember(f"migration probe {i} alpha")
    st.close()

    alt = _AltEmbedder(dim=64)
    report = reembed_home(Path(tmp_path), alt, assume_yes=True)
    assert report["migrated"] == 5
    assert report["from"] == "hashing-v1"
    assert report["to"] == "alt-v1"
    assert report["dim"] == 64
    assert report["index_kind"] == "brute"

    # reopen with the new embedder: identity pins match and recall works
    conn = dbmod.connect(tmp_path)
    assert get_meta(conn, "embedder") == "alt-v1"
    assert get_meta(conn, "vec_dim") == "64"
    st2 = MemoryStore(conn, alt,
                      BruteForceIndex.load(tmp_path / "vectors.npz", alt.dim),
                      Config(home=tmp_path, embedder="hashing",
                             vector_index="brute"))
    hits = st2.recall("migration probe 3", k=3)
    assert hits and hits[0]["content"] == "migration probe 3 alpha"
    st2.close()


def test_reembed_refuses_same_embedder(tmp_path):
    st = make_store(tmp_path)
    st.remember("seed")
    st.close()
    with pytest.raises(StoreError, match="already built"):
        reembed_home(Path(tmp_path), HashingEmbedder(), assume_yes=True)


def test_reembed_requires_confirmation_noninteractive(tmp_path):
    st = make_store(tmp_path)
    st.remember("seed")
    st.close()
    with pytest.raises(StoreError, match="--yes"):
        reembed_home(Path(tmp_path), _AltEmbedder(dim=8), assume_yes=False)


def test_reembed_crash_leaves_old_identity_and_heals(tmp_path):
    st = make_store(tmp_path)
    for i in range(300):  # > one 256-row chunk so the failure lands mid-loop
        st.remember(f"crash probe {i} alpha beta")
    st.close()

    alt = _AltEmbedder(dim=8)
    real_encode = alt.encode
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated crash mid-migration")
        return real_encode(texts)

    alt.encode = flaky
    with pytest.raises(RuntimeError, match="simulated crash"):
        reembed_home(Path(tmp_path), alt, assume_yes=True)

    # identity was pinned last: still the old embedder, so the home opens
    conn = dbmod.connect(tmp_path)
    assert get_meta(conn, "embedder") == "hashing-v1"
    conn.close()
    # reopening reconciles the torn sidecar (re-embeds from rows)
    st2 = make_store(tmp_path)
    assert st2.stats()["live"] == 300
    hits = st2.recall("crash probe 7", k=3)
    assert hits and "crash probe 7" in hits[0]["content"]
    st2.close()
