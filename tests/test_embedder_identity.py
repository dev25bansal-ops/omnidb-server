from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_store

from omnidb_server import cli
from omnidb_server import db as dbmod
from omnidb_server.config import Config
from omnidb_server.embeddings import make_embedder
from omnidb_server.server import open_owned_store
from omnidb_server.store import E_TYPE, E_VECTOR_DIM_MISMATCH, StoreError


def _install_fake_fastembed(monkeypatch, requested_models: list[str]) -> None:
    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models():
            return [
                {"model": "BAAI/bge-base-en-v1.5"},
                {"model": "BAAI/bge-small-en-v1.5"},
            ]

        def __init__(self, model_name):
            requested_models.append(model_name)
            self.model_name = model_name

        def embed(self, texts):
            vectors = [
                [1.0, 0.0, 0.0] if index % 2 == 0 else [0.0, 1.0, 0.0]
                for index, _ in enumerate(texts)
            ]
            return iter(np.asarray(vectors, dtype=np.float32))

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )


def test_reembedded_model_reopens_and_reports_same_identity(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    store = make_store(home)
    store.remember("identity persistence probe")
    store.close()

    model_name = "BAAI/bge-base-en-v1.5"
    requested_models: list[str] = []
    _install_fake_fastembed(monkeypatch, requested_models)

    assert cli.main([
        "re-embed", "--home", str(home), "--embedder", "fastembed",
        "--model", model_name, "--yes",
    ]) == 0
    capsys.readouterr()
    assert requested_models == [model_name]

    with open_owned_store(Config.load(home)) as reopened:
        assert reopened.stats()["embedder"]["name"] == f"fastembed:{model_name}"


def test_different_reembedded_models_do_not_collide(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    store = make_store(home)
    store.remember("model collision probe")
    store.close()

    model_names = ("BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5")
    (home / "omnidb.toml").write_text(
        'embedder = "fastembed"\nvector_index = "brute"\n', encoding="utf-8"
    )
    requested_models: list[str] = []
    _install_fake_fastembed(monkeypatch, requested_models)

    for model_name in model_names:
        assert cli.main([
            "re-embed", "--home", str(home), "--embedder", "fastembed",
            "--model", model_name, "--yes",
        ]) == 0
        capsys.readouterr()
        with open_owned_store(Config.load(home)) as reopened:
            assert reopened.stats()["embedder"]["name"] == f"fastembed:{model_name}"

    assert set(requested_models) == set(model_names)


def test_unknown_embedder_selector_raises_typed_error(tmp_path):
    cfg = Config(home=tmp_path, embedder="clip", vector_index="brute")

    with pytest.raises(StoreError) as exc_info:
        make_embedder(cfg)

    assert exc_info.value.code == E_TYPE
    assert "auto, hashing, fastembed" in str(exc_info.value)


def test_unknown_vector_backend_selector_raises_typed_error(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIDB_VECTOR_INDEX", "annoy")

    with pytest.raises(StoreError) as exc_info:
        Config.load(home=tmp_path)

    assert exc_info.value.code == E_TYPE
    assert "auto, hnsw, brute" in str(exc_info.value)


def test_unknown_pinned_fastembed_model_raises_typed_error(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.close()
    conn = dbmod.connect(tmp_path)
    dbmod.set_meta(conn, "embedder", "fastembed:unknown/model")
    conn.close()
    _install_fake_fastembed(monkeypatch, [])

    cfg = Config(home=tmp_path, embedder="fastembed", vector_index="brute")
    with pytest.raises(StoreError) as exc_info:
        make_embedder(cfg, stored_name="fastembed:unknown/model")

    assert exc_info.value.code == E_VECTOR_DIM_MISMATCH
    assert "unknown/model" in str(exc_info.value)


def test_fastembed_vectors_remain_float32(tmp_path, monkeypatch):
    requested_models: list[str] = []
    _install_fake_fastembed(monkeypatch, requested_models)
    cfg = Config(home=tmp_path, embedder="fastembed", vector_index="brute")

    embedder = make_embedder(cfg)

    vector = embedder.encode(["one", "two"])[0]
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
