from __future__ import annotations

import pytest
from conftest import make_store

import omnidb_server.consolidate as consolidate
from omnidb_server import llm as llm_mod
from omnidb_server.config import Config
from omnidb_server.llm import probe, resolve_llm_config

_OLLAMA_DEFAULT = "http://127.0.0.1:11434/v1"


def _cfg(**kwargs):
    return Config(home=".", **kwargs)


def _models_response(payload):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return R()


def test_explicit_config_wins(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _models_response({"data": [{"id": "wrong-model"}]})

    monkeypatch.setattr(llm_mod.httpx, "get", fake_get)
    cfg = _cfg(llm_base_url="http://llm.local/v1", llm_model="gpt-x",
               ollama_autodetect=True)
    assert resolve_llm_config(cfg) == ("http://llm.local/v1", "gpt-x")
    assert calls == []  # no probe when both values are pinned


def test_autodetect_off_passthrough():
    assert resolve_llm_config(_cfg(llm_base_url="http://b/v1")) == ("http://b/v1", None)
    assert resolve_llm_config(_cfg(llm_model="m")) == (None, "m")
    assert resolve_llm_config(_cfg()) == (None, None)


def test_autodetect_default_endpoint_first_model(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _models_response({"data": [{"id": "llama3.2"}]})

    monkeypatch.setattr(llm_mod.httpx, "get", fake_get)
    assert resolve_llm_config(_cfg(ollama_autodetect=True)) == (_OLLAMA_DEFAULT, "llama3.2")
    assert captured["url"] == f"{_OLLAMA_DEFAULT}/models"
    assert captured["timeout"] == 1.5

    # explicit base probes that endpoint; a pinned model is kept,
    # an unpinned model takes the first id from the listing
    cfg = _cfg(llm_base_url="http://llmfile.local/v1", llm_model="pinned",
               ollama_autodetect=True)
    assert resolve_llm_config(cfg) == ("http://llmfile.local/v1", "pinned")
    cfg = _cfg(llm_base_url="http://llmfile.local/v1", ollama_autodetect=True)
    assert resolve_llm_config(cfg) == ("http://llmfile.local/v1", "llama3.2")
    assert captured["url"] == "http://llmfile.local/v1/models"


def test_autodetect_failure_is_none(monkeypatch):
    def refused(url, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_mod.httpx, "get", refused)
    # even a pinned base degrades to (None, None) when unreachable
    assert resolve_llm_config(_cfg(ollama_autodetect=True)) == (None, None)
    assert resolve_llm_config(_cfg(llm_base_url="http://dead.local/v1",
                                   ollama_autodetect=True)) == (None, None)


def test_probe_rejects_unusable_listings(monkeypatch):
    # non-2xx status -> None
    def http_error(url, **kwargs):
        class R:
            def raise_for_status(self):
                raise RuntimeError("500")

            def json(self):
                return {"data": [{"id": "x"}]}

        return R()

    monkeypatch.setattr(llm_mod.httpx, "get", http_error)
    assert probe("http://anywhere/v1") is None

    # empty listing -> None; non-dict/blank entries are skipped
    monkeypatch.setattr(llm_mod.httpx, "get", lambda url, **kw: _models_response({"data": []}))
    assert probe("http://b/v1") is None
    monkeypatch.setattr(
        llm_mod.httpx, "get",
        lambda url, **kw: _models_response(
            {"data": [{"id": "m1"}, {"object": "model"}, {"id": None}, {"id": "m2"}]}))
    assert probe("http://b/v1") == ["m1", "m2"]


@pytest.mark.parametrize("payload,expected", [
    ({"data": [{"id": "x"}]}, ["x"]),
    ({"data": [["not-a-dict"]]}, None),
    ({"unexpected": []}, None),
])
def test_probe_shapes(monkeypatch, payload, expected):
    monkeypatch.setattr(llm_mod.httpx, "get", lambda url, **kw: _models_response(payload))
    assert probe("http://b/v1") == expected


def test_probe_malformed_json_never_raises(monkeypatch):
    def bad_json(url, **kwargs):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("not json")

        return R()

    monkeypatch.setattr(llm_mod.httpx, "get", bad_json)
    assert probe("http://b/v1") is None
    assert resolve_llm_config(_cfg(ollama_autodetect=True)) == (None, None)


def test_consolidation_uses_resolved_endpoint(tmp_path, monkeypatch):
    # integration: autodetect on with NO explicit config; the probe resolves
    # an endpoint and facts flow through it via the chat/completions call
    monkeypatch.setattr(llm_mod, "probe", lambda base: ["llama3.2"])
    st = make_store(tmp_path, ollama_autodetect=True)
    for i in range(3):
        st.remember(f"episode {i}: user prefers dark mode", kind="episodic")

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        captured["headers"] = headers
        class R:
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"message": {"content":
                    '[{"fact": "User prefers dark mode", "confidence": 0.9}]'}}]}
        return R()

    monkeypatch.setattr(consolidate.httpx, "post", fake_post)

    report = consolidate.run_consolidation(st, hours=24)
    assert report["llm_used"] is True
    assert report["facts_created"] == 1
    assert captured["url"] == f"{_OLLAMA_DEFAULT}/chat/completions"
    assert captured["model"] == "llama3.2"
    assert captured["headers"] == {}  # no api_key -> no Authorization header
    st.close()


def test_consolidation_autodetect_failure_runs_stub(tmp_path, monkeypatch):
    # no reachable endpoint -> (None, None) -> offline stub consumes episodes
    def refused(url, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_mod.httpx, "get", refused)
    st = make_store(tmp_path, ollama_autodetect=True)
    for i in range(3):
        st.remember(f"episode {i}: user prefers dark mode", kind="episodic")

    report = consolidate.run_consolidation(st, hours=24)
    assert report["llm_used"] is False
    assert report["facts_created"] == 3
    assert report["llm_error"] is None
    assert report["code"] is None
    st.close()
