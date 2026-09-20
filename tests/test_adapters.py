from __future__ import annotations

import pytest

from omnidb_server.adapters import OmniDBMemory, OmniDBStore
from omnidb_server.store import StoreError

# -- langgraph duck type ------------------------------------------------------


def test_langgraph_put_get_roundtrip(store):
    st = OmniDBStore(store)
    st.store.remember("unrelated filler about coffee roasting")
    st.put(("agents", "writer"), "pref-1", {"text": "prefers terse replies", "tone": "dry"})

    item = st.get(("agents", "writer"), "pref-1")
    assert item is not None
    assert item["key"] == "pref-1"
    assert item["namespace"] == ["agents", "writer"]
    assert item["value"]["text"] == "prefers terse replies"
    assert item["value"]["tone"] == "dry"
    assert st.get(("agents", "writer"), "missing") is None


def test_langgraph_put_falls_back_to_content_and_json(store):
    st = OmniDBStore(store)
    st.put(("a",), "via-content", {"content": "fallback to content key"})
    st.put(("a",), "via-json", {"count": 3, "unit": "apples"})

    assert st.get(("a",), "via-content")["value"] == {"content": "fallback to content key"}
    assert st.get(("a",), "via-json")["value"] == {"count": 3, "unit": "apples"}


def test_langgraph_search_filters_by_namespace(store):
    st = OmniDBStore(store)
    st.put(("agents", "writer"), "w-1", {"text": "writer note about deadlines"})
    st.put(("agents", "writer"), "w-2", {"text": "writer note about outlines"})
    st.put(("agents", "reader"), "r-1", {"text": "reader note about margins"})

    hits = st.search(("agents", "writer"), "writer note", limit=5)
    # w-1/w-2 are symmetric for this query; fusion weighting may flip their
    # tie-order, so assert membership rather than a specific order
    assert {h["key"] for h in hits} == {"w-1", "w-2"}
    assert all(h["namespace"] == ["agents", "writer"] for h in hits)
    # the reader note shares tokens but must be excluded by the namespace filter
    assert all(h["key"] != "r-1" for h in st.search(("agents", "writer"), "writer note", limit=10))
    assert all(h["key"] != "w-1" for h in st.search(("agents", "reader"), "writer note"))
    # child namespaces count toward a parent prefix
    st.put(("agents", "writer", "drafts"), "w-3", {"text": "writer draft fragment"})
    assert "w-3" in [h["key"] for h in st.search(("agents", "writer"), "writer", limit=10)]


def test_langgraph_delete_roundtrip(store):
    st = OmniDBStore(store)
    st.put(("s",), "doomed", {"text": "delete me"})
    st.delete(("s",), "doomed")
    assert st.get(("s",), "doomed") is None
    st.delete(("s",), "doomed")  # deleting a missing key is a no-op
    st.put(("s",), "kept", {"text": "still here"})
    assert st.get(("s",), "kept") is not None


def test_langgraph_rejects_wrong_shapes(store):
    st = OmniDBStore(store)
    with pytest.raises(StoreError, match="namespace must be a tuple"):
        st.put(["agents"], "k", {"text": "x"})  # type: ignore[arg-type]
    with pytest.raises(StoreError, match="key must be a non-empty string"):
        st.put(("s",), "", {"text": "x"})
    with pytest.raises(StoreError, match="value must be a dict"):
        st.put(("s",), "k", "bare string")  # type: ignore[arg-type]
    with pytest.raises(StoreError, match="namespace must not be empty"):
        st.get((), "k")
    with pytest.raises(StoreError, match="query must be a non-empty string"):
        st.search(("s",), "   ")
    with pytest.raises(StoreError, match="limit must be an int"):
        st.search(("s",), "query", limit=0)


# -- llamaindex duck type -----------------------------------------------------


def test_llamaindex_put_and_get_all(store):
    mem = OmniDBMemory(store)
    mem.put("user likes chai over coffee")
    mem.put("deploy happens friday noon")
    assert mem.get_all() == ["user likes chai over coffee", "deploy happens friday noon"]


def test_llamaindex_get_ranks_relevant_snippets(store):
    mem = OmniDBMemory(store)
    mem.put("the release train leaves friday noon sharp")
    mem.put("garden tomatoes need watering daily")
    hits = mem.get("release train friday")
    assert hits, "get() returned no snippets"
    assert hits[0] == "the release train leaves friday noon sharp"


def test_llamaindex_clear_removes_everything(store):
    mem = OmniDBMemory(store)
    mem.put("first snippet")
    mem.put("second snippet")
    mem.clear()
    assert mem.get_all() == []
    assert mem.store.stats()["live"] == 0
    mem.clear()  # clearing an empty store is a no-op


def test_llamaindex_rejects_wrong_shapes(store):
    mem = OmniDBMemory(store)
    with pytest.raises(StoreError, match="data must be a non-empty string"):
        mem.put(42)  # type: ignore[arg-type]
    with pytest.raises(StoreError, match="data must be a non-empty string"):
        mem.put("   ")
    with pytest.raises(StoreError, match="query must be a non-empty string"):
        mem.get(None)  # type: ignore[arg-type]


def test_adapters_package_lazy_and_optional_dep_free():
    import sys

    from omnidb_server import adapters

    # lazy re-exports resolve without importing either optional framework
    assert adapters.OmniDBStore is not None
    assert adapters.OmniDBMemory is not None
    assert not any(m in sys.modules for m in ("langgraph", "llama_index", "llama_index.core"))
    with pytest.raises(AttributeError):
        adapters.nonexistent
