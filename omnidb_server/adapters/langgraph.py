"""LangGraph adapter: MemoryStore behind a BaseStore-shaped duck type.

LangGraph's ``BaseStore`` (as exposed by ``InMemoryStore`` / ``PostgresStore``)
is a namespaced key-value memory with semantic search. This module implements
that surface without importing langgraph: any object accepting these method
signatures can stand in where LangGraph expects a store.

Mapping table (LangGraph -> OmniDB):

=========================  ====================================================
``put(ns, key, value)``    ``remember`` with content from ``value["text"]`` /
                           ``value["content"]`` (else the JSON dump of
                           ``value``); metadata carries the namespace join,
                           the framework key under ``"key"``, and the original
                           ``value`` under the reserved ``"_value"`` key so
                           ``get`` round-trips exactly; ids auto-assigned
``get(ns, key)``           metadata lookup for ``"key"``; returns
                           ``{"value": ..., "key": ..., "namespace": ...,
                           "id": row_id}`` (LangGraph item shape) or ``None``
``search(prefix, query)``  ``recall(query, k=limit*4)`` filtered to hits whose
                           metadata namespace equals the prefix or sits under
                           it, truncated to ``limit``
``delete(ns, key)``        locate the row like ``get`` then ``forget(id)``
=========================  ====================================================

``get``/``delete`` locate rows with a LIKE prefilter on the canonical
``json.dumps`` metadata form (``'"key": "<key>"'``) plus an exact comparison
on the parsed dict — the same two-stage heuristic ``skills.py`` uses, which
keeps the common path a single scan and never over-matches. Names containing
quotes or backslashes can under-match at the SQL stage (a documented
limitation of the heuristic, shared with ``skills.py``); the parsed
comparison stays exact.
"""

from __future__ import annotations

import json

from ..store import _SELECTIVE, E_TYPE, MemoryStore, StoreError

_TEXT_KEYS = ("text", "content")
_RESERVED_META = ("key", "namespace", "_value")


class OmniDBStore:
    """Duck-typed LangGraph BaseStore over an OmniDB MemoryStore.

    Writes are scoped to an ``agent_id`` (default ``"langgraph"``) so the
    adapter's puts/gets/searches/deletes never touch memories belonging to
    other agents sharing the same home (BUG-05). Pass ``agent_id=None`` to
    opt out and operate on the whole store.
    """

    def __init__(self, store: MemoryStore, agent_id: str | None = "langgraph"):
        self.store = store
        self.agent_id = agent_id

    # -- validation helpers -------------------------------------------------

    @staticmethod
    def _require_namespace(namespace, op: str) -> tuple:
        if not isinstance(namespace, tuple) or not all(
            isinstance(p, str) for p in namespace
        ):
            raise StoreError(
                f"{op}: namespace must be a tuple of strings, got {namespace!r}", E_TYPE
            )
        if not namespace:
            raise StoreError(f"{op}: namespace must not be empty", E_TYPE)
        return namespace

    @staticmethod
    def _require_key(key: str, op: str) -> str:
        if not isinstance(key, str) or not key:
            raise StoreError(f"{op}: key must be a non-empty string, got {key!r}", E_TYPE)
        return key

    def _find_row(self, namespace: tuple, key: str):
        """LIKE prefilter on the json metadata form, exact match on parse."""
        ns = "/".join(namespace)
        rows = self.store.conn.execute(
            f"SELECT * FROM memories WHERE metadata LIKE ? AND {_SELECTIVE} "
            "AND (? IS NULL OR agent_id = ?) ORDER BY id",
            ('%"key": "' + key + '"%', self.agent_id, self.agent_id),
        ).fetchall()
        for row in rows:
            meta = json.loads(row["metadata"])
            if meta.get("key") == key and meta.get("namespace") == ns:
                return row, meta
        return None, None

    # -- BaseStore surface ----------------------------------------------------

    def put(self, namespace: tuple, key: str, value: dict) -> None:
        namespace = self._require_namespace(namespace, "put")
        key = self._require_key(key, "put")
        if not isinstance(value, dict):
            raise StoreError(
                f"put: value must be a dict, got {type(value).__name__}", E_TYPE
            )
        content = next((value[k] for k in _TEXT_KEYS if isinstance(value.get(k), str)), None)
        if content is None:
            content = json.dumps(value, sort_keys=True)
        metadata = dict(value.get("metadata") or {})
        metadata["key"] = key
        metadata["namespace"] = "/".join(namespace)
        metadata["_value"] = json.dumps(value, sort_keys=True)
        # LangGraph put is an upsert (BUG-06): write the new row first, then
        # retire the previous one, so a failed write leaves the old value
        # intact instead of a missing key.
        existing, _meta = self._find_row(namespace, key)
        self.store.remember(content, kind="semantic", metadata=metadata,
                            agent_id=self.agent_id)
        if existing is not None:
            self.store.forget(int(existing["id"]))

    def get(self, namespace: tuple, key: str) -> dict | None:
        namespace = self._require_namespace(namespace, "get")
        key = self._require_key(key, "get")
        row, meta = self._find_row(namespace, key)
        if row is None:
            return None
        return {
            "value": _value_of(meta, row["content"]),
            "key": key,
            "namespace": list(namespace),
            "id": int(row["id"]),
        }

    def search(self, namespace_prefix: tuple, query: str, limit: int = 5) -> list[dict]:
        namespace_prefix = self._require_namespace(namespace_prefix, "search")
        if not isinstance(query, str) or not query.strip():
            raise StoreError("search: query must be a non-empty string", E_TYPE)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise StoreError(f"search: limit must be an int >= 1, got {limit!r}", E_TYPE)
        prefix = "/".join(namespace_prefix)
        out: list[dict] = []
        for hit in self.store.recall(query, k=limit * 4, agent_id=self.agent_id):
            meta = hit.get("metadata") or {}
            ns = str(meta.get("namespace", ""))
            if ns != prefix and not ns.startswith(prefix + "/"):
                continue
            out.append({
                "value": _value_of(meta, hit.get("content", "")),
                "key": meta.get("key"),
                "namespace": ns.split("/") if ns else [],
                "score": hit["score"],
            })
            if len(out) >= limit:
                break
        return out

    def delete(self, namespace: tuple, key: str) -> None:
        namespace = self._require_namespace(namespace, "delete")
        key = self._require_key(key, "delete")
        row, _meta = self._find_row(namespace, key)
        if row is not None:
            self.store.forget(int(row["id"]))


def _value_of(meta: dict, content: str) -> dict:
    """Reconstruct the put-time value from metadata (exact) or content (fallback)."""
    if "_value" in meta:
        return json.loads(meta["_value"])
    value = {k: v for k, v in meta.items() if k not in _RESERVED_META}
    if content and "text" not in value and "content" not in value:
        value["text"] = content
    return value
