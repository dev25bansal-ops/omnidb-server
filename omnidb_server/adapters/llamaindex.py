"""LlamaIndex adapter: MemoryStore behind a BaseMemoryStore-shaped duck type.

LlamaIndex's ``BaseMemoryStore`` is a flat string-blob memory: ``put`` files a
text snippet, ``get_all``/``get`` read snippets back, ``clear`` wipes them.
This module implements that surface without importing llama_index.

Mapping table (LlamaIndex -> OmniDB):

=========================  ====================================================
``put(data)``              ``remember(data, kind="episodic")``; ids auto-assigned
``get_all()``              direct SQL over the adapter's live rows, oldest first
                           (``ORDER BY created_ms``), capped at 1000 snippets
``get(query)``             ``recall(query, k=5)`` -> list of hit contents
``clear()``                ``forget_many`` of every live id in the adapter's scope
=========================  ====================================================

All reads and writes are scoped to an ``agent_id`` (default
``"llamaindex"``) so ``clear()`` deletes only what this adapter put there
(BUG-05) — a shared home keeps other agents' memories. Pass
``agent_id=None`` to opt out and operate on the whole store.

The store keeps OmniDB's recency/frequency scoring and FTS5+vector fusion for
``get`` while presenting the plain list-of-strings contract LlamaIndex memory
nodes expect. Wrong-shape inputs raise ``StoreError`` with an ``OMN-*`` code.
"""

from __future__ import annotations

from ..store import _SELECTIVE, E_TYPE, MemoryStore, StoreError

_GET_K = 5
_GET_ALL_CAP = 1000


class OmniDBMemory:
    """Duck-typed LlamaIndex BaseMemoryStore over an OmniDB MemoryStore."""

    def __init__(self, store: MemoryStore, agent_id: str | None = "llamaindex"):
        self.store = store
        self.agent_id = agent_id

    def put(self, data: str) -> None:
        if not isinstance(data, str) or not data.strip():
            raise StoreError(
                f"put: data must be a non-empty string, got {data!r}", E_TYPE
            )
        self.store.remember(data, kind="episodic", agent_id=self.agent_id)

    def get_all(self) -> list[str]:
        rows = self.store.conn.execute(
            f"SELECT content FROM memories WHERE {_SELECTIVE} "
            "AND (? IS NULL OR agent_id = ?) "
            f"ORDER BY created_ms, id LIMIT {_GET_ALL_CAP}",
            (self.agent_id, self.agent_id),
        ).fetchall()
        return [r["content"] for r in rows]

    def get(self, data: str) -> list[str]:
        if not isinstance(data, str) or not data.strip():
            raise StoreError(
                f"get: query must be a non-empty string, got {data!r}", E_TYPE
            )
        return [hit["content"] for hit in
                self.store.recall(data, k=_GET_K, agent_id=self.agent_id)]

    def clear(self) -> None:
        ids = [int(r["id"]) for r in self.store.conn.execute(
            f"SELECT id FROM memories WHERE {_SELECTIVE} "
            "AND (? IS NULL OR agent_id = ?)",
            (self.agent_id, self.agent_id),
        ).fetchall()]
        if ids:
            self.store.forget_many(ids)
