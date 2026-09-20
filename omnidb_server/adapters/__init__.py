"""Framework adapters exposing a MemoryStore behind third-party memory APIs.

Every adapter in this package wraps an already-built
:class:`omnidb_server.store.MemoryStore` and re-exports its remember/recall/
forget surface through the duck type of a framework memory interface. The
frameworks themselves (langgraph, llama_index) are optional dependencies and
are never imported here — the adapters only match their method signatures, so
plugging OmniDB into a framework does not pull that framework into OmniDB's
install graph, and OmniDB installs cleanly without either.

Contract shared by all adapters:

- the wrapped store is passed positionally: ``Adapter(memory_store)``
- writes map to ``MemoryStore.remember``; memory ids are always auto-assigned
  (any framework-level key is carried in memory metadata instead)
- reads map to ``MemoryStore.recall`` / direct SQL over the live row set
- wrong-shape inputs raise :class:`omnidb_server.store.StoreError` with an
  ``OMN-*`` code (type problems use ``OMN-002``)

Re-exports are lazy: importing ``omnidb_server.adapters`` imports neither
adapter module until ``OmniDBStore`` / ``OmniDBMemory`` is actually accessed.
"""

from __future__ import annotations

__all__ = ["OmniDBMemory", "OmniDBStore"]


def __getattr__(name: str):
    # Lazy: adapter modules stay unimported until one of these names is used.
    if name == "OmniDBStore":
        from .langgraph import OmniDBStore

        return OmniDBStore
    if name == "OmniDBMemory":
        from .llamaindex import OmniDBMemory

        return OmniDBMemory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
