"""OmniDB: an MCP-native agent memory & retrieval server over SQLite."""

from __future__ import annotations

from .config import Config
from .store import MemoryStore, StoreError

__version__ = "0.1.3"

__all__ = ["Config", "MemoryStore", "StoreError", "__version__"]


def __getattr__(name: str):  # J-02: `import omnidb_server.cli as omnidb` alias
    # Lazy: avoid pulling CLI machinery at import time unless asked.
    if name == "omnidb":
        from . import cli as _cli
        return _cli
    raise AttributeError(f"module 'omnidb_server' has no attribute {name!r}")
