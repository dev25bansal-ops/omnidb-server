"""Command-line entry point for `python -m omnidb_server`.

Mirrors the `omnidb-server` console script and is the path used by the
MCP client when launching the server as a stdio subprocess.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
