"""Two-minute walkthrough of OmniDB before wiring it into Claude Desktop.

    python examples/demo.py

Uses a throwaway home directory; safe to run repeatedly.
"""

import json
import tempfile
from pathlib import Path

from omnidb_server.config import Config
from omnidb_server.consolidate import run_consolidation
from omnidb_server.server import open_owned_store

home = Path(tempfile.mkdtemp()) / "omnidb-demo"

# AUDIT-I02: even a throwaway demo owns the home through the shared helper,
# so the lock is held before any writable initialization and released on exit.
with open_owned_store(Config.from_env(home)) as store:
    print(f"data dir: {home}\n")

    decisions = [
        ("We chose SQLite in WAL mode as the storage substrate", "semantic"),
        ("Recall fuses BM25 lexical and dense vector results with RRF", "semantic"),
        ("Standup: fixed the hnswlib deletion-tracking bug", "episodic"),
    ]
    ids = []
    for text, kind in decisions:
        result = store.remember(text, kind=kind)
        ids.append(result["id"])
        print(f"remember[{result['id']}] ({kind}): {text[:50]}...")

    store.link(ids[0], ids[1], label="related")
    print(f"\nlinked {ids[0]} -[{ 'related' }]-> {ids[1]}")

    hits = store.recall("how are lexical and vector search combined?", k=2)
    print("\nrecall 'how are lexical and vector search combined?':")
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['content'][:60]}")

    print(f"\nrelated({ids[0]}):", [r["content"][:40] for r in store.related(ids[0])])

    report = run_consolidation(store, hours=24)
    print(f"\nconsolidation: {json.dumps({k: report[k] for k in ('episodes_scanned', 'facts_created')})}")

    print(f"\nstats:\n{json.dumps(store.stats(), indent=2)}")

print("\nNext: point Claude Desktop at `python -m omnidb_server serve` (see README).")
