"""Responsiveness under a slow LLM (analysis §6): consolidation talking to a
300ms/roundtrip endpoint must not stall concurrent readers.

The anyio story (`server._call` -> anyio.to_thread.run_sync) only holds if
consolidation holds no lock the readers need; at the store level that means
recall()/stats() issued from another thread while run_consolidation blocks on
httpx must each complete well under the LLM round-trip time.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from conftest import make_store

from omnidb_server.consolidate import run_consolidation

LLM_DELAY_S = 0.3
READ_BUDGET_S = 0.5


class _SlowChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        time.sleep(LLM_DELAY_S)
        body = json.dumps({
            "choices": [{"message": {"content": json.dumps([
                {"fact": "User prefers dark mode", "confidence": 0.9},
                {"fact": "User works in the berlin office", "confidence": 0.8},
            ])}}],
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep pytest output clean
        pass


def test_readers_stay_responsive_while_slow_llm_consolidation_in_flight(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        store = make_store(tmp_path, llm_base_url=base_url, llm_model="test-model")
        for i in range(3):
            store.remember(f"episode {i}: user prefers dark mode, berlin office",
                           kind="episodic")

        done = threading.Event()
        outcome: dict = {}

        def _consolidate():
            try:
                outcome["result"] = run_consolidation(store, hours=24)
            except Exception as exc:  # surface worker failures to the main thread
                outcome["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=_consolidate, name="consolidation", daemon=True)
        worker.start()

        # consolidation is now blocked on the 300ms LLM round-trip; readers
        # must sail through while it is in flight
        timings = {}
        for name in ("recall", "recall2", "stats"):
            t0 = time.perf_counter()
            if name.startswith("recall"):
                hits = store.recall("dark mode preferences", k=3)
                assert isinstance(hits, list)
            else:
                payload = store.stats()
                assert payload["live"] >= 3
            timings[name] = time.perf_counter() - t0
            assert timings[name] < READ_BUDGET_S, (name, timings)

        assert done.wait(timeout=15), "consolidation thread never finished"
        worker.join(timeout=5)
        assert outcome.get("error") is None, outcome.get("error")

        result = outcome["result"]
        assert result["llm_used"] is True
        assert result["facts_created"] > 0
        store.close()
    finally:
        server.shutdown()
        server.server_close()
