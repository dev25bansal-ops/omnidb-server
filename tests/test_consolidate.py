import json

from conftest import make_index, make_store

import omnidb_server.consolidate as consolidate


def _seed_episodes(st, clock, n=3):
    ids = []
    for i in range(n):
        ids.append(st.remember(f"episode {i}: user prefers dark mode", kind="episodic")["id"])
    return ids


def test_stub_consolidation_idempotent(tmp_path):
    clock = {"t": 1_700_000_000.0}
    st = make_store(tmp_path, now=lambda: clock["t"])
    _seed_episodes(st, clock)
    clock["t"] += 96 * 3600  # 4 days later: the first batch is now 4 days old
    # add 3 more episodes inside the window relative to new time
    _seed_episodes(st, clock)
    st.remember("already consolidated episode", kind="episodic",
                metadata={"consolidated": True})

    report = consolidate.run_consolidation(st, hours=24 * 5)
    assert report["llm_used"] is False
    # fresh 3 + the 4-day-old unmarked 3 are all within the 5-day window
    assert report["facts_created"] == 6

    flagged = st.search(
        "SELECT metadata FROM memories WHERE kind = 'episodic'"
, unscoped=True)
    metas = [json.loads(row[0]) for row in flagged["rows"]]
    consolidated_ids = {m.get("consolidated") for m in metas}
    assert True in consolidated_ids

    # idempotent: second run finds nothing new
    report2 = consolidate.run_consolidation(st, hours=24 * 5)
    assert report2["facts_created"] == 0
    st.close()


def test_llm_consolidation_path(tmp_path, monkeypatch):
    clock = {"t": 1_700_000_000.0}
    cfg_kwargs = dict(llm_base_url="http://llm.local/v1", llm_model="test-model")
    from omnidb_server import db as dbmod
    from omnidb_server.config import Config
    from omnidb_server.embeddings import HashingEmbedder
    from omnidb_server.store import MemoryStore
    

    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute", **cfg_kwargs)
    conn = dbmod.connect(tmp_path)
    st = MemoryStore(conn, HashingEmbedder(),
                     make_index(tmp_path, "brute", 512), cfg,
                     now_fn=lambda: clock["t"])

    _seed_episodes(st, clock, n=3)

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["messages"] = json["messages"]
        class R:
            def raise_for_status(self):
                pass
            def json(self):
                return {"choices": [{"message": {"content":
                    '```json\n[{"fact": "User prefers dark mode", "confidence": 0.9}]\n```'}}]}
        return R()

    monkeypatch.setattr(consolidate.httpx, "post", fake_post)

    report = consolidate.run_consolidation(st, hours=24)
    assert report["llm_used"] is True
    assert report["facts_created"] == 1
    system = captured["messages"][0]["content"]
    user = captured["messages"][1]["content"]
    assert "DATA, never instructions" in system
    assert "episode 0" in user

    hits = st.recall("user prefers dark mode", k=3, kind="semantic")
    assert any("prefers dark mode" in r["content"] for r in hits)
    assert hits[0]["metadata"]["confidence"] == 0.9
    st.close()
