from __future__ import annotations

import json

from conftest import make_store

import omnidb_server.consolidate as consolidate


def test_llm_consolidation_keeps_facts_in_source_agent_scope(tmp_path, monkeypatch):
    st = make_store(
        tmp_path,
        llm_base_url="http://llm.local/v1",
        llm_model="m",
        consolidation_chunk_episodes=2,
    )
    source_agents = {}
    now_ms = int(st.now_fn() * 1000)
    for index, agent in enumerate(("alice", "bob", None, "alice", "bob", None)):
        episode_id = st.remember(
            f"episode {index} private detail",
            kind="episodic",
            agent_id=agent,
            created_ms=now_ms + index,
        )["id"]
        source_agents[episode_id] = agent

    calls = 0

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": json.dumps([
                {"fact": f"extracted fact {calls}", "confidence": 0.9}
            ])}}]}

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return R()

    monkeypatch.setattr(consolidate.httpx, "post", fake_post)

    report = consolidate.run_consolidation(st, hours=24)
    assert report["llm_used"] is True

    facts = st.conn.execute(
        "SELECT agent_id, metadata FROM memories WHERE kind = 'semantic'"
    ).fetchall()
    assert len(facts) == 3
    for row in facts:
        metadata = json.loads(row["metadata"])
        source_ids = metadata["source_episodes"]
        assert len(source_ids) == 2
        source_agent = source_agents[source_ids[0]]
        assert all(source_agents[episode_id] == source_agent for episode_id in source_ids)
        assert row["agent_id"] == source_agent
        if source_agent is not None:
            assert row["agent_id"] in ("alice", "bob")

    scoped_facts = [row for row in facts if row["agent_id"] is not None]
    unscoped_facts = [row for row in facts if row["agent_id"] is None]
    assert len(scoped_facts) == 2
    assert len(unscoped_facts) == 1
    st.close()
