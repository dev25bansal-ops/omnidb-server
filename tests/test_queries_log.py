import json
import subprocess
import sys

import conftest


def _queries_rows(store):
    return store.conn.execute(
        "SELECT kind, query, result_count, took_ms, agent_id FROM queries"
    ).fetchall()


def test_query_log_disabled_leaves_table_empty(tmp_path):
    st = conftest.make_store(tmp_path)
    try:
        st.remember("alpha bravo charlie delta")
        st.recall("alpha bravo", k=3)
        st.search("SELECT COUNT(*) AS n FROM memories")
        assert _queries_rows(st) == []
    finally:
        st.close()


def test_query_log_enabled_records_recall_and_search(tmp_path):
    st = conftest.make_store(tmp_path, query_log=True)
    try:
        st.remember("python decorators guide with examples and patterns")
        st.remember("python generators and iterators explained")
        st.remember("history of jazz chords and improvisation")
        st.recall("python decorators", k=5, agent_id="agent-A")
        st.search("SELECT COUNT(*) AS n FROM memories")
        rows = _queries_rows(st)
        assert len(rows) == 2
        kinds = [r["kind"] for r in rows]
        assert "recall" in kinds
        assert "search" in kinds
        recall_row = next(r for r in rows if r["kind"] == "recall")
        search_row = next(r for r in rows if r["kind"] == "search")
        assert recall_row["query"] == "python decorators"
        assert isinstance(recall_row["result_count"], int)
        assert recall_row["result_count"] >= 0
        assert isinstance(recall_row["took_ms"], float)
        assert recall_row["took_ms"] >= 0
        assert recall_row["agent_id"] == "agent-A"
        assert search_row["query"].startswith("SELECT COUNT")
        assert search_row["result_count"] >= 1
        assert isinstance(search_row["took_ms"], float)
    finally:
        st.close()


def test_cli_query_log_prints_json(tmp_path):
    st = conftest.make_store(tmp_path, query_log=True)
    try:
        st.remember("alpha bravo charlie")
        st.recall("alpha bravo", k=3)
        proc = subprocess.run(
            [sys.executable, "-m", "omnidb_server", "--home", str(tmp_path),
             "query-log", "--limit", "10"],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
        assert isinstance(payload, list)
        assert len(payload) >= 1
        row = payload[0]
        assert row["kind"] == "recall"
        assert "took_ms" in row
        assert "result_count" in row
    finally:
        st.close()