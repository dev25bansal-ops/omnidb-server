from conftest import make_store


def test_cold_start_recorded_and_stable(tmp_path):
    st = make_store(tmp_path)
    try:
        first = st.stats()
        second = st.stats()
        assert isinstance(first["cold_start_ms"], float)
        # Generous upper bound — cold start on a tiny hashing/brute home is
        # well under 500ms even on slow CI; we just need a sanity ceiling.
        assert first["cold_start_ms"] >= 0.0
        assert first["cold_start_ms"] < 500
        # Subsequent stats() calls must return the same cached value, not
        # re-measure (the tripwire is the first open, not every probe).
        assert second["cold_start_ms"] == first["cold_start_ms"]
    finally:
        st.close()


def test_recall_works_after_cold_start(tmp_path):
    st = make_store(tmp_path)
    try:
        st.remember("the quick brown fox")
        st.remember("lazy dog sleeps under the porch")
        # Smoke that the open path didn't leave the store half-built:
        # if _check_consistency or _recover_index_state had blown up silently
        # recall would crash or return empty.
        hits = st.recall("quick fox", k=2)
        assert hits, "recall returned no rows after a clean cold start"
        assert any("fox" in h["content"] for h in hits)
    finally:
        st.close()