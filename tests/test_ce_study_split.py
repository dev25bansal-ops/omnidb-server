"""Unit tests for CE-study split and manifest contracts."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from ce_study_split import (  # noqa: E402
    partition_query_items,
    split_manifest,
    split_query_ids,
    validate_manifest,
)


def test_split_is_stable_under_input_order_and_disjoint():
    ids = [f"q{i}" for i in range(31)]
    a = split_query_ids("fixture", ids)
    b = split_query_ids("fixture", reversed(ids))
    assert a == b
    assert not set(a["dev"]) & set(a["test"])
    assert set(a["dev"]) | set(a["test"]) == set(ids)
    assert len(a["dev"]) == int(len(ids) * 0.4)


def test_split_rejects_duplicate_ids_and_bad_fraction():
    with pytest.raises(ValueError, match="duplicate"):
        split_query_ids("fixture", ["q1", "q1"])
    for fraction in (0, 1, -0.1, 1.1):
        with pytest.raises(ValueError, match="dev_fraction"):
            split_query_ids("fixture", ["q1", "q2"], fraction)


def test_manifest_checks_hashes_and_partition():
    manifest = split_manifest({"a": ["q1", "q2", "q3"], "b": ["z", "y"]})
    validate_manifest(manifest)
    manifest["corpora"]["a"]["test_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(manifest)


def test_partition_keeps_query_rows_aligned_and_rejects_missing_duplicate():
    rows = [(f"q{i}", f"query {i}", {f"d{i}": 1}) for i in range(20)]
    split = split_query_ids("fixture", [r[0] for r in rows])
    parts = partition_query_items(rows, split)
    assert [r[0] for r in parts["dev"]] == [q for q in [r[0] for r in rows] if q in set(split["dev"])]
    assert [r[0] for r in parts["test"]] == [q for q in [r[0] for r in rows] if q in set(split["test"])]
    with pytest.raises(ValueError, match="duplicate"):
        partition_query_items(rows + [rows[0]], split)
    with pytest.raises(ValueError, match="match split"):
        partition_query_items(rows[:-1], split)
