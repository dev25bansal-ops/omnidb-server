"""Synthetic tests for locked CE-study inference utilities."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from ce_study_stats import clustered_cell_bootstrap, holm_adjust, paired_bootstrap_ci, sign_flip_mean_test


def test_sign_flip_keeps_exact_zero_differences_and_is_deterministic():
    d = [0.0, 0.0, 0.2, 0.4]
    a = sign_flip_mean_test(d, n_permutations=1000, seed=7)
    b = sign_flip_mean_test(d, n_permutations=1000, seed=7)
    assert a == b
    assert a["n_exact_zero"] == 2
    assert 0 < a["p_two_sided"] <= 1


def test_mean_bootstrap_includes_zero_and_positive_effect():
    r = paired_bootstrap_ci([0.0, 0.1, 0.2], n_resamples=2000, seed=3)
    assert r["ci_low"] < r["mean_delta"] < r["ci_high"]
    assert r["n_queries"] == 3


def test_holm_adjustment_known_family():
    p = holm_adjust({"a": .01, "b": .04, "c": .03, "d": .20})
    assert p == {"a": .04, "c": .09, "b": .09, "d": .20}


def test_cluster_bootstrap_preserves_shared_query_cluster_shape():
    cells = {
        "c1/modelA": {"q1": 0.1, "q2": 0.0, "q3": -0.1},
        "c1/modelB": {"q1": 0.2, "q2": 0.0, "q3": -0.2},
        "c2/modelA": {"r1": 0.1, "r2": 0.1},
    }
    a = clustered_cell_bootstrap(cells, n_resamples=1000, seed=9)
    b = clustered_cell_bootstrap(cells, n_resamples=1000, seed=9)
    assert a == b
    assert a["n_corpora"] == 2 and a["n_cells"] == 3


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError): sign_flip_mean_test([])
    with pytest.raises(ValueError): paired_bootstrap_ci([float("nan")])
    with pytest.raises(ValueError): holm_adjust({"x": 1.2})
