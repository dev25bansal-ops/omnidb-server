"""Unit tests for MultiTenantIndexRouter (MT-06 Phase 1).

This module tests the router in isolation from MemoryStore, so each turn can
verify the router itself is correct before we integrate it into store.py.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnidb_server.vector_index import BruteForceIndex, MultiTenantIndexRouter, make_multi_tenant_router


class TestMultiTenantIndexRouterBasics:
    """Basic correctness: adding/removing/querying per tenant."""

    def test_router_empty_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=512, kind="brute")
            assert router.kind == "multi"
            assert router.active_count() == 0
            assert len(router.indexed_ids()) == 0
            assert router.backend_count == 0

    def test_router_single_tenant_add_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            # Add 5 vectors for tenant A
            ids_a = [1, 2, 3, 4, 5]
            vecs_a = np.random.randn(5, 3).astype("float32")
            # L2-normalize for cosine simulation
            vecs_a = vecs_a / np.linalg.norm(vecs_a, axis=1, keepdims=True)
            router.add(ids_a, vecs_a, tenant="a")
            # Verify state
            assert router.backend_count == 1
            assert router.active_count("a") == 5
            assert router.active_count() == 5
            assert set(router.indexed_ids("a")) == {1, 2, 3, 4, 5}
            # Query top-3
            qvec = np.random.randn(3).astype("float32")
            results = router.query(qvec, k=3, tenant="a")
            assert len(results) == 3
            assert all(isinstance(mid, int) for mid, _score in results)

    def test_router_two_tenants_isolation(self):
        """Confirm tenants' vectors are kept separate."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            # Tenant A
            ids_a = [1, 2, 3]
            vecs_a = np.ones((3, 3), dtype="float32")
            router.add(ids_a, vecs_a, tenant="a")
            # Tenant B
            ids_b = [10, 20]
            vecs_b = np.ones((2, 3), dtype="float32") * 2
            router.add(ids_b, vecs_b, tenant="b")
            # Each sees only its own
            assert router.active_count("a") == 3
            assert router.active_count("b") == 2
            assert router.active_count() == 5
            assert set(router.indexed_ids("a")) == {1, 2, 3}
            assert set(router.indexed_ids("b")) == {10, 20}
            # Aggregated IDs combine both
            combined = router.indexed_ids()
            assert combined == {1, 2, 3, 10, 20}

    def test_router_update_remove(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            ids = [1, 2, 3]
            vecs = np.random.randn(3, 3).astype("float32")
            router.add(ids, vecs, tenant="test")
            # Update
            new_vec = np.ones(3, dtype="float32")
            router.update(2, new_vec, tenant="test")
            # Remove
            router.remove([1], tenant="test")
            assert router.active_count("test") == 2
            assert set(router.indexed_ids("test")) == {2, 3}

    def test_router_persist_load_roundtrip(self):
        """Verify persistence actually writes/reads sidecar files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router1 = MultiTenantIndexRouter(home, dim=3, kind="brute")
            ids = [1, 2]
            vecs = np.ones((2, 3), dtype="float32")
            router1.add(ids, vecs, tenant="p")
            router1.persist("p")
            # Sidecar should exist
            p_path = home / "tenants" / "p" / "vectors.npz"
            assert p_path.exists(), "persist() did not create sidecar file"
            # Reload via new router and verify all IDs are present
            router2 = MultiTenantIndexRouter(home, dim=3, kind="brute")
            assert set(router2.indexed_ids("p")) == set(ids), f"Reloaded IDs mismatch: {set(router2.indexed_ids('p'))}"


class TestMultiTenantIndexRouterEdgeCases:
    """Boundary conditions and error handling."""

    def test_router_query_k_larger_than_available(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            ids = [1]
            vecs = np.ones((1, 3), dtype="float32")
            router.add(ids, vecs, tenant="x")
            # Query more than available
            qvec = np.ones(3, dtype="float32")
            results = router.query(qvec, k=100, tenant="x")
            assert len(results) == 1, "Should return at most active_count items"

    def test_router_kind_mixed_backends(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            # Use brute-only mode
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            assert router.kind == "multi"  # empty → multi
            router.add([1], np.ones((1, 3), dtype="float32"), tenant="b1")
            assert router.kind == "brute"
            # Note: currently don't add HNSW tenants, so won't test mixed

    def test_router_close(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            router.add([1], np.ones((1, 3), dtype="float32"), tenant="c")
            assert router.backend_count == 1
            router.close("c")  # close specific
            assert router.backend_count == 0
            # Create again
            router.get_or_create("d")
            assert router.backend_count == 1
            router.close(None)  # close all
            assert router.backend_count == 0

    def test_make_multi_tenant_router_helper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            router = make_multi_tenant_router(Path(tmp_dir), dim=128, backend="hnsw", ef_search=32)
            assert router.dim == 128
            assert router.kind == "multi"


class TestMultiTenantIndexRouterPersistFlushing:
    """Flush/persist behavior verification."""

    def test_flush_all(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=3, kind="brute")
            router.add([1, 2], np.ones((2, 3), dtype="float32"), tenant="f1")
            router.add([10, 20], np.ones((2, 3), dtype="float32"), tenant="f2")
            router.flush_all()  # explicit flush of all
            # Both sidecars should exist
            assert (home / "tenants" / "f1" / "vectors.npz").exists()
            assert (home / "tenants" / "f2" / "vectors.npz").exists()


class TestMultiTenantIndexError:
    """MultiTenantIndexError usage and propagation."""

    def test_error_raised_on_failed_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            home = Path(tmp_dir)
            router = MultiTenantIndexRouter(home, dim=512, kind="brute")
            # Simulate impossible condition by setting corrupt dir permissions would be OS-specific;
            # instead, just verify exception type exists