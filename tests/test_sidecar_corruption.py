"""A corrupt brute-force sidecar must degrade, not crash the process.

Audit finding C6. `BruteForceIndex.load` had no `try` at all, so three failure
modes escaped raw on **both** open paths (writable and read-only):

- `KeyError` — an .npz that is a valid zip but has no `matrix` member
- `ValueError` — an .npz that is actually pickled object data
- `zipfile.BadZipFile` — a truncated .npz

`HnswlibIndex` already wraps its equivalents in `VectorIndexError`
(vector_index.py:169-173), and `open_read_only_store` catches only
`VectorIndexError` — so the asymmetry meant a corrupt hnsw sidecar degraded to
brute force with a message on stderr, while a corrupt brute sidecar produced a
raw traceback and a non-zero CLI exit. The F31 fix converted only the hnsw
side.

The second half of C6: when the read-only path *does* fall back, it builds a
freshly-constructed empty index. Reported as `{"kind": "brute", "count": 0}`
against a live row count of N — a store that looks healthy while dense recall
silently returns nothing. That is the exact false-clean class ARCH-04 was
written to end.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pytest

from omnidb_server.config import Config
from omnidb_server.server import open_owned_store, open_read_only_store
from omnidb_server.vector_index import BruteForceIndex, VectorIndexError

SIDECAR = "tenant-default.npz"


def _seed(tmp_path: Path, n: int = 4) -> Config:
    """A home with real rows and a persisted brute sidecar."""
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    with open_owned_store(cfg) as store:
        for i in range(n):
            store.remember(f"memory number {i} about deployment")
        store.flush()
    return cfg


def _corrupt_zip_without_matrix(path: Path) -> None:
    """A structurally valid .npz whose only member is named something else."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("not_matrix", np.zeros((4, 8), dtype=np.float32))


def _pickled_payload(path: Path) -> None:
    path.write_bytes(b"\x80\x04\x95 not actually an npz at all \x00\x01\x02")


def _truncated_zip(path: Path) -> None:
    _corrupt_zip_without_matrix(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


CORRUPTIONS = {
    "zip_without_matrix": _corrupt_zip_without_matrix,
    "pickled_payload": _pickled_payload,
    "truncated_zip": _truncated_zip,
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_load_raises_vector_index_error_not_a_raw_type(tmp_path, name):
    """The contract: corruption surfaces as VectorIndexError.

    Pinned on the classmethod directly because that is the unit both open paths
    call — whichever caller forgets to catch it, the type is already right.
    """
    path = tmp_path / SIDECAR
    CORRUPTIONS[name](path)
    with pytest.raises(VectorIndexError):
        BruteForceIndex.load(path, 512)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_the_error_names_the_sidecar(tmp_path, name):
    """An operator has to know which file to delete."""
    path = tmp_path / SIDECAR
    CORRUPTIONS[name](path)
    with pytest.raises(VectorIndexError) as excinfo:
        BruteForceIndex.load(path, 512)
    assert SIDECAR in str(excinfo.value)


def test_a_dim_mismatch_still_reports_its_own_message(tmp_path):
    """The existing dim check must survive the new except clause.

    A dim mismatch is a *configuration* fault with a specific remedy (re-embed at
    the configured dim), so it must not be laundered into the generic
    "unusable sidecar" message.
    """
    path = tmp_path / SIDECAR
    np.savez(path, matrix=np.zeros((3, 256), dtype=np.float32),
             ids=np.array([1, 2, 3], dtype=np.int64))
    with pytest.raises(VectorIndexError, match="dim"):
        BruteForceIndex.load(path, 512)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_owned_open_degrades_instead_of_raising(tmp_path, name):
    """`open_owned_store` must not propagate a numpy/zip error to the operator."""
    cfg = _seed(tmp_path)
    CORRUPTIONS[name](tmp_path / SIDECAR)

    # Either it opens on a rebuilt index, or it raises VectorIndexError — what it
    # must not do is raise KeyError/ValueError/BadZipFile.
    try:
        with open_owned_store(cfg) as store:
            assert store.stats(tenant="default")["live"] == 4
    except VectorIndexError:
        pass
    except (KeyError, ValueError, zipfile.BadZipFile, sqlite3.DatabaseError) as exc:
        pytest.fail(f"raw {type(exc).__name__} escaped the owned open: {exc}")


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_read_only_open_degrades_instead_of_raising(tmp_path, name):
    cfg = _seed(tmp_path)
    CORRUPTIONS[name](tmp_path / SIDECAR)
    # open_read_only_store is a plain function, not a context manager: it opens
    # without the write lock and performs zero persistence.
    try:
        store = open_read_only_store(cfg)
    except VectorIndexError:
        pass
    except (KeyError, ValueError, zipfile.BadZipFile, sqlite3.DatabaseError) as exc:
        pytest.fail(f"raw {type(exc).__name__} escaped the read-only open: {exc}")
    else:
        assert store is not None
        store.close()


def test_the_read_only_fallback_does_not_look_healthy(tmp_path, capsys):
    """A count-0 index against live rows is a false clean, and must say so.

    The corrupt sidecar is gone, the fallback index is empty, but `live` is 4.
    A reader comparing those two numbers should not have to know the internal
    naming to spot the problem, so `stats` has to carry the signal in the
    payload rather than only on stderr.
    """
    cfg = _seed(tmp_path)
    CORRUPTIONS["zip_without_matrix"](tmp_path / SIDECAR)

    store = open_read_only_store(cfg)
    try:
        stats = store.stats(tenant="default")
    finally:
        store.close()

    assert stats["live"] == 4, "the rows are still there; only the index is gone"
    index = stats["index"]
    assert index["count"] == 0
    # The divergence itself must be visible in the payload.
    assert index["count"] != stats["live"], \
        "index.count and live disagree but the payload gives no way to tell"
    drift = stats.get("index_drift") or index.get("drift")
    assert drift, (
        "stats reports count=0 against live=4 with no drift flag; a caller "
        "cannot distinguish an empty home from a broken index"
    )
    assert "index" in capsys.readouterr().err or True  # stderr allowed, payload required


def test_an_actually_empty_home_reports_no_drift(tmp_path):
    """The drift flag must mean something, or it is noise.

    Control: no sidecar corruption, no rows — nothing to reconcile, so nothing
    to flag.
    """
    cfg = Config(home=tmp_path, embedder="hashing", vector_index="brute")
    with open_owned_store(cfg) as store:
        stats = store.stats(tenant="default")
    assert stats["live"] == 0
    assert not stats.get("index_drift"), \
        "drift flagged on a home with no rows and no sidecar"


def test_a_missing_sidecar_is_not_corruption(tmp_path):
    """Absent is not corrupt: a fresh home has no sidecar and must open clean."""
    cfg = _seed(tmp_path)
    (tmp_path / SIDECAR).unlink()
    with open_owned_store(cfg) as store:
        stats = store.stats(tenant="default")
    assert stats["live"] == 4