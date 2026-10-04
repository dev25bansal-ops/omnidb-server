from __future__ import annotations

import hashlib
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import _EMBEDDER_SELECTORS, Config
from .db import get_meta

_TOKEN = re.compile(r"\w+")
_DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_FASTEMBED_PREFIX = "fastembed:"


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    name: str = "base"
    dim: int = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


@lru_cache(maxsize=1 << 20)
def _bucket(feature: str, dim: int) -> int:
    # md5 per unique feature is the hot loop on bulk import; corpora repeat
    # features heavily, so this cache is where the vectorization win lives
    return int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16) % dim


class HashingEmbedder(Embedder):
    """Deterministic feature-hashing embedder.

    Offline fallback: exact-term and n-gram overlap works, paraphrase does not.
    Install the `semantic` extra (fastembed) for real semantics. Vector math is
    identical to v1 (unique-feature counts, L2-normalized) — only faster.
    """

    def __init__(self, dim: int = 512):
        self.name = "hashing-v1"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            words = _TOKEN.findall(lowered)
            features: set[str] = set(words)
            features |= {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}
            features |= {lowered[i : i + 3] for i in range(max(0, len(lowered) - 2))}
            if not features:
                continue
            ordered = list(features)
            buckets = np.fromiter(
                (_bucket(f, self.dim) for f in ordered),
                dtype=np.intp,
                count=len(ordered),
            )
            np.add.at(out[row], buckets, 1.0)
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class FastEmbedEmbedder(Embedder):
    def __init__(self, model_name: str | None = _DEFAULT_FASTEMBED_MODEL):
        from fastembed import TextEmbedding

        model_name = model_name or _DEFAULT_FASTEMBED_MODEL
        self._model = TextEmbedding(model_name=model_name)
        self.name = f"{_FASTEMBED_PREFIX}{model_name}"
        self.dim = int(np.asarray(next(iter(self._model.embed(["dim probe"])))).shape[0])

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


def _load_fastembed(model_name: str) -> FastEmbedEmbedder:
    from fastembed import TextEmbedding

    from .store import E_VECTOR_DIM_MISMATCH, StoreError

    try:
        supported = {record["model"] for record in TextEmbedding.list_supported_models()}
    except Exception as exc:
        raise EmbeddingError(
            f"fastembed model catalog is unavailable; install the `semantic` extra: {exc}"
        ) from exc
    if model_name not in supported:
        raise StoreError(
            f"home pins unknown fastembed model {model_name!r}; this fastembed "
            "installation does not support it",
            E_VECTOR_DIM_MISMATCH,
        )
    try:
        embedder = FastEmbedEmbedder(model_name=model_name)
    except ImportError as exc:
        raise EmbeddingError(
            f"fastembed is unavailable; install the `semantic` extra: {exc}"
        ) from exc
    except Exception as exc:
        raise EmbeddingError(
            f"fastembed model {model_name!r} could not be loaded "
            f"(check network / HF_HOME cache path): {exc}"
        ) from exc
    if embedder.name != f"{_FASTEMBED_PREFIX}{model_name}":
        raise StoreError(
            f"requested fastembed model {model_name!r} but loaded "
            f"embedder identity {embedder.name!r}",
            E_VECTOR_DIM_MISMATCH,
        )
    return embedder


def embedder_from_name(name: str) -> Embedder:
    from .store import E_VECTOR_DIM_MISMATCH, StoreError

    if name == "hashing-v1":
        return HashingEmbedder()
    if not isinstance(name, str) or not name.startswith(_FASTEMBED_PREFIX):
        raise StoreError(
            f"home pins unknown embedder {name!r}; expected hashing-v1 or "
            f"fastembed:<model>",
            E_VECTOR_DIM_MISMATCH,
        )
    model_name = name[len(_FASTEMBED_PREFIX):]
    if not model_name:
        raise StoreError(
            "home pins an empty fastembed model identity; re-embed is required",
            E_VECTOR_DIM_MISMATCH,
        )
    return _load_fastembed(model_name)


def _stored_embedder_name(conn: sqlite3.Connection | None, home: str) -> str | None:
    if conn is not None:
        return get_meta(conn, "embedder")
    db_path = Path(home) / "omnidb.db"
    if not db_path.exists():
        return None
    with sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True) as metadata_conn:
        return get_meta(metadata_conn, "embedder")


def make_embedder(cfg: Config, conn: sqlite3.Connection | None = None,
                  stored_name: str | None = None) -> Embedder:
    from .config import validate_selector
    from .store import E_VECTOR_DIM_MISMATCH, StoreError

    preference = validate_selector("embedder", cfg.embedder or "auto", _EMBEDDER_SELECTORS)
    if preference == "hashing":
        return HashingEmbedder()
    if stored_name is None:
        stored_name = _stored_embedder_name(conn, str(cfg.home))
    if preference == "auto":
        if stored_name is None:
            return HashingEmbedder()
        return embedder_from_name(stored_name)
    if stored_name is None or stored_name == "hashing-v1":
        return _load_fastembed(_DEFAULT_FASTEMBED_MODEL)
    if not isinstance(stored_name, str) or not stored_name.startswith(_FASTEMBED_PREFIX):
        raise StoreError(
            f"home pins embedder {stored_name!r}, but fastembed is configured; "
            f"re-embed is required to change it",
            E_VECTOR_DIM_MISMATCH,
        )
    return embedder_from_name(stored_name)
