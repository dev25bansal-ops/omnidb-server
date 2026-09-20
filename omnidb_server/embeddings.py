from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from .config import Config

_TOKEN = re.compile(r"\w+")


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
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        self.dim = int(next(iter(self._model.embed(["dim probe"]))).shape[0])

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype=np.float32)


def make_embedder(cfg: Config) -> Embedder:
    preference = (cfg.embedder or "auto").lower()
    if preference == "fastembed":
        try:
            return FastEmbedEmbedder()
        except Exception as exc:
            raise EmbeddingError(
                f"OMNIDB_EMBEDDER=fastembed but fastembed is unavailable "
                f"(check network / HF_HOME cache path): {exc}"
            ) from exc
    # "auto" and "hashing" are offline-safe by design: a bare `stats` call must
    # never trigger an implicit ~100MB model download. Real semantics require
    # opting in with OMNIDB_EMBEDDER=fastembed.
    #
    # J-14: previously this path was "best-effort offline" — it tried
    # fastembed first and silently downgraded to hashing on any failure
    # (including a half-cached broken model). Now we attempt fastembed only
    # when the user explicitly asked for it; auto/hashing get hashing
    # with a single, loud rejection of fastembed if it is requested but
    # unavailable.
    return HashingEmbedder()
