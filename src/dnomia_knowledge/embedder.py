"""Embedding module using multilingual-e5-base with query/passage prefix."""

from __future__ import annotations

import gc
import hashlib
import time
from typing import Any

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_DIMENSION = 768


class Embedder:
    """Lazy-loading embedding with query/passage prefix and caching."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_ttl_minutes: int = 15,
    ):
        self.model_name = model_name
        self.dimension = DEFAULT_DIMENSION
        self._model: Any = None
        self._last_used: float = 0
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._cache_ttl = cache_ttl_minutes * 60

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._last_used = time.time()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        self._last_used = time.time()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        cache_key = hashlib.md5(f"q:{text}".encode()).hexdigest()
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        result = self._encode([f"query: {text}"])[0]
        self._set_cached(cache_key, result)
        return result

    def embed_passage(self, text: str) -> list[float]:
        return self._encode([f"passage: {text}"])[0]

    def embed_passages(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        all_results: list[list[float]] = []
        prefixed = [f"passage: {t}" for t in texts]
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            all_results.extend(self._encode(batch))
        return all_results

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def maybe_unload(self, idle_minutes: int = 10) -> None:
        if self._model and time.time() - self._last_used > idle_minutes * 60:
            self.unload()

    def _get_cached(self, key: str) -> list[float] | None:
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: list[float]) -> None:
        now = time.time()
        self._cache[key] = (value, now)
        # Evict expired entries
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self._cache_ttl]
        for k in expired:
            del self._cache[k]
