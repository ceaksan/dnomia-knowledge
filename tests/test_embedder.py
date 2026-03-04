"""Tests for embedding module."""

from __future__ import annotations

import pytest

from dnomia_knowledge.embedder import Embedder


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def setup(self, shared_embedder):
        self.embedder = shared_embedder

    def test_dimension(self):
        assert self.embedder.dimension == 768

    def test_embed_query_returns_list(self):
        result = self.embedder.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_embed_passage_returns_list(self):
        result = self.embedder.embed_passage("test passage")
        assert isinstance(result, list)
        assert len(result) == 768

    def test_embed_passages_batch(self):
        texts = ["passage one", "passage two", "passage three"]
        results = self.embedder.embed_passages(texts)
        assert len(results) == 3
        assert all(len(v) == 768 for v in results)

    def test_query_and_passage_differ(self):
        """query: prefix and passage: prefix should produce different embeddings."""
        q = self.embedder.embed_query("python testing")
        p = self.embedder.embed_passage("python testing")
        assert q != p

    def test_query_cache(self):
        """Same query should return cached result."""
        r1 = self.embedder.embed_query("cache test")
        r2 = self.embedder.embed_query("cache test")
        assert r1 == r2

    def test_lazy_loading(self):
        """Model should not be loaded until first embed call."""
        fresh = Embedder()
        assert fresh._model is None
        fresh.embed_query("trigger load")
        assert fresh._model is not None

    def test_unload(self):
        """Model should be unloadable."""
        fresh = Embedder()
        fresh.embed_query("trigger load")
        assert fresh._model is not None
        fresh.unload()
        assert fresh._model is None
