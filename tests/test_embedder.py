"""Tests for embedding module."""

from __future__ import annotations

import pytest

from dnomia_knowledge.embedder import Embedder


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.embedder = Embedder()

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
        embedder = Embedder()
        assert embedder._model is None
        embedder.embed_query("trigger load")
        assert embedder._model is not None

    def test_unload(self):
        """Model should be unloadable."""
        self.embedder.embed_query("trigger load")
        assert self.embedder._model is not None
        self.embedder.unload()
        assert self.embedder._model is None
