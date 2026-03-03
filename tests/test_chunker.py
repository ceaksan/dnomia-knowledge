"""Tests for markdown chunker."""

from __future__ import annotations

import pytest

from dnomia_knowledge.chunker.md_chunker import MdChunker
from dnomia_knowledge.models import Chunk


class TestMdChunker:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = MdChunker()

    def test_basic_heading_split(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_types_are_heading(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.chunk_type == "heading"

    def test_heading_names_extracted(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        names = [c.name for c in chunks]
        assert "Introduction" in names
        assert "Writing Tests" in names

    def test_frontmatter_in_metadata(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.metadata is not None
            assert "title" in c.metadata
            assert c.metadata["title"] == "Test Article"
            assert "tags" in c.metadata

    def test_code_blocks_preserved(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        code_found = any("def test_example" in c.content for c in chunks)
        assert code_found

    def test_small_chunks_merged(self):
        content = """---
title: Test
---

## Section A

Tiny.

## Section B

Also tiny.

## Section C

This section has enough content to stand alone. It contains multiple sentences
and paragraphs of text that make it a reasonable standalone chunk for the
knowledge system to index and search through.
"""
        chunker = MdChunker(min_chunk_chars=100)
        chunks = chunker.chunk("test.md", content)
        # Tiny sections should be merged
        assert len(chunks) < 3

    def test_line_numbers(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.start_line > 0
            assert c.end_line >= c.start_line

    def test_language_set_to_md(self, sample_markdown):
        chunks = self.chunker.chunk("test.md", sample_markdown)
        for c in chunks:
            assert c.language == "md"

    def test_mdx_language(self):
        chunks = self.chunker.chunk("test.mdx", "## Hello\n\nWorld")
        assert all(c.language == "mdx" for c in chunks)

    def test_empty_content(self):
        chunks = self.chunker.chunk("test.md", "")
        assert chunks == []

    def test_no_headings(self):
        content = "Just plain text without any headings."
        chunks = self.chunker.chunk("test.md", content)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "heading"
        assert chunks[0].name is None

    def test_overlap(self):
        content = """---
title: Test
---

## Section A

Line 1 of section A.
Line 2 of section A.
Last line of section A.

## Section B

Line 1 of section B.
"""
        chunker = MdChunker(overlap_lines=2)
        chunks = chunker.chunk("test.md", content)
        if len(chunks) >= 2:
            # Second chunk should contain overlap from first
            # (last 2 lines of section A should appear in section B's chunk)
            assert chunks[1].start_line <= chunks[0].end_line
