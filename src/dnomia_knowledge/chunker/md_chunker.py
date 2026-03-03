"""Heading-based markdown/MDX chunker with frontmatter support."""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from dnomia_knowledge.models import Chunk

# Heading pattern: ## and ### only (# is title, #### too granular)
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

# Default min chars for a chunk (below this, merge into previous)
_DEFAULT_MIN_CHUNK_CHARS = 50


class MdChunker:
    """Chunk markdown/MDX files by heading boundaries."""

    def __init__(
        self,
        overlap_lines: int = 0,
        min_chunk_chars: int = _DEFAULT_MIN_CHUNK_CHARS,
    ):
        self.overlap_lines = overlap_lines
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        if not content or not content.strip():
            return []

        ext = Path(file_path).suffix.lstrip(".")
        language = ext if ext in ("md", "mdx") else "md"

        # Parse frontmatter
        fm_meta = self._parse_frontmatter(content)
        body = self._strip_frontmatter(content)

        if not body.strip():
            return []

        # Find heading positions
        lines = body.split("\n")
        sections = self._split_by_headings(lines)

        if not sections:
            return [
                Chunk(
                    content=body.strip(),
                    chunk_type="heading",
                    name=None,
                    language=language,
                    start_line=1,
                    end_line=len(content.split("\n")),
                    metadata=fm_meta,
                )
            ]

        # Merge small sections
        merged = self._merge_small(sections)

        # Apply overlap
        if self.overlap_lines > 0:
            merged = self._apply_overlap(merged, lines)

        # Calculate frontmatter offset (lines before body starts)
        fm_lines = len(content.split("\n")) - len(lines)

        chunks = []
        for section in merged:
            chunks.append(
                Chunk(
                    content=section["text"].strip(),
                    chunk_type="heading",
                    name=section.get("heading"),
                    language=language,
                    start_line=section["start_line"] + fm_lines,
                    end_line=section["end_line"] + fm_lines,
                    metadata=fm_meta,
                )
            )
        return chunks

    def _parse_frontmatter(self, content: str) -> dict | None:
        try:
            post = frontmatter.loads(content)
            meta = dict(post.metadata)
            if not meta:
                return None
            # Keep only useful fields
            result = {}
            for key in (
                "title",
                "description",
                "tags",
                "categories",
                "slug",
                "tldr",
                "keyTakeaways",
                "faq",
                "pubDate",
                "lang",
            ):
                if key in meta:
                    val = meta[key]
                    if isinstance(val, (list, dict, str, int, float, bool)):
                        result[key] = val
            return result if result else None
        except Exception:
            return None

    def _strip_frontmatter(self, content: str) -> str:
        try:
            post = frontmatter.loads(content)
            return post.content
        except Exception:
            return content

    def _split_by_headings(self, lines: list[str]) -> list[dict]:
        sections: list[dict] = []
        current: dict | None = None

        for i, line in enumerate(lines):
            match = _HEADING_RE.match(line)
            if match:
                if current is not None:
                    current["end_line"] = i  # exclusive
                    current["text"] = "\n".join(lines[current["start_line"] : i])
                    sections.append(current)
                current = {
                    "heading": match.group(2).strip(),
                    "level": len(match.group(1)),
                    "start_line": i + 1,  # 1-indexed
                    "end_line": 0,
                    "text": "",
                }
            elif current is None and line.strip():
                # Content before first heading
                current = {
                    "heading": None,
                    "level": 0,
                    "start_line": 1,
                    "end_line": 0,
                    "text": "",
                }

        # Close last section
        if current is not None:
            current["end_line"] = len(lines)
            current["text"] = "\n".join(lines[current["start_line"] - 1 :])
            sections.append(current)

        return sections

    def _merge_small(self, sections: list[dict]) -> list[dict]:
        if not sections:
            return sections

        merged: list[dict] = [sections[0]]
        for section in sections[1:]:
            if len(section["text"].strip()) < self.min_chunk_chars:
                # Merge into previous
                prev = merged[-1]
                prev["text"] = prev["text"] + "\n\n" + section["text"]
                prev["end_line"] = section["end_line"]
            else:
                merged.append(section)

        return merged

    def _apply_overlap(self, sections: list[dict], all_lines: list[str]) -> list[dict]:
        for i in range(1, len(sections)):
            prev = sections[i - 1]
            prev_end = prev["end_line"]
            overlap_start = max(prev_end - self.overlap_lines, prev["start_line"])
            overlap_text = "\n".join(all_lines[overlap_start - 1 : prev_end])
            sections[i]["text"] = overlap_text + "\n\n" + sections[i]["text"]
            sections[i]["start_line"] = overlap_start
        return sections
