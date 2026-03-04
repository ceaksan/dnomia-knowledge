"""AST-aware code chunking with Tree-sitter and plain text fallback."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from dnomia_knowledge.chunker.languages import (
    classify_node,
    detect_language,
    extract_name,
    get_chunk_node_types,
)
from dnomia_knowledge.models import Chunk

logger = logging.getLogger(__name__)

# Lazy-loaded set of languages supported by tree-sitter-language-pack
_TREESITTER_SUPPORTED: set[str] | None = None

# Pre-compiled regexes for Astro script/style detection
_SCRIPT_RE = re.compile(r"<script\b[^>]*>", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style\b[^>]*>", re.IGNORECASE)

# Tree-sitter node types that represent methods inside classes
_METHOD_NODE_TYPES = frozenset(
    {
        "function_definition",
        "method_definition",
        "function_declaration",
        "method_declaration",
        "function_item",
        "constructor_declaration",
        "method",
        "singleton_method",
    }
)

# Tree-sitter node types that represent class/struct body containers
_BODY_NODE_TYPES = frozenset(
    {
        "block",
        "class_body",
        "declaration_list",
        "field_declaration_list",
    }
)

_CLASS_LIKE_TYPES = frozenset({"class", "struct", "impl"})


def _get_supported_languages() -> set[str]:
    global _TREESITTER_SUPPORTED
    if _TREESITTER_SUPPORTED is None:
        import tree_sitter_language_pack as tslp

        args = getattr(tslp.SupportedLanguage, "__args__", ())
        _TREESITTER_SUPPORTED = set(args)
    return _TREESITTER_SUPPORTED


def _has_non_blank(lines: list[str], start: int, end: int) -> bool:
    """Check if any line in the range has non-whitespace content."""
    return any(lines[i].strip() for i in range(start, min(end, len(lines))))


class AstChunker:
    """Chunk code files using Tree-sitter AST parsing.

    Falls back to sliding-window plain text chunking for unsupported
    languages or when parsing fails.
    """

    def __init__(self, max_chunk_lines: int = 50, overlap_lines: int = 2):
        if overlap_lines >= max_chunk_lines:
            raise ValueError("overlap_lines must be less than max_chunk_lines")
        self.max_chunk_lines = max_chunk_lines
        self.overlap_lines = overlap_lines

    def chunk(self, file_path: str, content: str) -> list[Chunk]:
        """Chunk a file into semantic pieces.

        Uses Tree-sitter AST parsing for supported languages,
        falls back to sliding window for unsupported ones.
        """
        if not content or not content.strip():
            return []

        language = detect_language(file_path)

        if language == "astro":
            return self._chunk_astro(content, file_path)

        if language and language in _get_supported_languages() and get_chunk_node_types(language):
            try:
                chunks = self._chunk_with_treesitter(content, language, file_path)
                if chunks:
                    return chunks
            except Exception as e:
                logger.debug(
                    "Tree-sitter parsing failed for %s: %s, using fallback",
                    file_path,
                    e,
                )

        return self._chunk_plain_text(content, language or "")

    def _chunk_with_treesitter(
        self, content: str, language: str, file_path: str = ""
    ) -> list[Chunk]:
        """Parse with Tree-sitter and extract semantic chunks."""
        import tree_sitter_language_pack as tslp

        parser = tslp.get_parser(language)
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node

        target_types = get_chunk_node_types(language)
        chunks: list[Chunk] = []
        lines = content.split("\n")

        def visit(node, depth: int = 0):
            if node.type in target_types:
                start = node.start_point[0]
                end = node.end_point[0]
                chunk_text = "\n".join(lines[start : end + 1])

                if chunk_text.strip():
                    name = extract_name(node)
                    chunk_type = classify_node(node.type)
                    node_line_count = end - start + 1

                    self._append_or_split(chunks, lines, start, end, chunk_type, name, language)

                    # For classes, extract methods only if the class was not split.
                    # When split, sub-chunks already cover method code.
                    if chunk_type in _CLASS_LIKE_TYPES and node_line_count <= self.max_chunk_lines:
                        self._extract_children(node, language, lines, chunks)
                    return

            for child in node.children:
                visit(child, depth + 1)

        visit(root)

        # If no chunks extracted (e.g. file with only top-level code),
        # treat whole file as one chunk (split if large)
        if not chunks and content.strip():
            mod_name = Path(file_path).stem if file_path else None
            if len(lines) > self.max_chunk_lines:
                chunks = self._split_large_node(lines, "module", mod_name, language, 0)
            else:
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_type="module",
                        name=mod_name,
                        start_line=1,
                        end_line=len(lines),
                        language=language,
                    )
                )

        return chunks

    def _append_or_split(
        self,
        chunks: list[Chunk],
        lines: list[str],
        start: int,
        end: int,
        chunk_type: str,
        name: str | None,
        language: str,
    ):
        """Append a node as a single chunk, or split it if it exceeds max_chunk_lines."""
        node_line_count = end - start + 1
        if node_line_count > self.max_chunk_lines:
            sub = self._split_large_node(lines[start : end + 1], chunk_type, name, language, start)
            chunks.extend(sub)
        else:
            chunk_text = "\n".join(lines[start : end + 1])
            chunks.append(
                Chunk(
                    content=chunk_text,
                    chunk_type=chunk_type,
                    name=name or None,
                    start_line=start + 1,
                    end_line=end + 1,
                    language=language,
                )
            )

    def _split_large_node(
        self,
        node_lines: list[str],
        chunk_type: str,
        name: str | None,
        language: str,
        file_start: int,
    ) -> list[Chunk]:
        """Split an oversized AST node into sub-chunks with sliding window."""
        total = len(node_lines)
        chunks: list[Chunk] = []
        start = 0
        step = self.max_chunk_lines - self.overlap_lines

        while start < total:
            remaining = total - start
            if remaining <= self.overlap_lines and chunks:
                break
            end = min(start + self.max_chunk_lines, total)
            chunk_text = "\n".join(node_lines[start:end])
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_type=chunk_type,
                        name=name or None,
                        start_line=file_start + start + 1,
                        end_line=file_start + end,
                        language=language,
                    )
                )
            start += step

        return chunks

    def _extract_children(self, parent_node, language: str, lines: list[str], chunks: list[Chunk]):
        """Extract method/function children from a class/struct/impl node."""
        for child in parent_node.children:
            if child.type in _METHOD_NODE_TYPES:
                start = child.start_point[0]
                end = child.end_point[0]
                chunk_text = "\n".join(lines[start : end + 1])
                if chunk_text.strip():
                    name = extract_name(child)
                    self._append_or_split(chunks, lines, start, end, "method", name, language)
            elif child.type in _BODY_NODE_TYPES:
                self._extract_children(child, language, lines, chunks)

    def _chunk_astro(self, content: str, file_path: str) -> list[Chunk]:
        """Chunk .astro files by structural boundaries (frontmatter, script, style, template)."""
        lines = content.split("\n")
        chunks: list[Chunk] = []

        # 1. Find frontmatter boundaries (--- ... ---)
        fm_start = None
        fm_end = None
        for i, line in enumerate(lines):
            if line.strip() == "---":
                if fm_start is None:
                    fm_start = i
                else:
                    fm_end = i
                    break

        if fm_start == 0 and fm_end is not None and fm_end > fm_start:
            if fm_end > fm_start + 1:
                self._append_or_split(
                    chunks, lines, fm_start, fm_end, "block", "frontmatter", "astro"
                )
            template_start = fm_end + 1
        else:
            template_start = 0

        # 2. Find <script> and <style> blocks
        blocks: list[tuple[int, int, str]] = []
        i = template_start
        while i < len(lines):
            line = lines[i]
            if _SCRIPT_RE.search(line):
                end = self._find_closing_tag(lines, i, "script")
                blocks.append((i, end, "script"))
                i = end + 1
            elif _STYLE_RE.search(line):
                end = self._find_closing_tag(lines, i, "style")
                blocks.append((i, end, "style"))
                i = end + 1
            else:
                i += 1

        # 3. Identify template regions (gaps between blocks)
        prev_end = template_start
        template_idx = 0
        for block_start, block_end, _ in blocks:
            if block_start > prev_end and _has_non_blank(lines, prev_end, block_start):
                template_idx += 1
                name = f"template-{template_idx}" if template_idx > 1 else "template"
                self._append_or_split(
                    chunks, lines, prev_end, block_start - 1, "block", name, "astro"
                )
            prev_end = block_end + 1

        # Trailing template after last block
        if prev_end < len(lines) and _has_non_blank(lines, prev_end, len(lines)):
            template_idx += 1
            name = f"template-{template_idx}" if template_idx > 1 else "template"
            self._append_or_split(chunks, lines, prev_end, len(lines) - 1, "block", name, "astro")

        # 4. Extract script/style blocks
        for block_start, block_end, block_type in blocks:
            if _has_non_blank(lines, block_start, block_end + 1):
                self._append_or_split(
                    chunks, lines, block_start, block_end, block_type, block_type, "astro"
                )

        chunks.sort(key=lambda c: c.start_line)

        if not chunks and content.strip():
            return self._chunk_plain_text(content, "astro")

        return chunks

    @staticmethod
    def _find_closing_tag(lines: list[str], start: int, tag: str) -> int:
        """Find the line index of the closing </tag>."""
        close = f"</{tag}>"
        for i in range(start, len(lines)):
            if close in lines[i].lower():
                return i
        logger.warning("Unclosed <%s> tag starting at line %d", tag, start + 1)
        return len(lines) - 1

    def _chunk_plain_text(self, content: str, language: str) -> list[Chunk]:
        """Sliding window chunking for unsupported languages."""
        lines = content.split("\n")
        total = len(lines)

        if total <= self.max_chunk_lines:
            return [
                Chunk(
                    content=content,
                    chunk_type="block",
                    name=None,
                    start_line=1,
                    end_line=total,
                    language=language or None,
                )
            ]

        return self._split_large_node(lines, "block", None, language or None, 0)
