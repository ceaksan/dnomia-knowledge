"""AST-aware code chunking with Tree-sitter and plain text fallback."""

from __future__ import annotations

import logging
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


def _get_supported_languages() -> set[str]:
    global _TREESITTER_SUPPORTED
    if _TREESITTER_SUPPORTED is None:
        import tree_sitter_language_pack as tslp

        args = getattr(tslp.SupportedLanguage, "__args__", ())
        _TREESITTER_SUPPORTED = set(args)
    return _TREESITTER_SUPPORTED


class AstChunker:
    """Chunk code files using Tree-sitter AST parsing.

    Falls back to sliding-window plain text chunking for unsupported
    languages or when parsing fails.
    """

    def __init__(self, max_chunk_lines: int = 50, overlap_lines: int = 2):
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

                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            chunk_type=chunk_type,
                            name=name or None,
                            start_line=start + 1,  # 1-indexed
                            end_line=end + 1,
                            language=language,
                        )
                    )

                    # For classes, also extract methods inside
                    if chunk_type in ("class", "struct", "impl"):
                        self._extract_children(node, language, lines, chunks)
                    return

            for child in node.children:
                visit(child, depth + 1)

        visit(root)

        # If no chunks extracted (e.g. file with only top-level code),
        # treat whole file as one chunk
        if not chunks and content.strip():
            chunks.append(
                Chunk(
                    content=content,
                    chunk_type="module",
                    name=Path(file_path).stem if file_path else None,
                    start_line=1,
                    end_line=len(lines),
                    language=language,
                )
            )

        return chunks

    def _extract_children(self, parent_node, language: str, lines: list[str], chunks: list[Chunk]):
        """Extract method/function children from a class/struct/impl node."""
        method_types = {
            "function_definition",
            "method_definition",
            "function_declaration",
            "method_declaration",
            "function_item",
            "constructor_declaration",
            "method",
            "singleton_method",
        }

        for child in parent_node.children:
            if child.type in method_types:
                start = child.start_point[0]
                end = child.end_point[0]
                chunk_text = "\n".join(lines[start : end + 1])
                if chunk_text.strip():
                    name = extract_name(child)
                    chunks.append(
                        Chunk(
                            content=chunk_text,
                            chunk_type="method",
                            name=name or None,
                            start_line=start + 1,
                            end_line=end + 1,
                            language=language,
                        )
                    )
            # Recurse into body nodes (e.g. class body, block)
            elif child.type in (
                "block",
                "class_body",
                "declaration_list",
                "field_declaration_list",
            ):
                self._extract_children(child, language, lines, chunks)

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

        chunks: list[Chunk] = []
        start = 0
        step = self.max_chunk_lines - self.overlap_lines

        while start < total:
            end = min(start + self.max_chunk_lines, total)
            chunk_text = "\n".join(lines[start:end])
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_type="block",
                        name=None,
                        start_line=start + 1,
                        end_line=end,
                        language=language or None,
                    )
                )
            start += step

        return chunks
