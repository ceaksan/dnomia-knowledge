"""Indexing pipeline: scan -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import pathspec

from dnomia_knowledge.chunker.md_chunker import MdChunker
from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.models import IndexResult
from dnomia_knowledge.store import Store

logger = logging.getLogger(__name__)

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".o",
    ".a",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".DS_Store",
}

DEFAULT_IGNORE_PATTERNS = [
    "node_modules",
    "dist",
    ".next",
    "__pycache__",
    ".git",
    "*.egg-info",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
]

MAX_FILE_SIZE_KB = 500


class Indexer:
    """Scan, chunk, embed, and store pipeline."""

    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._md_chunker = MdChunker()

    def index_file(
        self,
        project_id: str,
        project_path: str,
        file_path: str,
    ) -> int:
        """Index a single file. Returns number of chunks created."""
        content = self._read_file(file_path)
        if not content:
            return 0

        rel_path = os.path.relpath(file_path, project_path)
        chunks = self._md_chunker.chunk(file_path, content)
        if not chunks:
            return 0

        # Ensure project exists (needed when index_file is called directly)
        self.store.register_project(project_id, project_path, "content")

        # Delete old chunks for this file
        self.store.delete_file_chunks(project_id, rel_path)

        # Prepare chunk data
        chunk_dicts = [
            {
                "file_path": rel_path,
                "chunk_domain": "content",
                "chunk_type": c.chunk_type,
                "name": c.name,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
                "metadata": _serialize_metadata(c.metadata),
            }
            for c in chunks
        ]

        # Insert chunks
        chunk_ids = self.store.insert_chunks(project_id, chunk_dicts)

        # Embed and store vectors
        texts = [c.content for c in chunks]
        vectors = self.embedder.embed_passages(texts)
        self.store.insert_chunk_vectors(chunk_ids, vectors)

        # Update file index
        file_hash = _compute_file_hash(file_path)
        self.store.upsert_file_index(project_id, rel_path, file_hash, len(chunk_ids))

        return len(chunk_ids)

    def index_directory(
        self,
        project_id: str,
        directory: str,
        incremental: bool = True,
        ignore_patterns: list[str] | None = None,
    ) -> IndexResult:
        """Index all markdown files in a directory."""
        start_time = time.time()
        project_path = str(Path(directory).resolve())

        # Register project
        self.store.register_project(project_id, project_path, "content")

        # Set embedding metadata on first use
        if not self.store.get_metadata("embedding_model"):
            self.store.set_metadata("embedding_model", self.embedder.model_name)
            self.store.set_metadata("embedding_dim", str(self.embedder.dimension))

        # Scan files
        patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        all_files = self._scan_files(project_path, patterns)

        if incremental:
            stored_hashes = self.store.get_all_file_hashes(project_id)
            changed, deleted = self._detect_changes(all_files, stored_hashes, project_path)

            # Clean up deleted files
            for rel_path in deleted:
                self.store.delete_file_chunks(project_id, rel_path)
                self.store.delete_file_index(project_id, rel_path)

            files_to_index = changed
        else:
            files_to_index = all_files

        # Index changed files
        total_chunks = 0
        for file_path in files_to_index:
            try:
                count = self.index_file(project_id, project_path, file_path)
                total_chunks += count
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)

        duration = time.time() - start_time
        return IndexResult(
            project_id=project_id,
            total_files=len(all_files),
            indexed_files=len(files_to_index),
            total_chunks=total_chunks,
            content_chunks=total_chunks,
            duration_seconds=round(duration, 2),
        )

    def _scan_files(self, project_path: str, ignore_patterns: list[str]) -> list[str]:
        """Scan for markdown files, respecting .gitignore and ignore patterns."""
        root = Path(project_path)

        # Load .gitignore
        gitignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists():
            try:
                lines = gitignore_path.read_text().splitlines()
                gitignore_spec = pathspec.PathSpec.from_lines("gitignore", lines)
            except Exception:
                pass

        config_spec = pathspec.PathSpec.from_lines("gitignore", ignore_patterns)
        max_size = MAX_FILE_SIZE_KB * 1024
        files = []

        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, root)

            # Filter directories in-place
            dirnames[:] = [
                d
                for d in dirnames
                if not self._should_ignore_dir(d, rel_dir, config_spec, gitignore_spec)
            ]

            for fname in filenames:
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, root)
                ext = os.path.splitext(fname)[1].lower()

                # Skip binary
                if ext in BINARY_EXTENSIONS:
                    continue

                # Sprint 1: only markdown files
                if ext not in (".md", ".mdx"):
                    continue

                # Skip ignored
                if config_spec.match_file(rel_path):
                    continue
                if gitignore_spec and gitignore_spec.match_file(rel_path):
                    continue

                # Skip large files
                try:
                    if os.path.getsize(full_path) > max_size:
                        continue
                except OSError:
                    continue

                files.append(full_path)

        return files

    def _should_ignore_dir(
        self,
        dirname: str,
        parent_rel: str,
        config_spec: pathspec.PathSpec,
        gitignore_spec: pathspec.PathSpec | None,
    ) -> bool:
        if dirname.startswith(".") and dirname not in (".github", ".vscode", ".config"):
            return True

        rel_path = os.path.join(parent_rel, dirname) if parent_rel != "." else dirname

        if config_spec.match_file(rel_path + "/"):
            return True
        if config_spec.match_file(dirname):
            return True

        if gitignore_spec:
            if gitignore_spec.match_file(rel_path + "/"):
                return True
            if gitignore_spec.match_file(dirname + "/"):
                return True

        return False

    def _detect_changes(
        self,
        all_files: list[str],
        stored_hashes: dict[str, str],
        project_path: str,
    ) -> tuple[list[str], list[str]]:
        """Returns (changed_files, deleted_rel_paths)."""
        current_rel_paths = set()
        changed = []

        for fp in all_files:
            rel = os.path.relpath(fp, project_path)
            current_rel_paths.add(rel)
            current_hash = _compute_file_hash(fp)
            if rel not in stored_hashes or stored_hashes[rel] != current_hash:
                changed.append(fp)

        deleted = [rp for rp in stored_hashes if rp not in current_rel_paths]
        return changed, deleted

    def _read_file(self, file_path: str) -> str | None:
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return None


def _compute_file_hash(file_path: str) -> str:
    """MD5 content hash."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()  # noqa: S324
    except Exception:
        return ""


def _serialize_metadata(meta: dict | None) -> str | None:
    if meta is None:
        return None
    import json

    return json.dumps(meta, ensure_ascii=False, default=str)
