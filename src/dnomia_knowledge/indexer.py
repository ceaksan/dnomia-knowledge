"""Indexing pipeline: scan -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import pathspec

from dnomia_knowledge.chunker.ast_chunker import AstChunker
from dnomia_knowledge.chunker.md_chunker import MdChunker
from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.models import Chunk, IndexResult
from dnomia_knowledge.registry import ProjectConfig, compute_config_hash
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
        config: ProjectConfig | None = None,
    ) -> tuple[int, list[int]]:
        """Index a single file. Returns (chunk_count, chunk_ids)."""
        content = self._read_file(file_path)
        if not content:
            return 0, []

        rel_path = os.path.relpath(file_path, project_path)
        ext = Path(file_path).suffix.lower()

        content_exts = set(config.content.extensions) if config else {".md", ".mdx"}
        code_exts = set(config.code.resolved_extensions) if config else set()

        if ext in content_exts:
            chunks = self._md_chunker.chunk(file_path, content)
            domain = "content"
        elif ext in code_exts:
            max_lines = config.code.max_chunk_lines if config else 50
            ast_chunker = AstChunker(max_chunk_lines=max_lines, overlap_lines=2)
            chunks = ast_chunker.chunk(file_path, content)
            domain = "code"
        else:
            return 0, []

        if not chunks:
            return 0, []

        # Ensure project exists (needed when index_file is called directly)
        project_type = config.type if config else "content"
        graph_enabled = config.graph.enabled if config else False
        self.store.register_project(
            project_id,
            project_path,
            project_type,
            graph_enabled=graph_enabled,
        )

        # Delete old chunks for this file
        self.store.delete_file_chunks(project_id, rel_path)

        # Prepare chunk data
        chunk_dicts = [
            {
                "file_path": rel_path,
                "chunk_domain": domain,
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

        return len(chunk_ids), chunk_ids

    def index_directory(
        self,
        project_id: str,
        directory: str,
        incremental: bool = True,
        ignore_patterns: list[str] | None = None,
        config: ProjectConfig | None = None,
    ) -> IndexResult:
        """Index files in a directory. Config-driven when config is provided."""
        start_time = time.time()
        project_path = str(Path(directory).resolve())

        # Determine ignore patterns: explicit param > config > defaults
        if ignore_patterns is not None:
            patterns = ignore_patterns
        elif config:
            patterns = config.indexing.ignore_patterns
        else:
            patterns = DEFAULT_IGNORE_PATTERNS

        # Determine max file size
        max_file_size_kb = config.indexing.max_file_size_kb if config else MAX_FILE_SIZE_KB

        # Config hash
        config_hash = None
        if config:
            toml_path = Path(directory) / ".knowledge.toml"
            if toml_path.exists():
                config_hash = compute_config_hash(toml_path)

        # Register project
        project_type = config.type if config else "content"
        graph_enabled = config.graph.enabled if config else False
        self.store.register_project(
            project_id,
            project_path,
            project_type,
            graph_enabled=graph_enabled,
            config_hash=config_hash,
        )

        # Set embedding metadata on first use
        if not self.store.get_metadata("embedding_model"):
            self.store.set_metadata("embedding_model", self.embedder.model_name)
            self.store.set_metadata("embedding_dim", str(self.embedder.dimension))

        # Build extension sets for scanning
        content_exts = set(config.content.extensions) if config else {".md", ".mdx"}
        code_exts = set(config.code.resolved_extensions) if config else set()
        all_exts = content_exts | code_exts

        # Build path scoping
        content_paths = config.content.paths if config else []
        code_paths = config.code.paths if config else []

        # Scan files
        all_files = self._scan_files(
            project_path,
            patterns,
            allowed_extensions=all_exts,
            content_paths=content_paths,
            code_paths=code_paths,
            content_exts=content_exts,
            code_exts=code_exts,
            max_file_size_kb=max_file_size_kb,
        )

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
        content_chunks = 0
        code_chunks = 0
        all_new_chunk_ids: list[int] = []
        file_chunk_map: dict[str, list[int]] = {}

        for file_path in files_to_index:
            try:
                ext = Path(file_path).suffix.lower()
                count, chunk_ids = self.index_file(project_id, project_path, file_path, config)
                if ext in content_exts:
                    content_chunks += count
                else:
                    code_chunks += count
                total_chunks += count
                all_new_chunk_ids.extend(chunk_ids)
                rel = os.path.relpath(file_path, project_path)
                file_chunk_map[rel] = chunk_ids
            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)

        # Build graph edges if enabled
        if graph_enabled and all_new_chunk_ids:
            try:
                from dnomia_knowledge.graph import GraphBuilder

                graph_config = config.graph if config else None
                builder = GraphBuilder(self.store, graph_config)
                for rel_path, cids in file_chunk_map.items():
                    builder.build_edges_for_file(project_id, rel_path, cids)
                builder.build_semantic_edges(project_id, all_new_chunk_ids)
            except Exception as e:
                logger.warning("Graph building failed: %s", e)

        duration = time.time() - start_time
        return IndexResult(
            project_id=project_id,
            total_files=len(all_files),
            indexed_files=len(files_to_index),
            total_chunks=total_chunks,
            content_chunks=content_chunks,
            code_chunks=code_chunks,
            duration_seconds=round(duration, 2),
        )

    def _scan_files(
        self,
        project_path: str,
        ignore_patterns: list[str],
        allowed_extensions: set[str] | None = None,
        content_paths: list[str] | None = None,
        code_paths: list[str] | None = None,
        content_exts: set[str] | None = None,
        code_exts: set[str] | None = None,
        max_file_size_kb: int = MAX_FILE_SIZE_KB,
    ) -> list[str]:
        """Scan for files, respecting .gitignore, ignore patterns, and path scoping."""
        root = Path(project_path)

        # Defaults for backward compat
        if allowed_extensions is None:
            allowed_extensions = {".md", ".mdx"}
        if content_exts is None:
            content_exts = {".md", ".mdx"}
        if code_exts is None:
            code_exts = set()
        if content_paths is None:
            content_paths = []
        if code_paths is None:
            code_paths = []

        # Normalize path scoping: strip trailing slashes for consistent matching
        norm_content_paths = [p.rstrip("/") for p in content_paths]
        norm_code_paths = [p.rstrip("/") for p in code_paths]

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
        max_size = max_file_size_kb * 1024
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

                # Check extension is allowed
                if ext not in allowed_extensions:
                    continue

                # Path scoping: if paths are configured, enforce them
                if not self._file_in_scope(
                    rel_path,
                    ext,
                    content_exts,
                    code_exts,
                    norm_content_paths,
                    norm_code_paths,
                ):
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

    def _file_in_scope(
        self,
        rel_path: str,
        ext: str,
        content_exts: set[str],
        code_exts: set[str],
        content_paths: list[str],
        code_paths: list[str],
    ) -> bool:
        """Check if a file is within the configured path scope."""
        is_content = ext in content_exts
        is_code = ext in code_exts

        if is_content and content_paths:
            if not any(
                rel_path.startswith(p + os.sep) or rel_path.startswith(p + "/")
                for p in content_paths
            ):
                return False

        if is_code and code_paths:
            if not any(
                rel_path.startswith(p + os.sep) or rel_path.startswith(p + "/") for p in code_paths
            ):
                return False

        return is_content or is_code

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

    def _raw_chunk(
        self,
        file_path: str,
        content: str,
        config: ProjectConfig | None = None,
    ) -> list[Chunk]:
        """Simple raw text chunking for code files. Sprint 3 replaces with AST."""
        max_lines = config.code.max_chunk_lines if config else 50
        ext = Path(file_path).suffix.lstrip(".")
        lines = content.split("\n")

        if len(lines) <= max_lines:
            return [
                Chunk(
                    content=content,
                    chunk_type="block",
                    name=Path(file_path).stem,
                    language=ext,
                    start_line=1,
                    end_line=len(lines),
                )
            ]

        # Split into blocks
        chunks = []
        for i in range(0, len(lines), max_lines):
            block_lines = lines[i : i + max_lines]
            chunks.append(
                Chunk(
                    content="\n".join(block_lines),
                    chunk_type="block",
                    name=f"{Path(file_path).stem}:{i + 1}",
                    language=ext,
                    start_line=i + 1,
                    end_line=min(i + max_lines, len(lines)),
                )
            )
        return chunks

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
