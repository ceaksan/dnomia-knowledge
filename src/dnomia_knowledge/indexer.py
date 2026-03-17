"""Indexing pipeline: scan -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import subprocess
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
        self._ast_chunkers: dict[int, AstChunker] = {}

    def _get_ast_chunker(self, max_lines: int) -> AstChunker:
        """Return a cached AstChunker for the given max_lines."""
        if max_lines not in self._ast_chunkers:
            self._ast_chunkers[max_lines] = AstChunker(max_chunk_lines=max_lines, overlap_lines=2)
        return self._ast_chunkers[max_lines]

    def index_file(
        self,
        project_id: str,
        project_path: str,
        file_path: str,
        config: ProjectConfig | None = None,
        file_hash: str | None = None,
        _ensure_project: bool = True,
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
            chunks = self._get_ast_chunker(max_lines).chunk(file_path, content)
            domain = "code"
        else:
            return 0, []

        if not chunks:
            return 0, []

        # Ensure project exists (needed when index_file is called directly)
        if _ensure_project:
            project_type = config.type if config else "content"
            graph_enabled = config.graph.enabled if config else False
            self.store.register_project(
                project_id,
                project_path,
                project_type,
                graph_enabled=graph_enabled,
            )

        # All DB ops in a single transaction
        self.store.delete_file_chunks(project_id, rel_path, commit=False)

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

        chunk_ids = self.store.insert_chunks(project_id, chunk_dicts, commit=False)

        texts = [c.content for c in chunks]
        vectors = self.embedder.embed_passages(texts)
        self.store.insert_chunk_vectors(chunk_ids, vectors, commit=False)

        if file_hash is None:
            file_hash = _compute_file_hash(file_path)
        self.store.upsert_file_index(project_id, rel_path, file_hash, len(chunk_ids), commit=False)

        self.store.commit()
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

        # Scan files
        all_files = self._scan_files(project_path, patterns, config)

        if incremental:
            stored_hashes = self.store.get_all_file_hashes(project_id)
            changed, deleted = self._detect_changes(all_files, stored_hashes, project_path)

            # Clean up deleted files
            for rel_path in deleted:
                self.store.delete_file_chunks(project_id, rel_path)
                self.store.delete_file_index(project_id, rel_path)

            files_to_index = changed
        else:
            files_to_index = [(fp, None) for fp in all_files]

        # Index changed files
        content_exts = set(config.content.extensions) if config else {".md", ".mdx"}
        total_chunks = 0
        content_chunks = 0
        code_chunks = 0
        all_new_chunk_ids: list[int] = []
        file_chunk_map: dict[str, list[int]] = {}

        for file_path, file_hash in files_to_index:
            try:
                ext = Path(file_path).suffix.lower()
                count, chunk_ids = self.index_file(
                    project_id,
                    project_path,
                    file_path,
                    config,
                    file_hash=file_hash,
                    _ensure_project=False,
                )
                if ext in content_exts:
                    content_chunks += count
                else:
                    code_chunks += count
                total_chunks += count
                all_new_chunk_ids.extend(chunk_ids)
                rel = os.path.relpath(file_path, project_path)
                file_chunk_map[rel] = chunk_ids
            except (OSError, sqlite3.Error, ValueError) as e:
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
            except (ImportError, sqlite3.Error, ValueError) as e:
                logger.warning("Graph building failed: %s", e)

        # Update last_indexed
        commit_hash = _get_git_head(project_path)
        self.store.update_project_last_indexed(project_id, commit_hash=commit_hash)

        duration = time.time() - start_time

        # Incremental git sync (if project has been git-synced before)
        try:
            from .git_sync import GitSync
            git_sync = GitSync(self.store)
            git_sync.sync_incremental(project_id, project_path)
        except Exception as e:
            logger.warning("Git sync skipped: %s", e)

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
        config: ProjectConfig | None = None,
    ) -> list[str]:
        """Scan for files, respecting .gitignore, ignore patterns, and path scoping."""
        root = Path(project_path)

        content_exts = set(config.content.extensions) if config else {".md", ".mdx"}
        code_exts = set(config.code.resolved_extensions) if config else set()
        allowed_extensions = content_exts | code_exts
        content_paths = config.content.paths if config else []
        code_paths = config.code.paths if config else []
        max_file_size_kb = config.indexing.max_file_size_kb if config else MAX_FILE_SIZE_KB

        norm_content_paths = [p.rstrip("/") for p in content_paths]
        norm_code_paths = [p.rstrip("/") for p in code_paths]

        # Load .gitignore
        gitignore_spec = None
        gitignore_path = root / ".gitignore"
        if gitignore_path.exists():
            try:
                lines = gitignore_path.read_text().splitlines()
                gitignore_spec = pathspec.PathSpec.from_lines("gitignore", lines)
            except (OSError, UnicodeDecodeError):
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

    def _detect_changes(
        self,
        all_files: list[str],
        stored_hashes: dict[str, str],
        project_path: str,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Returns (changed_files_with_hash, deleted_rel_paths)."""
        current_rel_paths = set()
        changed: list[tuple[str, str]] = []

        for fp in all_files:
            rel = os.path.relpath(fp, project_path)
            current_rel_paths.add(rel)
            current_hash = _compute_file_hash(fp)
            if rel not in stored_hashes or stored_hashes[rel] != current_hash:
                changed.append((fp, current_hash))

        deleted = [rp for rp in stored_hashes if rp not in current_rel_paths]
        return changed, deleted

    def index_all(
        self,
        changed_only: bool = False,
        lock: bool = True,
    ) -> list[IndexResult]:
        """Index all registered projects. Returns list of IndexResult for indexed projects."""
        if lock:
            from dnomia_knowledge.lock import IndexLock

            _lock = IndexLock()
            if not _lock.acquire():
                logger.info("Another index process is running, skipping.")
                return []
        else:
            _lock = None

        try:
            projects = self.store.list_projects()
            results = []

            for proj in projects:
                project_path = proj["path"]
                project_id = proj["id"]

                if not os.path.isdir(project_path):
                    logger.warning(
                        "Project directory missing, skipping: %s (%s)", project_id, project_path
                    )
                    continue

                if changed_only and not self._project_has_changes(proj):
                    logger.debug("No changes detected, skipping: %s", project_id)
                    continue

                try:
                    from dnomia_knowledge.registry import load_config, default_config

                    config = load_config(project_path)
                    if config is None:
                        config = default_config(project_path)

                    result = self.index_directory(
                        project_id=project_id,
                        directory=project_path,
                        incremental=True,
                        config=config,
                    )
                    results.append(result)
                    logger.info(
                        "Indexed %s: %d files, %d chunks, %.1fs",
                        project_id,
                        result.indexed_files,
                        result.total_chunks,
                        result.duration_seconds,
                    )
                except (OSError, sqlite3.Error, ValueError) as e:
                    logger.warning("Failed to index %s: %s", project_id, e)

            return results
        finally:
            if _lock:
                _lock.release()

    def _project_has_changes(self, proj: dict) -> bool:
        """Check if project has changes since last index."""
        project_path = proj["path"]
        stored_commit = proj.get("last_indexed_commit")

        # Try git HEAD comparison first
        current_commit = _get_git_head(project_path)
        if current_commit and stored_commit:
            return current_commit != stored_commit
        if current_commit and not stored_commit:
            return True  # Never indexed with commit tracking

        # Fallback: check file mtimes against last_indexed
        last_indexed = proj.get("last_indexed")
        if not last_indexed:
            return True  # Never indexed

        from datetime import datetime

        try:
            last_dt = datetime.fromisoformat(last_indexed)
            last_ts = last_dt.timestamp()
        except (ValueError, TypeError):
            return True

        # Recursive walk to catch nested changes (vault subdirs, etc.)
        for root, _dirs, files in os.walk(project_path):
            for fname in files:
                try:
                    fpath = os.path.join(root, fname)
                    if os.stat(fpath).st_mtime > last_ts:
                        return True
                except OSError:
                    continue

        return False

    def _read_file(self, file_path: str) -> str | None:
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None


def _compute_file_hash(file_path: str) -> str:
    """MD5 content hash."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()  # noqa: S324
    except OSError:
        return ""


def _get_git_head(directory: str) -> str | None:
    """Get current git HEAD commit hash, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _serialize_metadata(meta: dict | None) -> str | None:
    if meta is None:
        return None
    import json

    return json.dumps(meta, ensure_ascii=False, default=str)
