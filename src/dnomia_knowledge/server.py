"""FastMCP server for dnomia-knowledge."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.registry import default_config, load_config
from dnomia_knowledge.search import HybridSearch
from dnomia_knowledge.store import Store

# Default DB path
_DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/dnomia-knowledge/knowledge.db")

# Lazy-loaded singletons
_store: Store | None = None
_embedder: Embedder | None = None
_search: HybridSearch | None = None
_indexer: Indexer | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        db_path = os.environ.get("DNOMIA_KNOWLEDGE_DB", _DEFAULT_DB_PATH)
        _store = Store(db_path)
    return _store


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_search() -> HybridSearch:
    global _search
    if _search is None:
        _search = HybridSearch(_get_store(), _get_embedder())
    return _search


def _get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(_get_store(), _get_embedder())
    return _indexer


def _default_project() -> str | None:
    return os.environ.get("DNOMIA_KNOWLEDGE_PROJECT")


def create_server() -> FastMCP:
    server = FastMCP("dnomia-knowledge")

    @server.tool()
    async def search(
        query: str,
        domain: str = "all",
        project: str | None = None,
        cross: bool = False,
        limit: int = 10,
    ) -> str:
        """Hybrid semantic + keyword search across project knowledge.

        Args:
            query: Search query text
            domain: Filter by "all", "code", or "content"
            project: Project ID (default: DNOMIA_KNOWLEDGE_PROJECT env var)
            cross: If True, also search linked projects
            limit: Maximum results to return
        """
        if not query or not query.strip():
            return "Empty query."

        project_id = project or _default_project()
        searcher = _get_search()
        results = searcher.search(query, project_id=project_id, domain=domain, limit=limit)

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            header = f"[{i}] {r.file_path}"
            if r.start_line:
                header += f":{r.start_line}-{r.end_line}"
            header += f" (score: {r.score:.4f}"
            if r.project_id:
                header += f", project: {r.project_id}"
            header += ")"

            name_line = ""
            if r.name:
                name_line = f"    {r.chunk_type} {r.name}"
            elif r.chunk_type:
                name_line = f"    [{r.chunk_type}]"

            snippet = ""
            if r.snippet:
                snippet_lines = r.snippet.split("\n")[:5]
                snippet = "\n".join(f"    {line}" for line in snippet_lines)

            parts = [header]
            if name_line:
                parts.append(name_line)
            if snippet:
                parts.append(snippet)
            lines.append("\n".join(parts))

        return "\n\n".join(lines)

    @server.tool()
    async def index_project(
        path: str,
        incremental: bool = True,
    ) -> str:
        """Index or reindex a project's content and code.

        Args:
            path: Absolute path to project root (must contain .md/.mdx files)
            incremental: If True, only reindex changed files (default)
        """
        if not Path(path).is_dir():
            return f"Error: {path} is not a directory."

        # Load project config (.knowledge.toml) or use defaults
        config = load_config(path)
        if config is None:
            config = default_config(path)

        project_id = config.name
        indexer = _get_indexer()

        result = indexer.index_directory(
            project_id=project_id,
            directory=path,
            incremental=incremental,
            config=config,
        )

        return (
            f"Indexed {result.project_id}: "
            f"{result.total_files} files, "
            f"{result.total_chunks} chunks "
            f"({result.content_chunks} content + {result.code_chunks} code)\n"
            f"  Changed: {result.indexed_files} files\n"
            f"  Duration: {result.duration_seconds}s"
        )

    @server.tool()
    async def project_info(
        project: str | None = None,
    ) -> str:
        """List registered projects with stats.

        Args:
            project: Specific project ID, or None for all projects
        """
        store = _get_store()

        if project:
            proj = store.get_project(project)
            if not proj:
                return f"Project '{project}' not found."
            stats = store.get_project_stats(project)
            return (
                f"{proj['id']} ({proj['type']})\n"
                f"  Path: {proj['path']}\n"
                f"  Chunks: {stats['total_chunks']} "
                f"({stats['content_chunks']} content, {stats['code_chunks']} code)\n"
                f"  Files: {stats['total_files']}\n"
                f"  Graph: {'enabled' if proj['graph_enabled'] else 'disabled'}\n"
                f"  Last indexed: {proj['last_indexed'] or 'never'}"
            )

        projects = store.list_projects()
        if not projects:
            return "No projects registered. Use index_project to add one."

        lines = []
        for p in projects:
            stats = store.get_project_stats(p["id"])
            lines.append(
                f"- {p['id']} ({p['type']}): "
                f"{stats['total_chunks']} chunks, "
                f"{stats['total_files']} files"
            )
        return "\n".join(lines)

    return server


# Entry point
server = create_server()

if __name__ == "__main__":
    server.run()
