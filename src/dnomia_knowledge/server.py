"""FastMCP server for dnomia-knowledge."""

from __future__ import annotations

import os
import subprocess as _subprocess
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.indexer import Indexer
from dnomia_knowledge.registry import default_config, load_config
from dnomia_knowledge.search import HybridSearch
from dnomia_knowledge.store import Store


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text converter using stdlib."""

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self) -> str:
        return "\n".join(line.strip() for line in "".join(self._text).split("\n") if line.strip())


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


def _find_project_for_path(file_path: str) -> tuple[str, str, str] | None:
    """Returns (project_id, project_path, rel_path) or None."""
    store = _get_store()
    projects = store.list_projects()
    best = None
    best_len = 0
    for proj in projects:
        proj_path = proj["path"]
        normalized = proj_path if proj_path.endswith("/") else proj_path + "/"
        if file_path.startswith(normalized) or file_path == proj_path:
            if len(proj_path) > best_len:
                best = (proj["id"], proj_path, file_path[len(proj_path) :].lstrip("/"))
                best_len = len(proj_path)
    return best


def create_server() -> FastMCP:
    server = FastMCP("dnomia-knowledge")

    @server.tool()
    async def search(
        query: str,
        domain: str = "all",
        project: str | None = None,
        cross: bool = False,
        limit: int = 10,
        language: str = "",
        file_pattern: str = "",
        show_content: bool = False,
    ) -> str:
        """Hybrid semantic + keyword search across project knowledge.

        Args:
            query: Search query text
            domain: Filter by "all", "code", or "content"
            project: Project ID (default: DNOMIA_KNOWLEDGE_PROJECT env var)
            cross: If True, also search linked projects
            limit: Maximum results to return
            language: Filter by language (e.g. "python", "typescript")
            file_pattern: Filter by file path pattern (e.g. "auth", "models.py")
            show_content: If True, show full chunk content instead of truncated snippet
        """
        if not query or not query.strip():
            return "Empty query."

        project_id = project or _default_project()
        searcher = _get_search()
        lang = language or None
        fpattern = file_pattern or None

        if cross and project_id:
            from dnomia_knowledge.registry import load_config

            config = None
            proj = _get_store().get_project(project_id)
            if proj:
                config = load_config(proj["path"])
            related = config.links.related if config else []
            if related:
                results = searcher.search_cross(
                    query,
                    project_id=project_id,
                    related_projects=related,
                    domain=domain,
                    limit=limit,
                    language=lang,
                    file_pattern=fpattern,
                )
            else:
                results = searcher.search(
                    query,
                    project_id=project_id,
                    domain=domain,
                    limit=limit,
                    language=lang,
                    file_pattern=fpattern,
                )
        else:
            results = searcher.search(
                query,
                project_id=project_id,
                domain=domain,
                limit=limit,
                language=lang,
                file_pattern=fpattern,
            )

        if not results:
            return "No results found."

        store = _get_store()
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

            content_block = ""
            if show_content:
                full_content = store.get_chunk_content(r.chunk_id)
                if full_content:
                    content_block = "\n".join(f"    {line}" for line in full_content.split("\n"))
            else:
                if r.snippet:
                    snippet_lines = r.snippet.split("\n")[:5]
                    content_block = "\n".join(f"    {line}" for line in snippet_lines)

            parts = [header]
            if name_line:
                parts.append(name_line)
            if content_block:
                parts.append(content_block)
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

    @server.tool()
    async def graph_query(
        chunk_id: int | None = None,
        project: str | None = None,
        mode: str = "neighbors",
        depth: int = 1,
    ) -> str:
        """Query the knowledge graph.

        Args:
            chunk_id: Chunk ID to start traversal from (required for neighbors mode)
            project: Project ID (required for communities mode)
            mode: "neighbors" for BFS traversal, "communities" for Louvain groups
            depth: BFS depth for neighbors mode (1-3)
        """
        store = _get_store()

        if mode == "neighbors":
            if chunk_id is None:
                return "Error: chunk_id required for neighbors mode."
            depth = max(1, min(depth, 3))
            neighbors = store.get_neighbors(chunk_id, depth=depth)
            if not neighbors:
                return f"No neighbors found for chunk {chunk_id} at depth {depth}."
            lines = [f"Neighbors of chunk {chunk_id} (depth={depth}):"]
            for n in neighbors:
                line = f"  [{n['chunk_id']}] {n['file_path']}"
                if n["name"]:
                    line += f" ({n['chunk_type']}: {n['name']})"
                line += f" via {n['edge_type']} (w={n['weight']:.2f}, d={n['depth']})"
                lines.append(line)
            return "\n".join(lines)

        elif mode == "communities":
            project_id = project or _default_project()
            if not project_id:
                return "Error: project required for communities mode."
            import json as _json

            conn = store._connect()
            rows = conn.execute(
                "SELECT id, name, file_path, metadata FROM chunks WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            communities: dict[str, list[str]] = {}
            for r in rows:
                meta = _json.loads(r["metadata"]) if r["metadata"] else {}
                cid = str(meta.get("community_id", "unassigned"))
                label = f"[{r['id']}] {r['file_path']}"
                if r["name"]:
                    label += f" ({r['name']})"
                communities.setdefault(cid, []).append(label)

            if not communities or (len(communities) == 1 and "unassigned" in communities):
                return "No communities found. Run rebuild-graph first."

            lines = [f"Communities for {project_id}:"]
            for cid, members in sorted(communities.items()):
                if cid == "unassigned":
                    continue
                lines.append(f"\n  Community {cid} ({len(members)} chunks):")
                for m in members[:10]:
                    lines.append(f"    {m}")
                if len(members) > 10:
                    lines.append(f"    ... and {len(members) - 10} more")
            return "\n".join(lines)

        return f"Unknown mode: {mode}. Use 'neighbors' or 'communities'."

    @server.tool()
    async def read_file(
        file_path: str, query: str | None = None, project: str | None = None
    ) -> str:
        """Smart file reading with index awareness.

        If the file is indexed and large, returns relevant chunks instead of full content.
        Falls back to raw file reading for non-indexed files.

        Args:
            file_path: Absolute path to the file to read
            query: Optional search query to find relevant sections in large files
            project: Project ID (default: auto-detect from file path)
        """
        p = Path(file_path)
        if not p.is_file():
            return f"Error: file not found: {file_path}"

        # Resolve project
        project_id = None
        rel_path = None
        if project:
            proj = _get_store().get_project(project)
            if proj:
                project_id = project
                proj_path = proj["path"]
                normalized = proj_path if proj_path.endswith("/") else proj_path + "/"
                if file_path.startswith(normalized):
                    rel_path = file_path[len(proj_path) :].lstrip("/")
        else:
            match = _find_project_for_path(file_path)
            if match:
                project_id, _, rel_path = match

        # Check if file is indexed and get line count
        store = _get_store()
        line_count = None
        if project_id and rel_path:
            line_count = store.get_file_line_count(project_id, rel_path)

        # Small file or not indexed: read from disk with line numbers
        if line_count is None or line_count < 200:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                return f"Error reading file: {e}"
            lines = text.split("\n")
            numbered = [f"{i + 1:6}\t{line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)

        # Large file + query: use search to find relevant chunks
        if query:
            searcher = _get_search()
            results = searcher.search(
                query,
                project_id=project_id,
                domain="all",
                limit=5,
                file_pattern=rel_path,
            )
            if not results:
                return f"No relevant sections found for query '{query}' in {file_path}."
            parts = [f"Relevant sections in {file_path} ({line_count} lines):"]
            for i, r in enumerate(results, 1):
                header = f"\n[{i}] L{r.start_line}-{r.end_line}"
                if r.name:
                    header += f" ({r.chunk_type}: {r.name})"
                header += f" score={r.score:.4f}"
                full_content = store.get_chunk_content(r.chunk_id)
                content = full_content if full_content else (r.snippet or "")
                parts.append(header)
                parts.append(content)
            return "\n".join(parts)

        # Large file + no query: return metadata + first 50 lines
        chunks = store.get_chunks_for_file(project_id, rel_path)
        parts = [f"{file_path} ({line_count} lines, {len(chunks)} chunks)"]
        parts.append("\nChunks:")
        for c in chunks:
            label = f"  {c['chunk_type']}"
            if c["name"]:
                label += f" {c['name']}"
            label += f" L{c['start_line']}-{c['end_line']}"
            parts.append(label)

        parts.append("\nFirst 50 lines:")
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"
        lines = text.split("\n")[:50]
        numbered = [f"{i + 1:6}\t{line}" for i, line in enumerate(lines)]
        parts.append("\n".join(numbered))
        return "\n".join(parts)

    @server.tool()
    async def execute(command: str, cwd: str | None = None, timeout: int = 30) -> str:
        """Execute a shell command and return output.

        Captures stdout and stderr. If output exceeds 100 lines, truncates with summary.
        Use this instead of Bash for commands that produce large output.

        Args:
            command: Shell command to execute
            cwd: Working directory (default: current directory)
            timeout: Timeout in seconds (max 120, default 30)
        """
        timeout = min(timeout, 120)

        try:
            result = _subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        except _subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s."
        except Exception as e:
            return f"Error: {e}"

        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n--- stderr (exit code {result.returncode}) ---\n{result.stderr}"
        elif result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"

        lines = output.split("\n")
        if len(lines) <= 100:
            return output

        truncated = "\n".join(lines[:100])
        return f"{truncated}\n\n... {len(lines) - 100} more lines truncated."

    @server.tool()
    async def fetch_and_index(url: str, project: str | None = None) -> str:
        """Fetch URL content, convert to text, and index for searching.

        After indexing, the content is searchable via the search tool.

        Args:
            url: URL to fetch and index
            project: Project ID to store under (default: derived from URL domain)
        """
        from dnomia_knowledge.chunker.md_chunker import MdChunker

        # Fetch URL
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
                raw = resp.read()
                content_type = resp.headers.get_content_type() or ""
                charset = resp.headers.get_content_charset() or "utf-8"
                body = raw.decode(charset, errors="replace")
        except Exception as e:
            return f"Error fetching {url}: {e}"

        # Strip HTML tags if HTML content
        if "html" in content_type:
            extractor = _HTMLTextExtractor()
            extractor.feed(body)
            text = extractor.get_text()
        else:
            text = body

        if not text.strip():
            return f"No content extracted from {url}."

        # Derive project name from domain
        project_id = project or urlparse(url).netloc.replace(".", "-")
        url_path = urlparse(url).path or "/"

        # Register project
        store = _get_store()
        store.register_project(project_id, url, "web")

        # Chunk text
        chunker = MdChunker()
        chunks = chunker.chunk(url_path, text)

        if not chunks:
            return f"No chunks extracted from {url}."

        # Build chunk dicts
        chunks_data = [
            {
                "file_path": url_path,
                "chunk_domain": "content",
                "chunk_type": c.chunk_type,
                "name": c.name,
                "language": "md",
                "content": c.content,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ]

        # Insert chunks + embed vectors
        chunk_ids = store.insert_chunks(project_id, chunks_data)
        embedder = _get_embedder()
        vectors = embedder.embed_passages([c.content for c in chunks])
        store.insert_chunk_vectors(chunk_ids, vectors)

        word_count = len(text.split())
        return (
            f"Indexed {url}: {word_count} words, {len(chunk_ids)} chunks. Use search tool to query."
        )

    return server


# Entry point
server = create_server()

if __name__ == "__main__":
    server.run()
