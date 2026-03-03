"""CLI for dnomia-knowledge."""

from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def _get_db_path() -> str:
    return os.environ.get(
        "DNOMIA_KNOWLEDGE_DB",
        os.path.expanduser("~/.local/share/dnomia-knowledge/knowledge.db"),
    )


def cmd_index(args: argparse.Namespace) -> None:
    """Index a project directory."""
    from dnomia_knowledge.embedder import Embedder
    from dnomia_knowledge.indexer import Indexer
    from dnomia_knowledge.registry import default_config, load_config
    from dnomia_knowledge.store import Store

    path = os.path.abspath(args.path)
    if not os.path.isdir(path):
        console.print(f"[red]Error:[/red] {path} is not a directory.")
        sys.exit(1)

    config = load_config(path)
    if config is None:
        config = default_config(path)
        console.print("[dim]No .knowledge.toml found, using defaults.[/dim]")
    else:
        console.print(f"[green]Loaded[/green] .knowledge.toml: {config.name} ({config.type})")

    store = Store(_get_db_path())
    embedder = Embedder()
    indexer = Indexer(store, embedder)

    incremental = not args.full

    with console.status("[bold green]Indexing..."):
        result = indexer.index_directory(
            project_id=config.name,
            directory=path,
            incremental=incremental,
            config=config,
        )

    console.print(f"\n[bold]{result.project_id}[/bold] indexed successfully")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Total files", str(result.total_files))
    table.add_row("Indexed files", str(result.indexed_files))
    table.add_row("Content chunks", str(result.content_chunks))
    table.add_row("Code chunks", str(result.code_chunks))
    table.add_row("Duration", f"{result.duration_seconds}s")
    console.print(table)

    store.close()


def cmd_search(args: argparse.Namespace) -> None:
    """Search indexed projects."""
    from dnomia_knowledge.embedder import Embedder
    from dnomia_knowledge.search import HybridSearch
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    embedder = Embedder()
    search = HybridSearch(store, embedder)

    results = search.search(
        query=args.query,
        project_id=args.project,
        domain=args.domain,
        limit=args.limit,
    )

    if not results:
        console.print("[dim]No results found.[/dim]")
        store.close()
        return

    for i, r in enumerate(results, 1):
        header = f"[bold][{i}][/bold] {r.file_path}"
        if r.start_line:
            header += f":{r.start_line}-{r.end_line}"
        header += f" [dim](score: {r.score:.4f})[/dim]"
        if r.project_id:
            header += f" [cyan]{r.project_id}[/cyan]"

        console.print(header)
        if r.name:
            console.print(f"    {r.chunk_type}: {r.name}")
        snippet_lines = r.snippet.split("\n")[:5]
        for line in snippet_lines:
            console.print(f"    [dim]{line}[/dim]")
        console.print()

    store.close()


def cmd_projects(args: argparse.Namespace) -> None:
    """List registered projects."""
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    projects = store.list_projects()

    if not projects:
        console.print("[dim]No projects registered.[/dim]")
        store.close()
        return

    table = Table(title="Projects")
    table.add_column("ID", style="bold")
    table.add_column("Type")
    table.add_column("Path")
    table.add_column("Last Indexed")

    for p in projects:
        table.add_row(
            p["id"],
            p["type"],
            p["path"],
            p["last_indexed"] or "never",
        )

    console.print(table)
    store.close()


def cmd_stats(args: argparse.Namespace) -> None:
    """Show detailed stats for a project."""
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    proj = store.get_project(args.project_id)
    if not proj:
        console.print(f"[red]Project '{args.project_id}' not found.[/red]")
        store.close()
        sys.exit(1)

    stats = store.get_project_stats(args.project_id)

    console.print(f"\n[bold]{proj['id']}[/bold] ({proj['type']})")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Path", proj["path"])
    table.add_row("Total chunks", str(stats["total_chunks"]))
    table.add_row("Content chunks", str(stats["content_chunks"]))
    table.add_row("Code chunks", str(stats["code_chunks"]))
    table.add_row("Total files", str(stats["total_files"]))
    table.add_row("Graph", "enabled" if proj["graph_enabled"] else "disabled")
    table.add_row("Last indexed", proj["last_indexed"] or "never")
    table.add_row("Config hash", proj["config_hash"] or "none")
    console.print(table)

    store.close()


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run health checks. (Implemented in Task 6.)"""
    console.print("[dim]Doctor command not yet implemented.[/dim]")


def cmd_gc(args: argparse.Namespace) -> None:
    """Garbage collection: clean up orphan data."""
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    projects = store.list_projects()
    cleaned = 0

    for proj in projects:
        hashes = store.get_all_file_hashes(proj["id"])
        project_path = proj["path"]
        for rel_path in hashes:
            full_path = os.path.join(project_path, rel_path)
            if not os.path.exists(full_path):
                store.delete_file_chunks(proj["id"], rel_path)
                store.delete_file_index(proj["id"], rel_path)
                cleaned += 1

    if cleaned:
        console.print(f"[green]Cleaned {cleaned} stale file entries.[/green]")
    else:
        console.print("[dim]No orphan data found.[/dim]")

    store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dnomia-knowledge",
        description="Unified knowledge management CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = subparsers.add_parser("index", help="Index a project directory")
    p_index.add_argument("path", help="Project root directory path")
    p_index.add_argument("--full", action="store_true", help="Full reindex (ignore cache)")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = subparsers.add_parser("search", help="Search indexed projects")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--project", "-p", help="Filter by project ID")
    p_search.add_argument("--domain", "-d", choices=["all", "code", "content"], default="all")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    # projects
    p_projects = subparsers.add_parser("projects", help="List registered projects")
    p_projects.set_defaults(func=cmd_projects)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show project stats")
    p_stats.add_argument("project_id", help="Project ID")
    p_stats.set_defaults(func=cmd_stats)

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Run health checks")
    p_doctor.add_argument("project_id", nargs="?", help="Specific project ID (optional)")
    p_doctor.set_defaults(func=cmd_doctor)

    # gc
    p_gc = subparsers.add_parser("gc", help="Clean up orphan data")
    p_gc.set_defaults(func=cmd_gc)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
