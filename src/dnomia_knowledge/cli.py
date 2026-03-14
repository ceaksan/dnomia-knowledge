"""CLI for dnomia-knowledge."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    """Run health checks on the knowledge system."""
    from pathlib import Path as PathLib

    from dnomia_knowledge.registry import compute_config_hash, load_config
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def ok(msg: str) -> None:
        nonlocal checks_passed
        console.print(f"  [green]OK[/green] {msg}")
        checks_passed += 1

    def fail(msg: str) -> None:
        nonlocal checks_failed
        console.print(f"  [red]FAIL[/red] {msg}")
        checks_failed += 1

    def warn(msg: str) -> None:
        nonlocal checks_warned
        console.print(f"  [yellow]WARN[/yellow] {msg}")
        checks_warned += 1

    # --- System Checks ---
    console.print("\n[bold]System Checks[/bold]")

    # Check 1: sqlite-vec extension loaded
    try:
        conn = store._connect()
        conn.execute("SELECT vec_version()")
        ok("sqlite-vec extension loaded")
    except Exception as e:
        fail(f"sqlite-vec extension: {e}")

    # Check 2: Embedding model loadable
    try:
        from dnomia_knowledge.embedder import Embedder

        embedder = Embedder()
        embedder._ensure_loaded()
        ok(f"Embedding model: {embedder.model_name}")
        embedder.unload()
    except Exception as e:
        fail(f"Embedding model: {e}")

    # Check 3: Schema version
    schema_version = store.get_metadata("schema_version")
    if schema_version:
        ok(f"Schema version: {schema_version}")
    else:
        warn("Schema version not set")

    # --- Per-Project Checks ---
    projects_to_check = []
    if args.project_id:
        proj = store.get_project(args.project_id)
        if proj:
            projects_to_check = [proj]
        else:
            fail(f"Project '{args.project_id}' not found")
    else:
        projects_to_check = store.list_projects()

    for proj in projects_to_check:
        console.print(f"\n[bold]Project: {proj['id']}[/bold]")
        project_path = proj["path"]

        # Check 4: .knowledge.toml parseable
        toml_path = PathLib(project_path) / ".knowledge.toml"
        if toml_path.exists():
            try:
                config = load_config(project_path)
                ok(f".knowledge.toml parsed: {config.name} ({config.type})")
            except Exception as e:
                fail(f".knowledge.toml parse error: {e}")
                continue
        else:
            warn(".knowledge.toml not found (using defaults)")

        # Check 5: config_hash current
        if toml_path.exists():
            current_hash = compute_config_hash(toml_path)
            stored_hash = proj.get("config_hash")
            if stored_hash == current_hash:
                ok("Config hash current")
            else:
                warn("Config changed since last index, reindex recommended")

        # Check 6: chunks count = chunks_vec count
        conn = store._connect()
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ?", (proj["id"],)
        ).fetchone()[0]
        vec_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_vec cv "
            "JOIN chunks c ON cv.id = c.id "
            "WHERE c.project_id = ?",
            (proj["id"],),
        ).fetchone()[0]
        if chunk_count == vec_count:
            ok(f"Chunk/vector count match: {chunk_count}")
        else:
            fail(f"Chunk count ({chunk_count}) != vector count ({vec_count})")

        # Check 7: file_index files exist on disk
        hashes = store.get_all_file_hashes(proj["id"])
        missing = []
        for rel_path in hashes:
            full_path = os.path.join(project_path, rel_path)
            if not os.path.exists(full_path):
                missing.append(rel_path)
        if not missing:
            ok(f"All {len(hashes)} indexed files exist on disk")
        else:
            warn(f"{len(missing)} files missing from disk (run gc to clean)")

    # --- Usage Data Checks ---
    console.print("\n[bold]Usage Data[/bold]")

    conn = store._connect()
    interaction_count = conn.execute("SELECT COUNT(*) FROM chunk_interactions").fetchone()[0]
    if interaction_count > 0:
        ok(f"chunk_interactions: {interaction_count} entries")
    else:
        warn("chunk_interactions is empty (no usage data collected yet)")

    search_log_count = conn.execute("SELECT COUNT(*) FROM search_log").fetchone()[0]
    if search_log_count > 0:
        ok(f"search_log: {search_log_count} entries")
    else:
        warn("search_log is empty (no searches recorded yet)")

    # Summary
    console.print(
        f"\n[bold]Summary:[/bold] {checks_passed} passed, "
        f"{checks_warned} warnings, {checks_failed} failed"
    )

    store.close()
    if checks_failed:
        sys.exit(1)


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

    orphaned = store.clean_orphaned_interactions()
    if orphaned:
        console.print(f"[green]Cleaned {orphaned} orphaned interactions.[/green]")

    if args.full:
        interactions_deleted = store.delete_old_interactions(90)
        logs_deleted = store.delete_old_search_logs(90)
        if interactions_deleted or logs_deleted:
            console.print(
                f"[green]Full cleanup:[/green] {interactions_deleted} old interactions, "
                f"{logs_deleted} old search logs removed"
            )
        else:
            console.print("[dim]No old interactions or search logs to clean.[/dim]")

    store.close()


def cmd_rebuild_graph(args: argparse.Namespace) -> None:
    """Rebuild knowledge graph for a project."""
    from dnomia_knowledge.graph import GraphBuilder
    from dnomia_knowledge.registry import GraphConfig, load_config
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    proj = store.get_project(args.project_id)
    if not proj:
        console.print(f"[red]Project '{args.project_id}' not found.[/red]")
        store.close()
        sys.exit(1)

    if not proj["graph_enabled"]:
        console.print("[yellow]Graph is not enabled for this project.[/yellow]")
        console.print("[dim]Set graph.enabled = true in .knowledge.toml and reindex.[/dim]")
        store.close()
        return

    config = load_config(proj["path"])
    graph_config = config.graph if config else GraphConfig(enabled=True)
    builder = GraphBuilder(store, graph_config)

    with console.status("[bold green]Rebuilding edges..."):
        counts = builder.rebuild_all_edges(args.project_id)

    console.print(f"\n[bold]{args.project_id}[/bold] graph rebuilt")
    table = Table(show_header=False, box=None, padding=(0, 2))
    for edge_type, count in sorted(counts.items()):
        table.add_row(f"{edge_type} edges", str(count))
    console.print(table)

    with console.status("[bold green]Running community detection..."):
        n_communities = builder.run_community_detection(args.project_id)

    console.print(f"Communities detected: {n_communities}")

    store.close()


def cmd_index_all(args: argparse.Namespace) -> None:
    """Index all registered projects."""
    from dnomia_knowledge.embedder import Embedder
    from dnomia_knowledge.indexer import Indexer
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())
    embedder = Embedder()
    indexer = Indexer(store, embedder)

    results = indexer.index_all(changed_only=args.changed)

    if not results:
        if args.changed:
            console.print("[dim]No projects with changes.[/dim]")
        else:
            console.print("[dim]No projects registered.[/dim]")
    else:
        for r in results:
            console.print(
                f"  [green]{r.project_id}[/green]: "
                f"{r.indexed_files} files, {r.total_chunks} chunks, {r.duration_seconds}s"
            )

    store.close()


HOOK_START_MARKER = "# --- dnomia-knowledge-start ---"
HOOK_END_MARKER = "# --- dnomia-knowledge-end ---"


def _get_hook_block(bin_path: str) -> str:
    return (
        f"{HOOK_START_MARKER}\n"
        f'DNOMIA_BIN="{bin_path}"\n'
        f'if [ -x "$DNOMIA_BIN" ]; then\n'
        f"  (\n"
        f"    mkdir /tmp/dnomia-knowledge.lock 2>/dev/null || exit 0\n"
        f"    trap 'rmdir /tmp/dnomia-knowledge.lock' EXIT\n"
        f'    "$DNOMIA_BIN" index "$(git rev-parse --show-toplevel)" >/dev/null 2>&1\n'
        f"  ) &\n"
        f"fi\n"
        f"{HOOK_END_MARKER}\n"
    )


def _install_hooks(store, bin_path: str | None = None) -> int:
    """Install post-commit hooks to all registered git projects. Returns count installed."""
    if bin_path is None:
        import shutil

        bin_path = shutil.which("dnomia-knowledge") or sys.executable.replace(
            "python", "dnomia-knowledge"
        )

    projects = store.list_projects()
    installed = 0

    for proj in projects:
        project_path = Path(proj["path"])
        git_dir = project_path / ".git"
        if not git_dir.is_dir():
            continue

        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        hook_path = hooks_dir / "post-commit"

        hook_block = _get_hook_block(bin_path)

        if hook_path.exists():
            existing = hook_path.read_text()
            if HOOK_START_MARKER in existing:
                continue  # Already installed
            new_content = existing.rstrip("\n") + "\n\n" + hook_block
        else:
            new_content = "#!/bin/sh\n\n" + hook_block

        hook_path.write_text(new_content)
        hook_path.chmod(0o755)
        installed += 1

    return installed


def _uninstall_hooks(store) -> int:
    """Remove dnomia-knowledge blocks from all post-commit hooks. Returns count removed."""
    projects = store.list_projects()
    removed = 0

    for proj in projects:
        project_path = Path(proj["path"])
        hook_path = project_path / ".git" / "hooks" / "post-commit"
        if not hook_path.exists():
            continue

        content = hook_path.read_text()
        if HOOK_START_MARKER not in content:
            continue

        # Remove the block
        lines = content.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if line.strip() == HOOK_START_MARKER.strip():
                skip = True
                continue
            if line.strip() == HOOK_END_MARKER.strip():
                skip = False
                continue
            if not skip:
                new_lines.append(line)

        new_content = "\n".join(new_lines).strip()
        if new_content and new_content != "#!/bin/sh":
            hook_path.write_text(new_content + "\n")
        else:
            hook_path.unlink()
        removed += 1

    return removed


def cmd_install_hooks(args: argparse.Namespace) -> None:
    """Install or uninstall git post-commit hooks."""
    from dnomia_knowledge.store import Store

    store = Store(_get_db_path())

    if args.uninstall:
        count = _uninstall_hooks(store)
        console.print(f"[green]Removed hooks from {count} projects.[/green]")
    else:
        count = _install_hooks(store)
        console.print(f"[green]Installed hooks in {count} projects.[/green]")

    store.close()


PLIST_LABEL = "com.dnomia-knowledge.index"


def _generate_plist(bin_path: str) -> str:
    """Generate launchd plist XML."""
    home = os.path.expanduser("~")
    log_dir = os.path.join(home, ".local", "share", "dnomia-knowledge")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{bin_path}</string>
        <string>index-all</string>
        <string>--changed</string>
    </array>
    <key>StartInterval</key>
    <integer>300</integer>
    <key>StandardOutPath</key>
    <string>{log_dir}/index.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/index-error.log</string>
</dict>
</plist>
"""


def cmd_install_launchd(args: argparse.Namespace) -> None:
    """Install or uninstall launchd periodic job."""
    import shutil
    import subprocess as sp

    home = os.path.expanduser("~")
    plist_path = os.path.join(home, "Library", "LaunchAgents", f"{PLIST_LABEL}.plist")
    log_dir = os.path.join(home, ".local", "share", "dnomia-knowledge")

    if args.uninstall:
        if os.path.exists(plist_path):
            sp.run(["launchctl", "unload", plist_path], capture_output=True)
            os.remove(plist_path)
            console.print(f"[green]Unloaded and removed {plist_path}[/green]")
        else:
            console.print("[dim]No plist found.[/dim]")
        return

    bin_path = shutil.which("dnomia-knowledge")
    if not bin_path:
        # Try venv
        venv_bin = os.path.join(os.path.dirname(sys.executable), "dnomia-knowledge")
        if os.path.exists(venv_bin):
            bin_path = venv_bin
        else:
            console.print("[red]Could not find dnomia-knowledge binary.[/red]")
            sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    plist_content = _generate_plist(bin_path)

    # Unload if existing
    if os.path.exists(plist_path):
        sp.run(["launchctl", "unload", plist_path], capture_output=True)

    with open(plist_path, "w") as f:
        f.write(plist_content)

    sp.run(["launchctl", "load", plist_path], capture_output=True)
    console.print(f"[green]Installed and loaded {plist_path}[/green]")
    console.print(f"[dim]Logs: {log_dir}/index.log[/dim]")


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

    # index-all
    p_index_all = subparsers.add_parser("index-all", help="Index all registered projects")
    p_index_all.add_argument(
        "--changed", action="store_true", help="Only index projects with changes"
    )
    p_index_all.set_defaults(func=cmd_index_all)

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
    p_gc.add_argument(
        "--full", action="store_true", help="Also clean old interactions and search logs (>90 days)"
    )
    p_gc.set_defaults(func=cmd_gc)

    # rebuild-graph
    p_graph = subparsers.add_parser("rebuild-graph", help="Rebuild knowledge graph")
    p_graph.add_argument("project_id", help="Project ID")
    p_graph.set_defaults(func=cmd_rebuild_graph)

    # install-hooks
    p_hooks = subparsers.add_parser("install-hooks", help="Install git post-commit hooks")
    p_hooks.add_argument("--uninstall", action="store_true", help="Remove hooks instead")
    p_hooks.set_defaults(func=cmd_install_hooks)

    # install-launchd
    p_launchd = subparsers.add_parser(
        "install-launchd", help="Install launchd periodic indexing job"
    )
    p_launchd.add_argument("--uninstall", action="store_true", help="Remove launchd job")
    p_launchd.set_defaults(func=cmd_install_launchd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
