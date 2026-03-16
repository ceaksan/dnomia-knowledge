"""SQLite store with FTS5 and sqlite-vec support."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = "2"

_PRAGMAS = [
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 67108864",
    "PRAGMA cache_size = -64000",
]

_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    graph_enabled INTEGER DEFAULT 0,
    last_indexed TEXT,
    last_indexed_commit TEXT,
    config_hash TEXT
);

CREATE TABLE IF NOT EXISTS file_index (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    last_indexed TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, file_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    chunk_domain TEXT NOT NULL DEFAULT 'content',
    chunk_type TEXT NOT NULL DEFAULT 'block',
    name TEXT,
    language TEXT,
    start_line INTEGER DEFAULT 0,
    end_line INTEGER DEFAULT 0,
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edges (
    source_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    target_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    metadata TEXT,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE TABLE IF NOT EXISTS search_log (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    project_id TEXT,
    domain TEXT,
    result_chunk_ids TEXT,
    result_count INTEGER DEFAULT 0,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunk_interactions (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    project_id TEXT,
    file_path TEXT,
    interaction TEXT NOT NULL,
    source_tool TEXT,
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content=chunks,
    content_rowid=id,
    tokenize='porter unicode61'
);
"""

_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_vec_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_vec WHERE id = old.id;
END;
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_domain ON chunks(chunk_domain);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_search_log_project ON search_log(project_id);
CREATE INDEX IF NOT EXISTS idx_search_log_ts ON search_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_chunk ON chunk_interactions(chunk_id);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON chunk_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_project ON chunk_interactions(project_id);
CREATE INDEX IF NOT EXISTS idx_interactions_file ON chunk_interactions(project_id, file_path);
CREATE INDEX IF NOT EXISTS idx_interactions_proj_ts ON chunk_interactions(project_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_search_log_query ON search_log(query);
CREATE INDEX IF NOT EXISTS idx_search_log_result_count ON search_log(result_count);
CREATE INDEX IF NOT EXISTS idx_search_log_result_ts ON search_log(result_count, timestamp);
"""

_DEFAULT_EMBEDDING_DIM = 768


class Store:
    """SQLite store for chunks, embeddings, and search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            for pragma in _PRAGMAS:
                self._conn.execute(pragma)
        return self._conn

    def execute_sql(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        """Execute SQL and return cursor. For read queries or single writes."""
        return self._connect().execute(sql, params)

    def fetchall(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        """Execute SQL and return all rows."""
        return self._connect().execute(sql, params).fetchall()

    def commit(self) -> None:
        """Commit the current transaction."""
        self._connect().commit()

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(_TABLES_SQL)
        # Migration v1: add last_indexed_commit column
        try:
            conn.execute("ALTER TABLE projects ADD COLUMN last_indexed_commit TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migration v2: recreate chunk_interactions without CASCADE, add project_id/file_path
        has_project_col = any(
            row[1] == "project_id"
            for row in conn.execute("PRAGMA table_info(chunk_interactions)").fetchall()
        )
        if not has_project_col:
            conn.executescript("""
                CREATE TABLE chunk_interactions_new (
                    id INTEGER PRIMARY KEY,
                    chunk_id INTEGER NOT NULL,
                    project_id TEXT,
                    file_path TEXT,
                    interaction TEXT NOT NULL,
                    source_tool TEXT,
                    timestamp TEXT DEFAULT (datetime('now'))
                );
                INSERT INTO chunk_interactions_new
                    (id, chunk_id, interaction, source_tool, timestamp)
                SELECT id, chunk_id, interaction, source_tool, timestamp
                FROM chunk_interactions;
                DROP TABLE chunk_interactions;
                ALTER TABLE chunk_interactions_new
                    RENAME TO chunk_interactions;
            """)
            # Backfill project_id/file_path from chunks table
            conn.execute("""
                UPDATE chunk_interactions SET
                    project_id = (
                        SELECT c.project_id FROM chunks c
                        WHERE c.id = chunk_interactions.chunk_id
                    ),
                    file_path = (
                        SELECT c.file_path FROM chunks c
                        WHERE c.id = chunk_interactions.chunk_id
                    )
                WHERE project_id IS NULL
                    AND chunk_id IN (SELECT id FROM chunks)
            """)
            # Clean orphaned interactions (chunks already deleted by old CASCADE)
            conn.execute(
                "DELETE FROM chunk_interactions WHERE chunk_id NOT IN (SELECT id FROM chunks)"
            )
            conn.commit()

        conn.executescript(_FTS_SQL)
        conn.executescript(_TRIGGERS_SQL)
        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{_DEFAULT_EMBEDDING_DIM}]
            )"""
        )
        conn.executescript(_INDEXES_SQL)
        conn.execute(
            "INSERT INTO system_metadata (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (SCHEMA_VERSION,),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Projects --

    def register_project(
        self,
        project_id: str,
        path: str,
        project_type: str,
        graph_enabled: bool = False,
        config_hash: str | None = None,
    ) -> None:
        conn = self._connect()
        conn.execute(
            """INSERT INTO projects (id, path, type, graph_enabled, config_hash)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   path = excluded.path,
                   type = excluded.type,
                   graph_enabled = excluded.graph_enabled,
                   config_hash = excluded.config_hash""",
            (project_id, path, project_type, int(graph_enabled), config_hash),
        )
        conn.commit()

    def get_project(self, project_id: str) -> dict | None:
        conn = self._connect()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return dict(row) if row else None

    def list_projects(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM projects ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def update_project_last_indexed(self, project_id: str, commit_hash: str | None = None) -> None:
        """Update last_indexed timestamp and optionally commit hash."""
        conn = self._connect()
        if commit_hash:
            conn.execute(
                "UPDATE projects SET last_indexed = datetime('now'), last_indexed_commit = ? WHERE id = ?",
                (commit_hash, project_id),
            )
        else:
            conn.execute(
                "UPDATE projects SET last_indexed = datetime('now') WHERE id = ?",
                (project_id,),
            )
        conn.commit()

    # -- Chunks --

    def insert_chunks(
        self, project_id: str, chunks: list[dict], *, commit: bool = True
    ) -> list[int]:
        conn = self._connect()
        ids = []
        for c in chunks:
            cursor = conn.execute(
                """INSERT INTO chunks
                   (project_id, file_path, chunk_domain, chunk_type, name,
                    language, start_line, end_line, content, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id,
                    c.get("file_path", ""),
                    c.get("chunk_domain", "content"),
                    c.get("chunk_type", "block"),
                    c.get("name"),
                    c.get("language"),
                    c.get("start_line", 0),
                    c.get("end_line", 0),
                    c["content"],
                    c.get("metadata"),
                ),
            )
            ids.append(cursor.lastrowid)
        if commit:
            conn.commit()
        return ids

    def insert_chunk_vectors(
        self, chunk_ids: list[int], vectors: list[list[float]], *, commit: bool = True
    ) -> None:
        conn = self._connect()
        for chunk_id, vec in zip(chunk_ids, vectors):
            conn.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, json.dumps(vec)),
            )
        if commit:
            conn.commit()

    def delete_file_chunks(self, project_id: str, file_path: str, *, commit: bool = True) -> int:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM chunks WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        if commit:
            conn.commit()
        return cursor.rowcount

    def get_project_stats(self, project_id: str) -> dict:
        conn = self._connect()
        row = conn.execute(
            """SELECT
                   COUNT(*) as total,
                   SUM(CASE WHEN chunk_domain = 'content' THEN 1 ELSE 0 END) as content,
                   SUM(CASE WHEN chunk_domain = 'code' THEN 1 ELSE 0 END) as code,
                   COUNT(DISTINCT file_path) as files
               FROM chunks WHERE project_id = ?""",
            (project_id,),
        ).fetchone()
        return {
            "total_chunks": row[0] or 0,
            "content_chunks": row[1] or 0,
            "code_chunks": row[2] or 0,
            "total_files": row[3] or 0,
        }

    # -- File Index --

    def upsert_file_index(
        self,
        project_id: str,
        file_path: str,
        file_hash: str,
        chunk_count: int,
        *,
        commit: bool = True,
    ) -> None:
        conn = self._connect()
        conn.execute(
            """INSERT INTO file_index (project_id, file_path, file_hash, chunk_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(project_id, file_path) DO UPDATE SET
                   file_hash = excluded.file_hash,
                   chunk_count = excluded.chunk_count,
                   last_indexed = datetime('now')""",
            (project_id, file_path, file_hash, chunk_count),
        )
        if commit:
            conn.commit()

    def get_file_hash(self, project_id: str, file_path: str) -> str | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT file_hash FROM file_index WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()
        return row[0] if row else None

    def get_all_file_hashes(self, project_id: str) -> dict[str, str]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT file_path, file_hash FROM file_index WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_file_index(self, project_id: str, file_path: str) -> None:
        conn = self._connect()
        conn.execute(
            "DELETE FROM file_index WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        conn.commit()

    # -- System Metadata --

    def set_metadata(self, key: str, value: str) -> None:
        conn = self._connect()
        conn.execute(
            "INSERT INTO system_metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()

    def get_metadata(self, key: str) -> str | None:
        conn = self._connect()
        row = conn.execute("SELECT value FROM system_metadata WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # -- Edges --

    def insert_edges(self, edges: list[dict]) -> int:
        """INSERT OR IGNORE edges. Returns count of inserted rows."""
        conn = self._connect()
        count = 0
        for e in edges:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, edge_type, weight, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    e["source_id"],
                    e["target_id"],
                    e["edge_type"],
                    e.get("weight", 1.0),
                    e.get("metadata"),
                ),
            )
            count += cursor.rowcount
        conn.commit()
        return count

    def delete_edges_for_chunk(self, chunk_id: int, edge_type: str | None = None) -> int:
        """Delete edges where chunk is source or target. Optionally filter by type."""
        conn = self._connect()
        if edge_type:
            c1 = conn.execute(
                "DELETE FROM edges WHERE source_id = ? AND edge_type = ?",
                (chunk_id, edge_type),
            )
            c2 = conn.execute(
                "DELETE FROM edges WHERE target_id = ? AND edge_type = ?",
                (chunk_id, edge_type),
            )
        else:
            c1 = conn.execute("DELETE FROM edges WHERE source_id = ?", (chunk_id,))
            c2 = conn.execute("DELETE FROM edges WHERE target_id = ?", (chunk_id,))
        conn.commit()
        return c1.rowcount + c2.rowcount

    def delete_edges_for_project(self, project_id: str, edge_type: str | None = None) -> int:
        """Delete all edges for chunks belonging to a project."""
        conn = self._connect()
        if edge_type:
            cursor = conn.execute(
                """DELETE FROM edges WHERE edge_type = ? AND (
                       source_id IN (SELECT id FROM chunks WHERE project_id = ?)
                       OR target_id IN (SELECT id FROM chunks WHERE project_id = ?))""",
                (edge_type, project_id, project_id),
            )
        else:
            cursor = conn.execute(
                """DELETE FROM edges WHERE
                       source_id IN (SELECT id FROM chunks WHERE project_id = ?)
                       OR target_id IN (SELECT id FROM chunks WHERE project_id = ?)""",
                (project_id, project_id),
            )
        conn.commit()
        return cursor.rowcount

    def get_neighbors(
        self,
        chunk_id: int,
        depth: int = 1,
        edge_types: list[str] | None = None,
    ) -> list[dict]:
        """BFS traversal from chunk_id. Returns chunk info + edge metadata."""
        conn = self._connect()
        visited = set()
        result = []
        frontier = {chunk_id}

        for d in range(depth):
            if not frontier:
                break
            next_frontier = set()
            for cid in frontier:
                if cid in visited:
                    continue
                visited.add(cid)
                # Get outgoing edges
                if edge_types:
                    placeholders = ",".join("?" for _ in edge_types)
                    rows = conn.execute(
                        f"""SELECT e.target_id, e.edge_type, e.weight, e.metadata,
                                   c.file_path, c.chunk_type, c.name, c.content
                            FROM edges e
                            JOIN chunks c ON c.id = e.target_id
                            WHERE e.source_id = ? AND e.edge_type IN ({placeholders})""",
                        (cid, *edge_types),
                    ).fetchall()
                    # Also incoming edges
                    rows += conn.execute(
                        f"""SELECT e.source_id, e.edge_type, e.weight, e.metadata,
                                   c.file_path, c.chunk_type, c.name, c.content
                            FROM edges e
                            JOIN chunks c ON c.id = e.source_id
                            WHERE e.target_id = ? AND e.edge_type IN ({placeholders})""",
                        (cid, *edge_types),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT e.target_id, e.edge_type, e.weight, e.metadata,
                                  c.file_path, c.chunk_type, c.name, c.content
                           FROM edges e
                           JOIN chunks c ON c.id = e.target_id
                           WHERE e.source_id = ?""",
                        (cid,),
                    ).fetchall()
                    rows += conn.execute(
                        """SELECT e.source_id, e.edge_type, e.weight, e.metadata,
                                  c.file_path, c.chunk_type, c.name, c.content
                           FROM edges e
                           JOIN chunks c ON c.id = e.source_id
                           WHERE e.target_id = ?""",
                        (cid,),
                    ).fetchall()

                for row in rows:
                    neighbor_id = row[0]
                    if neighbor_id not in visited:
                        next_frontier.add(neighbor_id)
                        result.append(
                            {
                                "chunk_id": neighbor_id,
                                "edge_type": row[1],
                                "weight": row[2],
                                "edge_metadata": row[3],
                                "file_path": row[4],
                                "chunk_type": row[5],
                                "name": row[6],
                                "content": row[7],
                                "depth": d + 1,
                            }
                        )
            frontier = next_frontier

        return result

    def get_edges_for_project(self, project_id: str) -> list[dict]:
        """Get all edges for a project."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT e.source_id, e.target_id, e.edge_type, e.weight, e.metadata
               FROM edges e
               WHERE e.source_id IN (SELECT id FROM chunks WHERE project_id = ?)""",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_project_edge_stats(self, project_id: str) -> dict[str, int]:
        """Count edges by type for a project."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT e.edge_type, COUNT(*) as cnt
               FROM edges e
               WHERE e.source_id IN (SELECT id FROM chunks WHERE project_id = ?)
               GROUP BY e.edge_type""",
            (project_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_chunk_ids_for_project(self, project_id: str) -> list[int]:
        """Get all chunk IDs for a project."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id FROM chunks WHERE project_id = ? ORDER BY id",
            (project_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def update_chunk_metadata(self, chunk_id: int, metadata_updates: dict) -> None:
        """JSON merge metadata updates into existing chunk metadata."""
        conn = self._connect()
        row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return
        existing = json.loads(row[0]) if row[0] else {}
        existing.update(metadata_updates)
        conn.execute(
            "UPDATE chunks SET metadata = ? WHERE id = ?",
            (json.dumps(existing, ensure_ascii=False, default=str), chunk_id),
        )
        conn.commit()

    def batch_update_chunk_metadata(self, updates: list[tuple[int, dict]]) -> None:
        """Batch JSON merge metadata into multiple chunks in a single transaction."""
        if not updates:
            return
        conn = self._connect()
        for chunk_id, meta_updates in updates:
            conn.execute(
                "UPDATE chunks SET metadata = json_patch(COALESCE(metadata, '{}'), ?) WHERE id = ?",
                (json.dumps(meta_updates, ensure_ascii=False, default=str), chunk_id),
            )
        conn.commit()

    # -- Chunk Interactions --

    def log_interaction(
        self,
        chunk_id: int,
        interaction: str,
        source_tool: str,
        project_id: str | None = None,
        file_path: str | None = None,
    ) -> None:
        """Log a chunk interaction (read, edit, search_hit)."""
        conn = self._connect()
        conn.execute(
            "INSERT INTO chunk_interactions "
            "(chunk_id, project_id, file_path, interaction, source_tool) "
            "VALUES (?, ?, ?, ?, ?)",
            (chunk_id, project_id, file_path, interaction, source_tool),
        )
        conn.commit()

    def batch_log_interactions(
        self, interactions: list[tuple[int, str, str, str | None, str | None]]
    ) -> None:
        """Batch log interactions.

        Tuples: (chunk_id, interaction, source_tool, project_id, file_path).
        """
        if not interactions:
            return
        conn = self._connect()
        conn.executemany(
            "INSERT INTO chunk_interactions "
            "(chunk_id, interaction, source_tool, project_id, file_path) "
            "VALUES (?, ?, ?, ?, ?)",
            interactions,
        )
        conn.commit()

    def get_interaction_counts(
        self,
        chunk_ids: list[int],
        days: int = 30,
        interactions: list[str] | None = None,
        project_id: str | None = None,
    ) -> dict[int, int]:
        """Batch count interactions within N days, filtered by type and project."""
        if not chunk_ids:
            return {}
        conn = self._connect()
        placeholders = ",".join("?" for _ in chunk_ids)
        params: list = list(chunk_ids)

        days_param = f"-{int(days)} days"
        where = [
            f"ci.chunk_id IN ({placeholders})",
            "ci.timestamp >= datetime('now', ?)",
        ]
        params.append(days_param)

        if interactions:
            i_ph = ",".join("?" for _ in interactions)
            where.append(f"ci.interaction IN ({i_ph})")
            params.extend(interactions)

        if project_id:
            where.append("c.project_id = ?")
            params.append(project_id)

        where_sql = " AND ".join(where)
        join = "JOIN chunks c ON c.id = ci.chunk_id" if project_id else ""

        sql = f"""
            SELECT ci.chunk_id, COUNT(*) as cnt
            FROM chunk_interactions ci
            {join}
            WHERE {where_sql}
            GROUP BY ci.chunk_id
        """
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_old_interactions(self, days: int = 90) -> int:
        """Delete interactions older than N days."""
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM chunk_interactions WHERE timestamp < datetime('now', ?)",
            (f"-{int(days)} days",),
        )
        conn.commit()
        return cursor.rowcount

    def clean_orphaned_interactions(self) -> int:
        """Remove interactions whose chunk_id no longer exists in chunks table."""
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM chunk_interactions WHERE chunk_id NOT IN (SELECT id FROM chunks)"
        )
        conn.commit()
        return cursor.rowcount

    # -- Trace Aggregations --

    @staticmethod
    def _trace_filter(
        days: int,
        project_id: str | None,
        limit: int,
        ts_col: str = "timestamp",
        proj_col: str = "project_id",
    ) -> tuple[str, list]:
        """Build WHERE clause and params for trace queries."""
        params: list = [f"-{int(days)} days"]
        parts = [f"{ts_col} >= datetime('now', ?)"]
        if project_id:
            parts.append(f"{proj_col} = ?")
            params.append(project_id)
        params.append(limit)
        return " AND ".join(parts), params

    def get_hot_chunks(
        self,
        project_id: str | None = None,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """Most interacted files, grouped by (project_id, file_path)."""
        conn = self._connect()
        where_sql, params = self._trace_filter(
            days, project_id, limit, ts_col="ci.timestamp", proj_col="ci.project_id"
        )

        sql = f"""
            SELECT ci.project_id, ci.file_path,
                SUM(CASE WHEN ci.interaction = 'read'
                    THEN 1 ELSE 0 END) as reads,
                SUM(CASE WHEN ci.interaction = 'edit'
                    THEN 1 ELSE 0 END) as edits,
                SUM(CASE WHEN ci.interaction = 'search_hit'
                    THEN 1 ELSE 0 END) as searches,
                COUNT(*) as total
            FROM chunk_interactions ci
            WHERE {where_sql}
            GROUP BY ci.project_id, ci.file_path
            ORDER BY total DESC, ci.file_path ASC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_knowledge_gaps(
        self,
        project_id: str | None = None,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """Searches that returned 0 results, grouped by query."""
        conn = self._connect()
        where_sql, params = self._trace_filter(days, project_id, limit)
        where_sql = "result_count = 0 AND " + where_sql

        sql = f"""
            SELECT query,
                COUNT(*) as count,
                MAX(timestamp) as last_searched
            FROM search_log
            WHERE {where_sql}
            GROUP BY query
            ORDER BY count DESC, query ASC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_decaying_chunks(
        self,
        project_id: str | None = None,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """Files active in previous period but inactive now."""
        conn = self._connect()
        days_int = int(days)
        prev_start = f"-{days_int * 2} days"
        prev_end = f"-{days_int} days"
        curr_start = f"-{days_int} days"

        params: list = [prev_start, prev_end, curr_start, prev_start]
        where_parts = ["ci.timestamp >= datetime('now', ?)"]

        if project_id:
            where_parts.append("ci.project_id = ?")
            params.append(project_id)

        where_sql = " AND ".join(where_parts)
        params.append(limit)

        sql = f"""
            SELECT ci.project_id, ci.file_path,
                SUM(CASE
                    WHEN ci.timestamp >= datetime('now', ?)
                         AND ci.timestamp < datetime('now', ?)
                    THEN 1 ELSE 0
                END) as before_count,
                SUM(CASE
                    WHEN ci.timestamp >= datetime('now', ?)
                    THEN 1 ELSE 0
                END) as now_count
            FROM chunk_interactions ci
            WHERE {where_sql}
            GROUP BY ci.project_id, ci.file_path
            HAVING before_count > 3
               AND (before_count * 1.0 / (now_count + 1)) > 3
            ORDER BY (before_count * 1.0 / (now_count + 1)) DESC,
                     ci.file_path ASC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [
            {**dict(r), "ratio": round(r["before_count"] / (r["now_count"] + 1), 1)} for r in rows
        ]

    def get_top_queries(
        self,
        project_id: str | None = None,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """Most frequent search queries."""
        conn = self._connect()
        where_sql, params = self._trace_filter(days, project_id, limit)

        sql = f"""
            SELECT query,
                COUNT(*) as count,
                ROUND(AVG(result_count), 1) as avg_results,
                MAX(timestamp) as last_searched
            FROM search_log
            WHERE {where_sql}
            GROUP BY query
            ORDER BY count DESC, query ASC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -- Search Log --

    def log_search(
        self,
        query: str,
        project_id: str | None,
        domain: str,
        result_chunk_ids: list[int],
        result_count: int,
    ) -> None:
        """Log a search query and its results."""
        conn = self._connect()
        conn.execute(
            """INSERT INTO search_log (query, project_id, domain, result_chunk_ids, result_count)
               VALUES (?, ?, ?, ?, ?)""",
            (query, project_id, domain, json.dumps(result_chunk_ids), result_count),
        )
        conn.commit()

    def get_search_log(self, project_id: str | None = None, limit: int = 100) -> list[dict]:
        """Fetch recent search log entries."""
        conn = self._connect()
        if project_id:
            rows = conn.execute(
                "SELECT * FROM search_log WHERE project_id = ? ORDER BY timestamp DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM search_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_old_search_logs(self, days: int = 90) -> int:
        """Delete search logs older than N days."""
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM search_log WHERE timestamp < datetime('now', ?)",
            (f"-{int(days)} days",),
        )
        conn.commit()
        return cursor.rowcount

    # -- Chunk Content --

    def get_chunk_content(self, chunk_id: int) -> str | None:
        """Fetch full content for a single chunk by ID."""
        conn = self._connect()
        row = conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return row[0] if row else None

    # -- File -> Chunk Mapping --

    def get_chunk_ids_for_file(self, project_id: str, file_path: str) -> list[int]:
        """Get chunk IDs for a specific file in a project."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id FROM chunks WHERE project_id = ? AND file_path = ? ORDER BY id",
            (project_id, file_path),
        ).fetchall()
        return [r[0] for r in rows]

    def get_chunks_for_file(self, project_id: str, file_path: str) -> list[dict]:
        """All chunks for a file with full details, ordered by start_line."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT id, chunk_type, name, language, start_line, end_line, content
               FROM chunks WHERE project_id = ? AND file_path = ?
               ORDER BY start_line""",
            (project_id, file_path),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_file_line_count(self, project_id: str, file_path: str) -> int | None:
        """Max end_line from indexed chunks."""
        conn = self._connect()
        row = conn.execute(
            "SELECT MAX(end_line) FROM chunks WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()
        return row[0] if row and row[0] else None
