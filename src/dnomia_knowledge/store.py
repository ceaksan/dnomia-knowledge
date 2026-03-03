"""SQLite store with FTS5 and sqlite-vec support."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = "1"

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
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
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

    def _init_db(self) -> None:
        conn = self._connect()
        conn.executescript(_TABLES_SQL)
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
            "INSERT OR IGNORE INTO system_metadata (key, value) VALUES ('schema_version', ?)",
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

    # -- Chunks --

    def insert_chunks(self, project_id: str, chunks: list[dict]) -> list[int]:
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
        conn.commit()
        return ids

    def insert_chunk_vectors(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        conn = self._connect()
        for chunk_id, vec in zip(chunk_ids, vectors):
            conn.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, json.dumps(vec)),
            )
        conn.commit()

    def delete_file_chunks(self, project_id: str, file_path: str) -> int:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM chunks WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        )
        conn.commit()
        return cursor.rowcount

    def get_project_stats(self, project_id: str) -> dict:
        conn = self._connect()
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        content = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ? AND chunk_domain = 'content'",
            (project_id,),
        ).fetchone()[0]
        code = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE project_id = ? AND chunk_domain = 'code'",
            (project_id,),
        ).fetchone()[0]
        files = conn.execute(
            "SELECT COUNT(DISTINCT file_path) FROM chunks WHERE project_id = ?",
            (project_id,),
        ).fetchone()[0]
        return {
            "total_chunks": total,
            "content_chunks": content,
            "code_chunks": code,
            "total_files": files,
        }

    # -- File Index --

    def upsert_file_index(
        self, project_id: str, file_path: str, file_hash: str, chunk_count: int
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
