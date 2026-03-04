"""Hybrid search: FTS5 + sqlite-vec + RRF merge."""

from __future__ import annotations

import json
import logging
import re

from dnomia_knowledge.embedder import Embedder
from dnomia_knowledge.models import SearchResult
from dnomia_knowledge.store import Store

logger = logging.getLogger(__name__)


def rrf_merge(
    list_a: list[SearchResult],
    list_b: list[SearchResult],
    k: int = 60,
    limit: int = 10,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion merge of two ranked lists."""
    scores: dict[int, float] = {}
    result_map: dict[int, SearchResult] = {}

    for rank, r in enumerate(list_a):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
        result_map[r.chunk_id] = r

    for rank, r in enumerate(list_b):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
        if r.chunk_id not in result_map:
            result_map[r.chunk_id] = r

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]

    results = []
    for chunk_id in sorted_ids:
        r = result_map[chunk_id]
        r.score = round(scores[chunk_id], 6)
        results.append(r)
    return results


def _sanitize_fts_query(query: str) -> str:
    """Remove FTS5 special characters to prevent syntax errors."""
    return re.sub(r"[^\w\s]", " ", query).strip()


class HybridSearch:
    """Hybrid FTS5 + vector search with RRF merge."""

    def __init__(self, store: Store, embedder: Embedder):
        self._store = store
        self._embedder = embedder

    def search(
        self,
        query: str,
        project_id: str | None = None,
        domain: str = "all",
        limit: int = 10,
        language: str | None = None,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            return []

        # Run FTS5 and vector search in sequence (same thread, same DB)
        fetch_limit = limit * 3  # over-fetch for RRF merge quality
        fts_results = self._search_fts(
            query, project_id, domain, fetch_limit, language=language, file_pattern=file_pattern
        )
        vec_results = self._search_vector(
            query, project_id, domain, fetch_limit, language=language, file_pattern=file_pattern
        )

        if not fts_results and not vec_results:
            # Fallback: prefix match
            fts_results = self._search_fts_prefix(
                query, project_id, domain, fetch_limit, language=language, file_pattern=file_pattern
            )

        results = rrf_merge(fts_results, vec_results, k=60, limit=limit)
        results = self._apply_interaction_boost(results, project_id)
        self._log_search_results(query, project_id, domain, results)
        return results

    def _apply_interaction_boost(
        self,
        results: list[SearchResult],
        project_id: str | None,
        weight: float = 0.1,
    ) -> list[SearchResult]:
        """Boost results based on read/edit interaction history."""
        if not results:
            return results

        chunk_ids = [r.chunk_id for r in results]
        try:
            counts = self._store.get_interaction_counts(
                chunk_ids, days=30, interactions=["read", "edit"], project_id=project_id
            )
        except Exception:
            logger.debug("Failed to get interaction counts", exc_info=True)
            return results

        for r in results:
            count = counts.get(r.chunk_id, 0)
            bonus = weight * min(count, 10) / 10
            r.score = round(r.score + bonus, 6)

        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _log_search_results(
        self,
        query: str,
        project_id: str | None,
        domain: str,
        results: list[SearchResult],
    ) -> None:
        """Log search query and mark result chunks as search_hit."""
        try:
            chunk_ids = [r.chunk_id for r in results]
            self._store.log_search(query, project_id, domain, chunk_ids, len(results))
            self._store.batch_log_interactions([(cid, "search_hit", "search") for cid in chunk_ids])
        except Exception:
            logger.debug("Failed to log search results", exc_info=True)

    def _search_fts(
        self,
        query: str,
        project_id: str | None,
        domain: str,
        limit: int,
        language: str | None = None,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []

        conn = self._store._connect()
        where_clauses = []
        params: list = []

        if project_id:
            where_clauses.append("c.project_id = ?")
            params.append(project_id)
        if domain != "all":
            where_clauses.append("c.chunk_domain = ?")
            params.append(domain)
        if language:
            where_clauses.append("c.language = ?")
            params.append(language)
        if file_pattern:
            where_clauses.append("c.file_path LIKE ?")
            params.append(f"%{file_pattern}%")

        where_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT c.id, c.project_id, c.file_path, c.chunk_domain, c.chunk_type,
                   c.name, c.language, c.start_line, c.end_line, c.content,
                   bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            {where_sql}
            ORDER BY score
            LIMIT ?
        """
        params = [sanitized] + params + [limit]

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []

        return [self._row_to_result(r) for r in rows]

    def _search_fts_prefix(
        self,
        query: str,
        project_id: str | None,
        domain: str,
        limit: int,
        language: str | None = None,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        """Layer 2 fallback: prefix match."""
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []
        prefix_query = " ".join(f"{word}*" for word in sanitized.split())
        conn = self._store._connect()

        where_clauses = []
        params: list = []
        if project_id:
            where_clauses.append("c.project_id = ?")
            params.append(project_id)
        if domain != "all":
            where_clauses.append("c.chunk_domain = ?")
            params.append(domain)
        if language:
            where_clauses.append("c.language = ?")
            params.append(language)
        if file_pattern:
            where_clauses.append("c.file_path LIKE ?")
            params.append(f"%{file_pattern}%")
        where_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT c.id, c.project_id, c.file_path, c.chunk_domain, c.chunk_type,
                   c.name, c.language, c.start_line, c.end_line, c.content,
                   bm25(chunks_fts) AS score
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ?
            {where_sql}
            ORDER BY score
            LIMIT ?
        """
        params = [prefix_query] + params + [limit]
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []
        return [self._row_to_result(r) for r in rows]

    def _search_vector(
        self,
        query: str,
        project_id: str | None,
        domain: str,
        limit: int,
        language: str | None = None,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        query_vec = self._embedder.embed_query(query)
        conn = self._store._connect()

        # sqlite-vec KNN search
        vec_sql = """
            SELECT id, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        """
        try:
            vec_rows = conn.execute(vec_sql, (json.dumps(query_vec), limit)).fetchall()
        except Exception:
            return []

        if not vec_rows:
            return []

        # Get chunk details for matched IDs
        chunk_ids = [r[0] for r in vec_rows]
        distances = {r[0]: r[1] for r in vec_rows}
        placeholders = ",".join("?" * len(chunk_ids))

        where_clauses = [f"c.id IN ({placeholders})"]
        params: list = list(chunk_ids)

        if project_id:
            where_clauses.append("c.project_id = ?")
            params.append(project_id)
        if domain != "all":
            where_clauses.append("c.chunk_domain = ?")
            params.append(domain)
        if language:
            where_clauses.append("c.language = ?")
            params.append(language)
        if file_pattern:
            where_clauses.append("c.file_path LIKE ?")
            params.append(f"%{file_pattern}%")

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT c.id, c.project_id, c.file_path, c.chunk_domain, c.chunk_type,
                   c.name, c.language, c.start_line, c.end_line, c.content
            FROM chunks c
            WHERE {where_sql}
        """

        rows = conn.execute(sql, params).fetchall()

        results = []
        for r in rows:
            result = self._row_to_result(r)
            dist = distances.get(r[0], 1.0)
            result.score = round(1.0 / (1.0 + dist), 6)  # distance to similarity
            results.append(result)

        # Sort by score descending (highest similarity first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def search_cross(
        self,
        query: str,
        project_id: str,
        related_projects: list[str],
        domain: str = "all",
        limit: int = 10,
        language: str | None = None,
        file_pattern: str | None = None,
    ) -> list[SearchResult]:
        """Search across primary project and related projects, merge with RRF."""
        all_project_ids = [project_id] + [p for p in related_projects if p != project_id]

        all_results: list[list[SearchResult]] = []
        per_project_limit = max(limit, 10)

        for pid in all_project_ids:
            proj = self._store.get_project(pid)
            if not proj:
                logger.warning("Related project '%s' not found, skipping.", pid)
                continue
            results = self.search(
                query,
                project_id=pid,
                domain=domain,
                limit=per_project_limit,
                language=language,
                file_pattern=file_pattern,
            )
            if results:
                # Normalize scores within each project
                _normalize_scores(results)
                all_results.append(results)

        if not all_results:
            return []

        if len(all_results) == 1:
            return all_results[0][:limit]

        return _rrf_merge_multi(all_results, k=60, limit=limit)

    def _row_to_result(self, row) -> SearchResult:
        keys = set(row.keys())
        content = row["content"] if "content" in keys else ""
        all_lines = content.split("\n")
        snippet = "\n".join(all_lines[:10])
        if len(all_lines) > 10:
            snippet += "\n..."

        return SearchResult(
            chunk_id=row["id"] if "id" in keys else row[0],
            project_id=row["project_id"] if "project_id" in keys else "",
            file_path=row["file_path"] if "file_path" in keys else "",
            chunk_domain=row["chunk_domain"] if "chunk_domain" in keys else "",
            chunk_type=row["chunk_type"] if "chunk_type" in keys else "",
            name=row["name"] if "name" in keys else None,
            language=row["language"] if "language" in keys else None,
            start_line=row["start_line"] if "start_line" in keys else 0,
            end_line=row["end_line"] if "end_line" in keys else 0,
            score=row["score"] if "score" in keys else 0.0,
            snippet=snippet,
        )


def _normalize_scores(results: list[SearchResult]) -> None:
    """Min-max normalize scores in-place."""
    if len(results) <= 1:
        return
    scores = [r.score for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r.score = 1.0
        return
    for r in results:
        r.score = round((r.score - min_s) / (max_s - min_s), 6)


def _rrf_merge_multi(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    limit: int = 10,
) -> list[SearchResult]:
    """Multi-list RRF merge."""
    scores: dict[int, float] = {}
    result_map: dict[int, SearchResult] = {}

    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + 1.0 / (k + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]

    merged = []
    for chunk_id in sorted_ids:
        r = result_map[chunk_id]
        r.score = round(scores[chunk_id], 6)
        merged.append(r)
    return merged
