"""Knowledge graph builder: edges, community detection, PageRank."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import PurePosixPath

import networkx as nx

from dnomia_knowledge.registry import GraphConfig
from dnomia_knowledge.store import Store

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PY_IMPORT_RE = re.compile(r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
    re.MULTILINE,
)


class GraphBuilder:
    """Builds and manages a knowledge graph over indexed chunks."""

    def __init__(self, store: Store, config: GraphConfig | None = None) -> None:
        self.store = store
        self.config = config or GraphConfig()

    # -- Public API --

    def build_edges_for_file(self, project_id: str, file_path: str, chunk_ids: list[int]) -> int:
        """Build link, tag, and category edges for chunks in a file."""
        if not self.config.enabled:
            return 0

        chunks = self._load_chunks(chunk_ids)
        if not chunks:
            return 0

        total = 0
        if "link" in self.config.edge_types:
            total += self._build_link_edges(project_id, file_path, chunks)
        if "tag" in self.config.edge_types:
            total += self._build_tag_edges(chunks)
        if "category" in self.config.edge_types:
            total += self._build_category_edges(chunks)
        return total

    def build_semantic_edges(self, project_id: str, chunk_ids: list[int] | None = None) -> int:
        """Build semantic similarity edges via KNN from sqlite-vec."""
        if not self.config.enabled:
            return 0
        if "semantic" not in self.config.edge_types:
            return 0

        if chunk_ids is None:
            chunk_ids = self.store.get_chunk_ids_for_project(project_id)
        if not chunk_ids:
            return 0

        edges: list[dict] = []
        threshold = self.config.semantic_threshold

        for cid in chunk_ids:
            vec_row = self.store.execute_sql(
                "SELECT embedding FROM chunks_vec WHERE id = ?", (cid,)
            ).fetchone()
            if vec_row is None:
                continue

            query_vec = vec_row[0]
            # sqlite-vec KNN query
            rows = self.store.fetchall(
                """SELECT id, distance
                   FROM chunks_vec
                   WHERE embedding MATCH ?
                   AND k = ?""",
                (query_vec, 20),
            )

            for row in rows:
                neighbor_id = row[0]
                distance = row[1]
                if neighbor_id == cid:
                    continue
                # L2 distance -> cosine similarity for normalized vectors
                similarity = 1 - (distance**2 / 2)
                if similarity >= threshold:
                    edges.append(
                        {
                            "source_id": cid,
                            "target_id": neighbor_id,
                            "edge_type": "semantic",
                            "weight": round(similarity, 4),
                        }
                    )

        return self.store.insert_edges(edges) if edges else 0

    def build_import_edges(self, project_id: str, file_path: str, chunk_ids: list[int]) -> int:
        """Build import/require edges from code chunks."""
        if not self.config.enabled:
            return 0
        if "import" not in self.config.edge_types:
            return 0

        chunks = self._load_chunks(chunk_ids)
        if not chunks:
            return 0

        edges: list[dict] = []

        for chunk in chunks:
            if chunk["chunk_domain"] != "code":
                continue

            content = chunk["content"]
            import_targets = self._parse_imports(content, file_path)

            for target_path in import_targets:
                # Find chunks in the target file
                target_rows = self.store.fetchall(
                    "SELECT id FROM chunks WHERE project_id = ? AND file_path = ?",
                    (project_id, target_path),
                )
                for tr in target_rows:
                    edges.append(
                        {
                            "source_id": chunk["id"],
                            "target_id": tr[0],
                            "edge_type": "import",
                            "weight": 1.0,
                        }
                    )

        return self.store.insert_edges(edges) if edges else 0

    def rebuild_all_edges(self, project_id: str) -> dict[str, int]:
        """Delete all edges and rebuild everything."""
        if not self.config.enabled:
            return {}

        self.store.delete_edges_for_project(project_id)

        files = self.store.fetchall(
            "SELECT DISTINCT file_path FROM chunks WHERE project_id = ?",
            (project_id,),
        )

        counts: dict[str, int] = {
            "link": 0,
            "tag": 0,
            "category": 0,
            "semantic": 0,
            "import": 0,
        }

        for row in files:
            fp = row[0]
            chunk_rows = self.store.fetchall(
                "SELECT id FROM chunks WHERE project_id = ? AND file_path = ?",
                (project_id, fp),
            )
            cids = [r[0] for r in chunk_rows]

            chunks = self._load_chunks(cids)
            if not chunks:
                continue

            if "link" in self.config.edge_types:
                counts["link"] += self._build_link_edges(project_id, fp, chunks)
            if "tag" in self.config.edge_types:
                counts["tag"] += self._build_tag_edges(chunks)
            if "category" in self.config.edge_types:
                counts["category"] += self._build_category_edges(chunks)
            if "import" in self.config.edge_types:
                for chunk in chunks:
                    if chunk["chunk_domain"] != "code":
                        continue
                    import_targets = self._parse_imports(chunk["content"], fp)
                    import_edges: list[dict] = []
                    for target_path in import_targets:
                        target_rows = self.store.fetchall(
                            "SELECT id FROM chunks WHERE project_id = ? AND file_path = ?",
                            (project_id, target_path),
                        )
                        for tr in target_rows:
                            import_edges.append(
                                {
                                    "source_id": chunk["id"],
                                    "target_id": tr[0],
                                    "edge_type": "import",
                                    "weight": 1.0,
                                }
                            )
                    if import_edges:
                        counts["import"] += self.store.insert_edges(import_edges)

        if "semantic" in self.config.edge_types:
            counts["semantic"] += self.build_semantic_edges(project_id)

        return counts

    def run_community_detection(self, project_id: str) -> int:
        """Run Louvain community detection and PageRank, write to chunk metadata."""
        if not self.config.enabled:
            return 0

        edges = self.store.get_edges_for_project(project_id)
        chunk_ids = self.store.get_chunk_ids_for_project(project_id)
        if not chunk_ids:
            return 0

        # Build undirected graph for Louvain
        G = nx.Graph()
        G.add_nodes_from(chunk_ids)
        for e in edges:
            G.add_edge(e["source_id"], e["target_id"], weight=e["weight"])

        # Louvain communities
        communities = nx.community.louvain_communities(G, resolution=1.0, seed=42)

        # Build directed graph for PageRank
        G_directed = nx.DiGraph()
        G_directed.add_nodes_from(chunk_ids)
        for e in edges:
            G_directed.add_edge(e["source_id"], e["target_id"], weight=e["weight"])

        pagerank = nx.pagerank(G_directed, alpha=0.85)

        # Write results to chunk metadata (batched)
        updates: list[tuple[int, dict]] = []
        for community_idx, community in enumerate(communities):
            for cid in community:
                updates.append(
                    (
                        cid,
                        {
                            "community_id": community_idx,
                            "pagerank": round(pagerank.get(cid, 0.0), 6),
                        },
                    )
                )
        self.store.batch_update_chunk_metadata(updates)

        return len(communities)

    # -- Private helpers --

    def _load_chunks(self, chunk_ids: list[int]) -> list[dict]:
        """Load chunk rows by IDs."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.store.fetchall(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        )
        return [dict(r) for r in rows]

    def _build_link_edges(self, project_id: str, file_path: str, chunks: list[dict]) -> int:
        """Parse markdown links and create link edges."""
        edges: list[dict] = []

        for chunk in chunks:
            content = chunk["content"]
            for _text, href in _MD_LINK_RE.findall(content):
                # Skip external URLs and anchors
                if href.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                # Resolve relative path
                base = PurePosixPath(file_path).parent
                resolved = str(base / href)
                # Normalize
                resolved = str(PurePosixPath(resolved))

                # Find target chunks
                target_rows = self.store.fetchall(
                    "SELECT id FROM chunks WHERE project_id = ? AND file_path = ?",
                    (project_id, resolved),
                )
                for tr in target_rows:
                    edges.append(
                        {
                            "source_id": chunk["id"],
                            "target_id": tr[0],
                            "edge_type": "link",
                            "weight": 1.0,
                        }
                    )

        return self.store.insert_edges(edges) if edges else 0

    def _build_tag_edges(self, chunks: list[dict]) -> int:
        """Chunks sharing 2+ tags get a tag edge."""
        chunk_tags: list[tuple[int, set[str]]] = []
        for chunk in chunks:
            meta = json.loads(chunk["metadata"]) if chunk.get("metadata") else {}
            tags = meta.get("tags", [])
            if tags:
                chunk_tags.append((chunk["id"], set(tags)))

        edges: list[dict] = []
        for (id_a, tags_a), (id_b, tags_b) in combinations(chunk_tags, 2):
            shared = tags_a & tags_b
            if len(shared) >= 2:
                total_unique = len(tags_a | tags_b)
                weight = len(shared) / total_unique if total_unique > 0 else 0.0
                edges.append(
                    {
                        "source_id": id_a,
                        "target_id": id_b,
                        "edge_type": "tag",
                        "weight": round(weight, 4),
                    }
                )

        return self.store.insert_edges(edges) if edges else 0

    def _build_category_edges(self, chunks: list[dict]) -> int:
        """Chunks in same category get a category edge."""
        cat_map: dict[str, list[int]] = {}
        for chunk in chunks:
            meta = json.loads(chunk["metadata"]) if chunk.get("metadata") else {}
            categories = meta.get("categories", [])
            for cat in categories:
                cat_map.setdefault(cat, []).append(chunk["id"])

        edges: list[dict] = []
        for _cat, cids in cat_map.items():
            for id_a, id_b in combinations(cids, 2):
                edges.append(
                    {
                        "source_id": id_a,
                        "target_id": id_b,
                        "edge_type": "category",
                        "weight": 1.0,
                    }
                )

        return self.store.insert_edges(edges) if edges else 0

    def _parse_imports(self, content: str, file_path: str) -> list[str]:
        """Parse import/require statements and resolve to file paths."""
        targets: list[str] = []
        base = PurePosixPath(file_path).parent

        # Python imports
        for match in _PY_IMPORT_RE.finditer(content):
            module = match.group(1) or match.group(2)
            # Convert dotted module to path
            path = module.replace(".", "/") + ".py"
            targets.append(str(base / path))

        # JS/TS imports
        for match in _JS_IMPORT_RE.finditer(content):
            module = match.group(1) or match.group(2)
            # Skip node_modules
            if not module.startswith("."):
                continue
            resolved = str(base / module)
            # Add common extensions if none
            if not PurePosixPath(resolved).suffix:
                for ext in (".ts", ".tsx", ".js", ".jsx"):
                    targets.append(resolved + ext)
            else:
                targets.append(resolved)

        return targets
