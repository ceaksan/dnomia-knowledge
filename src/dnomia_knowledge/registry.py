"""Project configuration and .knowledge.toml parser."""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from dnomia_knowledge.presets import resolve_extensions


class ContentConfig(BaseModel):
    paths: list[str] = []
    extensions: list[str] = Field(default_factory=lambda: [".md", ".mdx"])
    chunking: str = "heading"  # "heading" | "sliding"
    chunk_overlap: int = 0


class CodeConfig(BaseModel):
    preset: str | None = None  # "web" | "python" | "django" | "mixed"
    paths: list[str] = []
    extensions: list[str] | None = None  # explicit override
    max_chunk_lines: int = 50

    @property
    def resolved_extensions(self) -> list[str]:
        return resolve_extensions(self.preset, self.extensions)


class GraphConfig(BaseModel):
    enabled: bool = False
    edge_types: list[str] = Field(default_factory=lambda: ["link", "tag", "category", "semantic"])
    semantic_threshold: float = 0.75


class LinksConfig(BaseModel):
    related: list[str] = []


class IndexingConfig(BaseModel):
    ignore_patterns: list[str] = Field(
        default_factory=lambda: ["node_modules", "dist", ".next", "__pycache__"]
    )
    max_file_size_kb: int = 500
    batch_size: int = 16


class ProjectConfig(BaseModel):
    name: str
    type: str = "content"  # "content" | "saas" | "static"
    content: ContentConfig = Field(default_factory=ContentConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    links: LinksConfig = Field(default_factory=LinksConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)


def load_config(project_path: str | Path) -> ProjectConfig | None:
    """Load .knowledge.toml from project root. Returns None if not found."""
    config_file = Path(project_path) / ".knowledge.toml"
    if not config_file.exists():
        return None

    with open(config_file, "rb") as f:
        data = tomllib.load(f)

    project_data = data.get("project", {})
    config_dict = {
        "name": project_data.get("name", Path(project_path).name),
        "type": project_data.get("type", "content"),
        "content": data.get("content", {}),
        "code": data.get("code", {}),
        "graph": data.get("graph", {}),
        "links": data.get("links", {}),
        "indexing": data.get("indexing", {}),
    }
    return ProjectConfig(**config_dict)


def default_config(project_path: str | Path) -> ProjectConfig:
    """Generate default config when no .knowledge.toml exists."""
    return ProjectConfig(name=Path(project_path).name)


def compute_config_hash(config_path: str | Path) -> str:
    """MD5 hash of the .knowledge.toml file content."""
    with open(config_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()  # noqa: S324
