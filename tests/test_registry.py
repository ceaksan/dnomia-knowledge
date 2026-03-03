"""Tests for .knowledge.toml parser and ProjectConfig."""

from __future__ import annotations

import pytest

from dnomia_knowledge.registry import (
    CodeConfig,
    compute_config_hash,
    default_config,
    load_config,
)


class TestLoadConfig:
    def test_valid_config(self, tmp_dir):
        (tmp_dir / ".knowledge.toml").write_bytes(b'[project]\nname = "test"\ntype = "saas"\n')
        config = load_config(tmp_dir)
        assert config is not None
        assert config.name == "test"
        assert config.type == "saas"

    def test_returns_none_when_missing(self, tmp_dir):
        assert load_config(tmp_dir) is None

    def test_full_config(self, tmp_dir):
        toml = b"""
[project]
name = "ceaksan"
type = "content"

[content]
paths = ["src/content/"]
extensions = [".mdx", ".md"]
chunk_overlap = 2

[code]
preset = "web"
paths = ["src/"]

[graph]
enabled = true
semantic_threshold = 0.8

[links]
related = ["dnomia-app"]

[indexing]
ignore_patterns = ["node_modules", "dist"]
max_file_size_kb = 300
batch_size = 8
"""
        (tmp_dir / ".knowledge.toml").write_bytes(toml)
        config = load_config(tmp_dir)
        assert config.content.paths == ["src/content/"]
        assert config.content.chunk_overlap == 2
        assert ".ts" in config.code.resolved_extensions
        assert config.graph.enabled is True
        assert config.links.related == ["dnomia-app"]
        assert config.indexing.max_file_size_kb == 300

    def test_partial_config_uses_defaults(self, tmp_dir):
        (tmp_dir / ".knowledge.toml").write_bytes(b'[project]\nname = "minimal"\n')
        config = load_config(tmp_dir)
        assert config.content.extensions == [".md", ".mdx"]
        assert config.graph.enabled is False
        assert config.indexing.max_file_size_kb == 500

    def test_default_config(self, tmp_dir):
        config = default_config(tmp_dir)
        assert config.name == tmp_dir.name
        assert config.type == "content"

    def test_code_resolved_extensions_preset(self):
        code = CodeConfig(preset="web")
        assert ".ts" in code.resolved_extensions

    def test_code_resolved_extensions_explicit(self):
        code = CodeConfig(preset="web", extensions=[".go"])
        assert code.resolved_extensions == [".go"]

    def test_config_hash_changes(self, tmp_dir):
        p = tmp_dir / ".knowledge.toml"
        p.write_bytes(b'[project]\nname = "v1"\n')
        h1 = compute_config_hash(p)
        p.write_bytes(b'[project]\nname = "v2"\n')
        h2 = compute_config_hash(p)
        assert h1 != h2

    def test_invalid_toml_raises(self, tmp_dir):
        (tmp_dir / ".knowledge.toml").write_bytes(b"invalid = = = toml")
        with pytest.raises(Exception):
            load_config(tmp_dir)
