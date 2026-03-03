"""Tests for extension presets."""

from __future__ import annotations

import pytest

from dnomia_knowledge.presets import PRESETS, resolve_extensions


class TestPresets:
    def test_web_has_ts(self):
        assert ".ts" in PRESETS["web"]

    def test_python_has_py(self):
        assert ".py" in PRESETS["python"]

    def test_resolve_explicit_overrides_preset(self):
        result = resolve_extensions(preset="web", explicit_extensions=[".custom"])
        assert result == [".custom"]

    def test_resolve_mixed_combines(self):
        result = resolve_extensions(preset="mixed")
        assert ".ts" in result and ".py" in result

    def test_resolve_none_returns_empty(self):
        assert resolve_extensions() == []

    def test_resolve_deduplicates(self):
        result = resolve_extensions(preset="django")
        assert len(result) == len(set(result))

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError):
            resolve_extensions(preset="nonexistent")
