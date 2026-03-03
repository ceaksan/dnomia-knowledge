"""Tests for MCP server tool definitions."""

from __future__ import annotations

from dnomia_knowledge.server import create_server


class TestServerCreation:
    def test_server_created(self):
        server = create_server()
        assert server is not None
        assert server.name == "dnomia-knowledge"
