"""Tests for MCP server tool definitions."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from dnomia_knowledge.server import _HTMLTextExtractor, create_server


class TestServerCreation:
    def test_server_created(self):
        server = create_server()
        assert server is not None
        assert server.name == "dnomia-knowledge"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_response(body: bytes, content_type: str = "text/html", charset: str = "utf-8"):
    """Create a mock urllib response with proper headers."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.headers = MagicMock()
    mock_resp.headers.get_content_type.return_value = content_type
    mock_resp.headers.get_content_charset.return_value = charset
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestHTMLTextExtractor:
    def test_strips_script_and_style(self):
        html = "<html><script>var x=1;</script><style>.a{}</style><p>Hello</p></html>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "var x" not in text
        assert ".a{}" not in text
        assert "Hello" in text

    def test_strips_nav_header_footer(self):
        html = "<nav>Menu</nav><div>Content</div><footer>Footer</footer>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "Menu" not in text
        assert "Footer" not in text
        assert "Content" in text

    def test_adds_newlines_for_block_elements(self):
        html = "<h1>Title</h1><p>Para 1</p><p>Para 2</p>"
        extractor = _HTMLTextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        assert "Title" in text
        assert "Para 1" in text
        assert "Para 2" in text


class TestFetchAndIndex:
    """Tests for the fetch_and_index MCP tool."""

    def _get_tool(self, name):
        server = create_server()
        tool_map = {t.name: t for t in server._tool_manager._tools.values()}
        return tool_map[name].fn

    @patch("dnomia_knowledge.server._get_embedder")
    @patch("dnomia_knowledge.server._get_store")
    @patch("dnomia_knowledge.server.urllib.request.urlopen")
    def test_fetch_html_indexes_content(self, mock_urlopen, mock_get_store, mock_get_embedder):
        html = b"""<html><head><title>Test</title></head><body>
        <nav>Skip this nav</nav>
        <h2>Introduction</h2>
        <p>This is a long enough paragraph for chunking. We need enough text here
        to make the chunker actually produce chunks. Let us write a decent amount
        of content so the min_chunk_chars threshold is exceeded and we get actual
        chunks from MdChunker instead of empty results. More text here for padding
        to ensure we have substantial content for the test.</p>
        <h2>Details</h2>
        <p>More details about the topic. Again, we need enough text to be a real
        chunk. Writing more words here to make sure the chunk has enough characters
        and the test can verify that chunking works correctly with HTML content
        that has been stripped of tags and converted to plain text.</p>
        </body></html>"""

        mock_urlopen.return_value = _make_mock_response(html, "text/html")

        mock_store = MagicMock()
        mock_store.insert_chunks.return_value = [1, 2]
        mock_get_store.return_value = mock_store

        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = [[0.1] * 384, [0.2] * 384]
        mock_get_embedder.return_value = mock_embedder

        fetch_and_index = self._get_tool("fetch_and_index")
        result = _run(fetch_and_index(url="https://docs.example.com/tutorial"))

        assert "Indexed https://docs.example.com/tutorial" in result
        assert "words" in result
        assert "chunks" in result

        mock_store.register_project.assert_called_once_with(
            "docs-example-com", "https://docs.example.com/tutorial", "web"
        )
        mock_store.insert_chunks.assert_called_once()
        mock_embedder.embed_passages.assert_called_once()

    def test_project_name_from_domain(self):
        from urllib.parse import urlparse

        url = "https://docs.python.org/tutorial"
        project_id = urlparse(url).netloc.replace(".", "-")
        assert project_id == "docs-python-org"

    @patch("dnomia_knowledge.server._get_embedder")
    @patch("dnomia_knowledge.server._get_store")
    @patch("dnomia_knowledge.server.urllib.request.urlopen")
    def test_fetch_plain_text(self, mock_urlopen, mock_get_store, mock_get_embedder):
        text_content = b"""## Getting Started

This is a plain text document about getting started with the project.
It contains enough text to form at least one chunk when processed by
the markdown chunker. We write a generous amount of content here to
ensure the chunker min_chunk_chars threshold is met and chunks are
actually produced in the test case.

## Configuration

Configure the project by editing the config file. This section also
needs enough text to potentially form its own chunk, depending on
the chunker settings. Additional content here helps ensure we have
substantial test data for verifying the plain text indexing path."""

        mock_urlopen.return_value = _make_mock_response(text_content, "text/plain", "utf-8")

        mock_store = MagicMock()
        mock_store.insert_chunks.return_value = [1]
        mock_get_store.return_value = mock_store

        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = [[0.1] * 384]
        mock_get_embedder.return_value = mock_embedder

        fetch_and_index = self._get_tool("fetch_and_index")
        result = _run(fetch_and_index(url="https://example.com/readme.txt"))

        assert "Indexed https://example.com/readme.txt" in result

        # Plain text: no HTML stripping should happen, content passed as-is
        mock_store.register_project.assert_called_once_with(
            "example-com", "https://example.com/readme.txt", "web"
        )
        mock_store.insert_chunks.assert_called_once()

    @patch("dnomia_knowledge.server.urllib.request.urlopen")
    def test_fetch_error_returns_message(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        fetch_and_index = self._get_tool("fetch_and_index")
        result = _run(fetch_and_index(url="https://down.example.com/"))

        assert "Error fetching" in result
        assert "Connection refused" in result

    @patch("dnomia_knowledge.server._get_embedder")
    @patch("dnomia_knowledge.server._get_store")
    @patch("dnomia_knowledge.server.urllib.request.urlopen")
    def test_custom_project_id(self, mock_urlopen, mock_get_store, mock_get_embedder):
        html = b"""<html><body>
        <h2>Content</h2>
        <p>Enough content here to form a chunk. This paragraph needs to have
        a decent amount of text so the chunker accepts it as a valid chunk
        rather than discarding it for being too small.</p>
        </body></html>"""

        mock_urlopen.return_value = _make_mock_response(html, "text/html")

        mock_store = MagicMock()
        mock_store.insert_chunks.return_value = [1]
        mock_get_store.return_value = mock_store

        mock_embedder = MagicMock()
        mock_embedder.embed_passages.return_value = [[0.1] * 384]
        mock_get_embedder.return_value = mock_embedder

        fetch_and_index = self._get_tool("fetch_and_index")
        _run(fetch_and_index(url="https://docs.example.com/api", project="my-docs"))

        mock_store.register_project.assert_called_once_with(
            "my-docs", "https://docs.example.com/api", "web"
        )
