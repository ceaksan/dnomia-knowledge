"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def db_path(tmp_dir):
    """Temporary database path."""
    return str(tmp_dir / "test.db")


@pytest.fixture
def tmp_store(db_path):
    """Store instance backed by a temp DB."""
    from dnomia_knowledge.store import Store

    store = Store(db_path)
    yield store
    store.close()


@pytest.fixture(scope="session")
def shared_embedder():
    """Session-scoped embedder to avoid reloading the model per test."""
    from dnomia_knowledge.embedder import Embedder

    return Embedder()


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return (
        "---\n"
        "title: Test Article\n"
        "tags: [python, testing]\n"
        "categories: [tutorial]\n"
        "description: A test article about Python testing\n"
        "---\n"
        "\n"
        "## Introduction\n"
        "\n"
        "This is a test article about Python testing.\n"
        "It covers the basics of pytest and how to use it\n"
        "effectively in your projects. Understanding testing\n"
        "fundamentals is crucial for writing reliable software.\n"
        "We will explore various testing patterns and best\n"
        "practices that help maintain code quality over time.\n"
        "\n"
        "## Writing Tests\n"
        "\n"
        "Here is how you write a test. First you need to\n"
        "understand the basic structure of a pytest test\n"
        "function. Each test should focus on a single behavior\n"
        "and have a clear assertion that verifies the expected\n"
        "outcome.\n"
        "\n"
        "```python\n"
        "def test_example():\n"
        "    assert 1 + 1 == 2\n"
        "```\n"
        "\n"
        "Always use descriptive test names that explain what\n"
        "behavior is being verified. Good test names serve as\n"
        "documentation for the codebase and make it easier to\n"
        "understand failures when they occur.\n"
        "\n"
        "## Running Tests\n"
        "\n"
        "You can run tests with pytest using various command\n"
        "line options. The verbose flag shows individual test\n"
        "results and helps identify which specific tests are\n"
        "passing or failing during development.\n"
        "\n"
        "```bash\n"
        "pytest -v\n"
        "```\n"
        "\n"
        "You can also run specific test files or test functions\n"
        "by providing the path. Use the -k flag to filter tests\n"
        "by name pattern, which is useful when working on a\n"
        "specific feature or debugging a particular test failure.\n"
        "\n"
        "## Conclusion\n"
        "\n"
        "Testing is important for code quality and long-term\n"
        "maintainability of software projects. A comprehensive\n"
        "test suite gives developers confidence to refactor and\n"
        "extend code without fear of breaking existing\n"
        "functionality. Invest time in writing good tests and\n"
        "your future self will thank you.\n"
    )


@pytest.fixture
def sample_markdown_file(tmp_dir, sample_markdown):
    """Write sample markdown to a file and return the path."""
    p = tmp_dir / "test-article.md"
    p.write_text(sample_markdown)
    return p
