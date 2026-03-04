"""Tests for AST-based code chunker."""

from __future__ import annotations

import pytest

from dnomia_knowledge.chunker.ast_chunker import AstChunker
from dnomia_knowledge.models import Chunk


class TestAstChunkerPython:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = AstChunker()

    def test_python_function_extracted(self):
        code = 'def hello():\n    return "world"\n'
        chunks = self.chunker.chunk("example.py", code)
        funcs = [c for c in chunks if c.chunk_type == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "hello"

    def test_python_class_with_methods(self):
        code = (
            "class MyClass:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "\n"
            "    def greet(self):\n"
            '        return "hi"\n'
        )
        chunks = self.chunker.chunk("example.py", code)
        classes = [c for c in chunks if c.chunk_type == "class"]
        methods = [c for c in chunks if c.chunk_type == "method"]
        assert len(classes) == 1
        assert classes[0].name == "MyClass"
        assert len(methods) == 2
        method_names = {m.name for m in methods}
        assert "__init__" in method_names
        assert "greet" in method_names

    def test_decorated_function_includes_decorator(self):
        code = "@app.route('/hello')\ndef hello():\n    return 'world'\n"
        chunks = self.chunker.chunk("example.py", code)
        func_chunks = [c for c in chunks if c.chunk_type in ("function", "block")]
        assert len(func_chunks) >= 1
        # The chunk should include the decorator line
        combined = "\n".join(c.content for c in func_chunks)
        assert "@app.route" in combined

    def test_language_field_set_correctly(self):
        code = "def foo():\n    pass\n"
        chunks = self.chunker.chunk("example.py", code)
        for c in chunks:
            assert c.language == "python"

    def test_line_numbers_1_indexed(self):
        code = "def foo():\n    pass\n"
        chunks = self.chunker.chunk("example.py", code)
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_no_extractable_nodes_produces_module_chunk(self):
        code = "x = 1\ny = 2\nprint(x + y)\n"
        chunks = self.chunker.chunk("example.py", code)
        assert len(chunks) >= 1
        module_chunks = [c for c in chunks if c.chunk_type == "module"]
        assert len(module_chunks) == 1
        assert module_chunks[0].name == "example"


class TestAstChunkerTypeScript:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = AstChunker()

    def test_typescript_interface_extracted(self):
        code = "interface User {\n  name: string;\n  age: number;\n}\n"
        chunks = self.chunker.chunk("types.ts", code)
        ifaces = [c for c in chunks if c.chunk_type == "interface"]
        assert len(ifaces) == 1
        assert ifaces[0].name == "User"
        assert ifaces[0].language == "typescript"

    def test_typescript_function_extracted(self):
        code = "function greet(name: string): string {\n  return `Hello ${name}`;\n}\n"
        chunks = self.chunker.chunk("utils.ts", code)
        funcs = [c for c in chunks if c.chunk_type == "function"]
        assert len(funcs) == 1
        assert funcs[0].name == "greet"


class TestAstChunkerFallback:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)

    def test_unknown_extension_falls_back_to_sliding_window(self):
        code = "\n".join(f"line {i}" for i in range(20))
        chunks = self.chunker.chunk("data.xyz", code)
        assert len(chunks) >= 1
        assert all(c.chunk_type == "block" for c in chunks)

    def test_invalid_syntax_falls_back(self):
        code = "def broken(\n  this is not valid python at all {{{{{\n"
        chunks = self.chunker.chunk("broken.py", code)
        # Should not crash, should return chunks via fallback or AST
        assert len(chunks) >= 1

    def test_sliding_window_overlap(self):
        chunker = AstChunker(max_chunk_lines=5, overlap_lines=2)
        code = "\n".join(f"line {i}" for i in range(12))
        chunks = chunker.chunk("data.xyz", code)
        assert len(chunks) >= 2
        # Second chunk should start before first chunk ends (overlap)
        if len(chunks) >= 2:
            assert chunks[1].start_line <= chunks[0].end_line + 1


class TestAstChunkerEdgeCases:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.chunker = AstChunker()

    def test_empty_file_returns_empty(self):
        assert self.chunker.chunk("empty.py", "") == []

    def test_whitespace_only_returns_empty(self):
        assert self.chunker.chunk("space.py", "   \n\n  \n") == []

    def test_returns_chunk_model(self):
        code = "def foo():\n    pass\n"
        chunks = self.chunker.chunk("example.py", code)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_small_file_single_block(self):
        code = "just a line"
        chunks = self.chunker.chunk("file.xyz", code)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "block"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1
