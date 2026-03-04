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


class TestLargeNodeSplitting:
    def test_large_function_split_into_subchunks(self):
        chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)
        body = "\n".join(f"    x = {i}" for i in range(30))
        code = f"def big_func():\n{body}\n"
        chunks = chunker.chunk("big.py", code)
        func_chunks = [c for c in chunks if c.name == "big_func"]
        assert len(func_chunks) > 1
        for c in func_chunks:
            assert c.chunk_type == "function"
            assert c.language == "python"

    def test_subchunks_have_correct_line_numbers(self):
        chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)
        body = "\n".join(f"    x = {i}" for i in range(30))
        code = f"def big_func():\n{body}\n"
        chunks = chunker.chunk("big.py", code)
        func_chunks = [c for c in chunks if c.name == "big_func"]
        assert func_chunks[0].start_line == 1
        for c in func_chunks:
            assert c.start_line >= 1
            assert c.end_line >= c.start_line

    def test_small_function_not_split(self):
        chunker = AstChunker(max_chunk_lines=50, overlap_lines=2)
        code = "def small():\n    return 1\n"
        chunks = chunker.chunk("small.py", code)
        funcs = [c for c in chunks if c.chunk_type == "function"]
        assert len(funcs) == 1

    def test_large_module_fallback_split(self):
        chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)
        code = "\n".join(f"x_{i} = {i}" for i in range(30))
        chunks = chunker.chunk("consts.py", code)
        assert len(chunks) > 1
        for c in chunks:
            assert c.chunk_type == "module"
            assert c.name == "consts"

    def test_large_ts_export_split(self):
        chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)
        entries = ",\n".join(f'  key{i}: "value{i}"' for i in range(30))
        code = f"export const TRANSLATIONS = {{\n{entries}\n}};\n"
        chunks = chunker.chunk("consts.ts", code)
        assert len(chunks) > 1
        for c in chunks:
            assert c.language == "typescript"


class TestAstroChunking:
    def test_simple_astro_file(self):
        chunker = AstChunker()
        code = (
            "---\n"
            "const title = 'Hello';\n"
            "---\n"
            "<h1>{title}</h1>\n"
            "<p>Content here</p>\n"
            "<script>\n"
            "console.log('hi');\n"
            "</script>\n"
        )
        chunks = chunker.chunk("page.astro", code)
        names = [c.name for c in chunks]
        assert "frontmatter" in names
        assert "template" in names
        assert "script" in names
        for c in chunks:
            assert c.language == "astro"

    def test_no_frontmatter(self):
        chunker = AstChunker()
        code = "<div>Hello</div>\n<p>World</p>\n"
        chunks = chunker.chunk("simple.astro", code)
        assert len(chunks) >= 1
        names = [c.name for c in chunks]
        assert "frontmatter" not in names
        assert "template" in names

    def test_large_template_split(self):
        chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)
        fm = "---\nconst x = 1;\n---\n"
        template = "\n".join(f"<p>line {i}</p>" for i in range(30))
        code = fm + template + "\n"
        chunks = chunker.chunk("big.astro", code)
        fm_chunks = [c for c in chunks if c.name == "frontmatter"]
        tmpl_chunks = [c for c in chunks if c.name and c.name.startswith("template")]
        assert len(fm_chunks) == 1
        assert len(tmpl_chunks) > 1

    def test_script_and_style_blocks(self):
        chunker = AstChunker()
        code = (
            "---\nconst x = 1;\n---\n"
            "<div>template</div>\n"
            "<script>\nconst y = 2;\n</script>\n"
            "<style>\n.cls { color: red; }\n</style>\n"
        )
        chunks = chunker.chunk("styled.astro", code)
        names = [c.name for c in chunks]
        assert "frontmatter" in names
        assert "template" in names
        assert "script" in names
        assert "style" in names

    def test_astro_chunk_line_numbers(self):
        chunker = AstChunker()
        code = "---\nconst title = 'Hello';\n---\n<h1>{title}</h1>\n"
        chunks = chunker.chunk("page.astro", code)
        fm = [c for c in chunks if c.name == "frontmatter"][0]
        assert fm.start_line == 1
        assert fm.end_line == 3

    def test_unclosed_script_tag(self):
        chunker = AstChunker()
        code = "---\nconst x = 1;\n---\n<div>template</div>\n<script>\nconst y = 2;\n"
        chunks = chunker.chunk("broken.astro", code)
        assert len(chunks) >= 1
        scripts = [c for c in chunks if c.name == "script"]
        assert len(scripts) == 1
        # Unclosed script consumes to end of file (trailing \n = empty line 7)
        assert scripts[0].end_line == 7

    def test_multiple_script_blocks(self):
        chunker = AstChunker()
        code = (
            "---\nconst x = 1;\n---\n"
            "<div>template</div>\n"
            "<script>\nconst a = 1;\n</script>\n"
            "<script is:inline>\nconst b = 2;\n</script>\n"
        )
        chunks = chunker.chunk("multi.astro", code)
        scripts = [c for c in chunks if c.name == "script"]
        assert len(scripts) == 2

    def test_frontmatter_only_no_template(self):
        chunker = AstChunker()
        code = "---\nconst x = 1;\n---\n"
        chunks = chunker.chunk("fm.astro", code)
        names = [c.name for c in chunks]
        assert "frontmatter" in names
        assert "template" not in names

    def test_hr_not_detected_as_frontmatter(self):
        chunker = AstChunker()
        code = "<p>Hello</p>\n---\n<p>World</p>\n---\n<p>End</p>\n"
        chunks = chunker.chunk("hr.astro", code)
        names = [c.name for c in chunks]
        # --- not on line 0, should not be treated as frontmatter
        assert "frontmatter" not in names
        assert "template" in names


class TestAstChunkerValidation:
    def test_overlap_equals_max_raises(self):
        with pytest.raises(ValueError, match="overlap_lines must be less than max_chunk_lines"):
            AstChunker(max_chunk_lines=5, overlap_lines=5)

    def test_overlap_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="overlap_lines must be less than max_chunk_lines"):
            AstChunker(max_chunk_lines=5, overlap_lines=10)

    def test_zero_overlap_works(self):
        chunker = AstChunker(max_chunk_lines=5, overlap_lines=0)
        code = "\n".join(f"line {i}" for i in range(12))
        chunks = chunker.chunk("data.xyz", code)
        assert len(chunks) >= 2
        # With zero overlap, chunks should not overlap
        assert chunks[1].start_line > chunks[0].end_line


class TestLargeClassSplitting:
    def test_large_class_split_skips_child_extraction(self):
        chunker = AstChunker(max_chunk_lines=10, overlap_lines=2)
        methods = "\n".join(f"    def method_{i}(self):\n        return {i}\n" for i in range(15))
        code = f"class BigClass:\n{methods}"
        chunks = chunker.chunk("big_class.py", code)
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        # Class was split, so no separate method chunks should exist
        assert len(class_chunks) > 1
        assert len(method_chunks) == 0

    def test_small_class_still_extracts_methods(self):
        chunker = AstChunker(max_chunk_lines=50, overlap_lines=2)
        code = (
            "class SmallClass:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "\n"
            "    def greet(self):\n"
            '        return "hi"\n'
        )
        chunks = chunker.chunk("small_class.py", code)
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(class_chunks) == 1
        assert len(method_chunks) == 2
