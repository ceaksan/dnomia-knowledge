"""Language detection and Tree-sitter node type mappings."""

from __future__ import annotations

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml_interface",
    ".zig": "zig",
    ".nim": "nim",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".sql": "sql",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
    ".xml": "xml",
    ".md": "markdown",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".proto": "proto",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".dockerfile": "dockerfile",
    ".nix": "nix",
    ".el": "elisp",
    ".clj": "clojure",
    ".fs": "fsharp",
    ".fsx": "fsharp",
}

# Node types that represent extractable code chunks per language.
# These are the top-level or class-level constructs we want to extract.
CHUNK_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "variable_declaration",
        "expression_statement",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "variable_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "expression_statement",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "variable_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "expression_statement",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "var_declaration",
        "const_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
        "const_item",
        "static_item",
        "type_item",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "method_declaration",
        "constructor_declaration",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "declaration",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
        "declaration",
        "template_declaration",
    },
    "csharp": {
        "class_declaration",
        "interface_declaration",
        "struct_declaration",
        "enum_declaration",
        "method_declaration",
        "constructor_declaration",
        "namespace_declaration",
    },
    "ruby": {
        "method",
        "class",
        "module",
        "singleton_method",
    },
    "php": {
        "function_definition",
        "class_declaration",
        "method_declaration",
        "interface_declaration",
        "trait_declaration",
    },
    "swift": {
        "function_declaration",
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "protocol_declaration",
        "extension_declaration",
    },
    "kotlin": {
        "function_declaration",
        "class_declaration",
        "object_declaration",
        "companion_object",
    },
    "scala": {
        "function_definition",
        "class_definition",
        "object_definition",
        "trait_definition",
    },
    "elixir": {
        "call",  # def, defp, defmodule are all calls in elixir grammar
    },
    "dart": {
        "function_signature",
        "class_definition",
        "method_signature",
    },
    "zig": {
        "TopLevelDecl",
        "FnProto",
    },
}


def _classify_node_type(node_type: str) -> str:
    """Map a tree-sitter node type to a chunk_type category."""
    node_lower = node_type.lower()
    if "function" in node_lower or "method" in node_lower or "fn" in node_lower:
        return "function"
    if "class" in node_lower:
        return "class"
    if "struct" in node_lower:
        return "struct"
    if "interface" in node_lower or "trait" in node_lower or "protocol" in node_lower:
        return "interface"
    if "enum" in node_lower:
        return "enum"
    if "impl" in node_lower:
        return "impl"
    if "module" in node_lower or "namespace" in node_lower or "mod" in node_lower:
        return "module"
    if "type" in node_lower:
        return "type"
    return "block"


def _extract_name(node) -> str:
    """Try to extract the name from a tree-sitter node."""
    # Look for a 'name' field first
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")

    # For decorated definitions, look inside
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_name(child)

    # For export statements, look at the declaration inside
    if node.type == "export_statement":
        decl = node.child_by_field_name("declaration")
        if decl:
            return _extract_name(decl)
        # Arrow function exports: export const foo = ...
        for child in node.children:
            if child.type in ("lexical_declaration", "variable_declaration"):
                return _extract_name(child)

    # For variable/lexical declarations, get the declarator name
    if node.type in ("lexical_declaration", "variable_declaration"):
        for child in node.children:
            if child.type == "variable_declarator":
                name_n = child.child_by_field_name("name")
                if name_n:
                    return name_n.text.decode("utf-8", errors="replace")

    return ""


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension. Returns None if unknown."""
    from pathlib import Path

    p = Path(file_path)
    name = p.name.lower()
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "dockerfile"
    if name == "makefile" or name == "gnumakefile":
        return "make"
    return EXTENSION_TO_LANGUAGE.get(p.suffix.lower())


def get_chunk_node_types(language: str) -> set[str]:
    """Get the set of node types to extract for a language."""
    return CHUNK_NODE_TYPES.get(language, set())


def classify_node(node_type: str) -> str:
    """Classify a tree-sitter node type into a chunk category."""
    return _classify_node_type(node_type)


def extract_name(node) -> str:
    """Extract name from a tree-sitter node."""
    return _extract_name(node)
