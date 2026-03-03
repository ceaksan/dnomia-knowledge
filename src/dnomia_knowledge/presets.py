"""Extension presets for code file types."""

from __future__ import annotations

PRESETS: dict[str, list[str]] = {
    "web": [
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".astro",
        ".vue",
        ".svelte",
        ".css",
        ".scss",
        ".html",
    ],
    "python": [".py", ".pyi"],
    "django": [".py", ".pyi", ".html", ".txt"],
    "mixed": [],  # web + python, resolved at runtime
}


def resolve_extensions(
    preset: str | None = None,
    explicit_extensions: list[str] | None = None,
) -> list[str]:
    """Resolve file extensions. Explicit list overrides preset."""
    if explicit_extensions is not None:
        return sorted(set(explicit_extensions))
    if preset is None:
        return []
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Valid: {list(PRESETS.keys())}")
    if preset == "mixed":
        combined = set(PRESETS["web"]) | set(PRESETS["python"])
        return sorted(combined)
    return sorted(set(PRESETS[preset]))
