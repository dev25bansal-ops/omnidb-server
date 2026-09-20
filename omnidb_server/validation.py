"""AUDIT-I21: shared write-path validation for every mutation entry point.

Creation, update and bulk import must apply the SAME documented content and
trigger-pattern policy before any database or index mutation, so operators
can rely on the configured size bounds. This module is dependency-free so
both store.py and portability.py can share one contract; callers translate
the raised TypeError/ValueError onto the OMN-* error taxonomy.
"""
from __future__ import annotations

# AUDIT-I06: admission bounds for procedural trigger patterns. The matcher
# side (recall.py) additionally bounds per-recall evaluation work; these
# caps bound what a home can ever store.
MAX_TRIGGER_PATTERNS = 32
MAX_TRIGGER_PATTERN_CHARS = 512


def validate_content(content: object, max_chars: int) -> str:
    """Return the content unchanged when it satisfies the type and size
    policy; raise TypeError/ValueError otherwise."""
    if not isinstance(content, str):
        raise TypeError(f"content must be a string, got {type(content).__name__}")
    if len(content) > max_chars:
        raise ValueError(
            f"content exceeds max_content_chars ({len(content)} > {max_chars} chars)"
        )
    return content


def validate_trigger_patterns(patterns: object) -> list[str]:
    """Return the pattern list when it satisfies the admission policy;
    raise TypeError/ValueError otherwise."""
    if not isinstance(patterns, list):
        raise TypeError("trigger_patterns must be a list of strings")
    if len(patterns) > MAX_TRIGGER_PATTERNS:
        raise ValueError(
            f"trigger_patterns exceeds {MAX_TRIGGER_PATTERNS} entries "
            f"(got {len(patterns)})"
        )
    out: list[str] = []
    for pat in patterns:
        if not isinstance(pat, str):
            raise TypeError("each trigger pattern must be a string")
        if len(pat) > MAX_TRIGGER_PATTERN_CHARS:
            raise ValueError(
                f"trigger pattern exceeds {MAX_TRIGGER_PATTERN_CHARS} chars "
                f"(got {len(pat)})"
            )
        out.append(pat)
    return out
