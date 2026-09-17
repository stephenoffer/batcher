"""Plan-time constants for the `.str` parameters that select another engine's semantics.

Batcher's string functions follow DuckDB. A few parameters restore the reading Polars,
Spark, Daft or Ray Data give the same function (`contains(literal=False)`,
`trim(whitespace="all")`, `count_matches(literal=True)`), and some of those are exact
compositions of nodes the engine already has, built from a constant computed here rather
than from a new kernel. Everything in this module works on a plan literal, never on a row.
"""

from __future__ import annotations

__all__ = ["UNICODE_WHITE_SPACE", "escape_rust_regex"]

#: Every character with the Unicode `White_Space` property: what Rust's `char::is_whitespace`
#: tests, and so what Polars `strip_chars()` and Daft `strip()` remove with no argument.
#: The engine's argument-less `trim` removes only the space separators (`Zs`), as DuckDB
#: does, so the difference is the C0 controls (tab through carriage return), U+0085 and
#: the two Unicode line and paragraph separators. Passing this set as `chars` is exactly
#: the `White_Space` trim, which is why it needs no kernel of its own.
UNICODE_WHITE_SPACE = "".join(
    chr(code)
    for code in (
        *range(0x09, 0x0E),  # tab, line feed, vertical tab, form feed, carriage return
        0x20,
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),  # en quad through hair space
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
)

# `regex_syntax::is_meta_character`: the characters `regex::escape` backslashes.
_RUST_REGEX_META = frozenset("\\.+*?()|[]{}^$#&-~")


def escape_rust_regex(text: str) -> str:
    """Escape `text` so the engine's regex matches it literally, as `regex::escape` does.

    Python's `re.escape` is not a substitute. It also backslashes whitespace, and whether a
    backslash before a tab or a newline is a valid escape is the Rust crate's decision, not
    Python's; this escapes exactly the metacharacters the crate itself defines.

    Args:
        text: A plan-time literal.

    Returns:
        The literal with each regex metacharacter backslash-escaped.
    """
    return "".join("\\" + c if c in _RUST_REGEX_META else c for c in text)
