"""Plan-time constants for the `.str` parameters that select another engine's semantics.

Batcher's string functions follow DuckDB. A few parameters restore the reading Polars,
Spark, Daft or Ray Data give the same function (`contains(literal=False)`,
`trim(whitespace="all")`, `count_matches(literal=True)`), and some of those are exact
compositions of nodes the engine already has, built from a constant computed here rather
than from a new kernel. Everything in this module works on a plan literal, never on a row.
"""

from __future__ import annotations

from collections.abc import Iterator

from batcher._internal.errors import PlanError

__all__ = ["UNICODE_WHITE_SPACE", "check_regex", "escape_rust_regex", "regex_group_names"]

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


# --- the engine's regex dialect -------------------------------------------------------
#
# Patterns run on the Rust `regex` crate, which guarantees linear-time matching and so has
# no construct that needs backtracking. Python's `re` and the Rust crate disagree on syntax,
# so a pattern is never compiled here; it is *scanned*, just far enough to find the
# constructs the engine will refuse and the capture groups it will number. The scan tracks
# escapes and (nested) character classes, since a `(` in either is a literal.

#: Constructs the engine's regex refuses, by the text that opens them, with what to do
#: instead. Longest first, so `(?<=` is not mistaken for a named group `(?<name>`.
_UNSUPPORTED: tuple[tuple[str, str, str], ...] = (
    ("(?<=", "lookbehind", "capture the context with a group and take what follows it"),
    ("(?<!", "negative lookbehind", "test the preceding text with a second regexp_matches"),
    ("(?=", "lookahead", "capture the context with a group and take what precedes it"),
    ("(?!", "negative lookahead", "combine patterns: regexp_matches(a) & ~regexp_matches(b)"),
    ("(?>", "an atomic group", "use (?:...); the engine never backtracks, so it is the same"),
)

#: A numbered (`\1` to `\9`) or named (`\k<name>`) backreference.
_BACKREF_TOKENS = frozenset({*(f"\\{d}" for d in "123456789"), "\\k"})

_BACKREF_ADVICE = "capture both parts with str.extract_groups and compare the fields"


def _scan(pattern: str) -> Iterator[tuple[int, str]]:
    """Yield ``(index, token)`` for each group opener and escape outside a character class.

    A token is ``"("`` for an opening parenthesis or the two characters of an escape
    (``"\\1"``). Everything inside ``[...]``, nested classes included, is skipped.
    """
    i, depth, n = 0, 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\":
            if depth == 0:
                yield i, pattern[i : i + 2]
            i += 2
            continue
        if c == "[":
            # `]` straight after `[` or `[^` is a literal, not the end of the class.
            depth += 1
            i += 1
            if i < n and pattern[i] == "^":
                i += 1
            if i < n and pattern[i] == "]":
                i += 1
            continue
        if c == "]" and depth:
            depth -= 1
        elif c == "(" and depth == 0:
            yield i, "("
        i += 1


def check_regex(pattern: str, method: str) -> str:
    """Refuse a construct the engine's regex cannot run, before any row is scanned.

    The engine's regex is the Rust `regex` crate (RE2's syntax family): linear time, and
    so no lookaround, no backreferences and no atomic groups. Such a pattern used to build
    a plan that failed only once the scan started, with *invalid regular expression*.

    Args:
        pattern: The plan-time pattern.
        method: The ``.str`` method name, for the message.

    Returns:
        `pattern`, unchanged.

    Raises:
        PlanError: Naming the construct and what to write instead.
    """
    for i, token in _scan(pattern):
        if token == "(":
            for opener, name, advice in _UNSUPPORTED:
                if pattern.startswith(opener, i):
                    raise PlanError(_unsupported(method, name, pattern, i, advice))
        elif token in _BACKREF_TOKENS:
            raise PlanError(_unsupported(method, "a backreference", pattern, i, _BACKREF_ADVICE))
    return pattern


def _unsupported(method: str, construct: str, pattern: str, at: int, advice: str) -> str:
    return (
        f"str.{method}(): the regex {pattern!r} uses {construct} at position {at}, which the "
        "engine's regex does not support (it is the Rust `regex` crate, RE2 syntax: "
        f"linear-time, with no lookaround or backreferences). Instead, {advice}."
    )


def regex_group_names(pattern: str) -> list[str]:
    """The field name of each capture group in `pattern`, in group order.

    A named group, ``(?P<name>...)`` or ``(?<name>...)``, keeps its name; an unnamed one is
    called by its 1-based group number, as `bc-expr`'s `groups::group_names` names it from
    the compiled regex. Non-capturing groups and flag groups are skipped.

    Args:
        pattern: The plan-time pattern.

    Returns:
        One name per capturing group; empty when there are none.
    """
    names: list[str] = []
    for i, token in _scan(pattern):
        if token != "(":
            continue
        rest = pattern[i + 1 :]
        for prefix in ("?P<", "?<"):
            if rest.startswith(prefix) and not rest.startswith(("?<=", "?<!")):
                names.append(rest[len(prefix) : rest.index(">")])
                break
        else:
            if not rest.startswith("?"):
                names.append(str(len(names) + 1))
    return names
