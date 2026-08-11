"""Text constraints: patterns, lengths, blankness, and the well-known formats.

The named formats (`email`, `url`, `uuid`, `ipv4`) are ordinary regex constraints with the
pattern supplied. They exist because the pattern is the part people get wrong — a hand-written
email regex that rejects `+` tagging, a UUID one that forgets the hyphens — and because a
report reading ``is_email(contact)`` says what failed where ``matches(contact, '^[^@...')``
says only that something did.
"""

from __future__ import annotations

import re

from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import RowConstraint
from batcher.plan.expr_ir import Col

__all__ = [
    "FORMATS",
    "compile_pattern",
    "matches",
    "matches_format",
    "not_empty",
    "not_matches",
    "str_length_between",
]

FORMATS: dict[str, str] = {
    # Deliberately permissive where the standard is: an address may carry `+` tags and a
    # long TLD, and the only universal rules are "exactly one @, no whitespace, a dotted
    # domain". A stricter pattern rejects valid mail, which is the more expensive mistake.
    "email": r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$",
    "url": r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+$",
    "uuid": r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    "ipv4": (
        r"^((25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
        r"(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$"
    ),
}
"""The named text formats `matches_format` understands, as regular expressions."""


def compile_pattern(check: str, column: str, pattern: str) -> None:
    """Reject a malformed regex here, where the check and the column are still known.

    The engine otherwise reported a bare ``invalid regular expression: [`` from the Rust
    matcher, with no indication of which of a pipeline's checks carried it. Python's regex
    dialect is not the engine's, so this catches the malformed patterns rather than every
    pattern the engine would reject; a dialect difference still surfaces there.

    Args:
        check: The name of the calling check, for the message.
        column: The column the pattern was written against.
        pattern: The regular expression to validate.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise PlanError(
            f"{check}({column!r}, {pattern!r}) is not a valid regular expression: {exc}."
        ) from exc


def matches(column: str, pattern: str) -> RowConstraint:
    """`column` must match the regex `pattern` (NULL passes).

    Args:
        column: The column to test.
        pattern: The regular expression each value must match.

    Returns:
        The row constraint.
    """
    compile_pattern("matches", column, pattern)
    c = Col(column)
    return RowConstraint(
        f"matches({column}, {pattern!r})", c.is_null() | c.str.regexp_matches(pattern)
    )


def not_matches(column: str, pattern: str) -> RowConstraint:
    """`column` must **not** match the regex `pattern` (NULL passes).

    The shape a PII or placeholder scan takes: a column is fine unless it looks like a
    credit-card number, or unless it still carries the `TODO` an upstream job left behind.

    Args:
        column: The column to test.
        pattern: The regular expression no value may match.

    Returns:
        The row constraint.
    """
    compile_pattern("not_matches", column, pattern)
    c = Col(column)
    return RowConstraint(
        f"not_matches({column}, {pattern!r})", c.is_null() | ~c.str.regexp_matches(pattern)
    )


def matches_format(column: str, fmt: str) -> RowConstraint:
    """`column` must match a well-known text format (NULL passes).

    Args:
        column: The column to test.
        fmt: One of the keys of `FORMATS`.

    Returns:
        The row constraint.
    """
    pattern = FORMATS.get(fmt)
    if pattern is None:
        known = ", ".join(sorted(FORMATS))
        raise PlanError(
            f"matches_format({column!r}, {fmt!r}): unknown format. Known formats: {known}. "
            "Use matches() with your own regular expression for anything else."
        )
    c = Col(column)
    return RowConstraint(f"is_{fmt}({column})", c.is_null() | c.str.regexp_matches(pattern))


def str_length_between(column: str, low: int, high: int | None = None) -> RowConstraint:
    """`column`'s character length must lie in ``[low, high]`` (NULL passes).

    Length is counted in **characters**, not bytes, so a name in a non-Latin script is not
    rejected for being three bytes per character.

    Args:
        column: The text column to test.
        low: Inclusive minimum length.
        high: Inclusive maximum length, or `None` for no upper bound.

    Returns:
        The row constraint.
    """
    if low < 0:
        raise PlanError(f"str_length_between({column!r}): low must be >= 0, got {low!r}.")
    if high is not None and high < low:
        raise PlanError(
            f"str_length_between({column!r}): low ({low}) > high ({high}) — swap the arguments?"
        )
    c = Col(column)
    n = c.str.len_chars()
    test = n >= low if high is None else n.between(low, high)
    bound = f"{low}, {high}" if high is not None else f"{low}, None"
    return RowConstraint(f"str_length_between({column}, {bound})", c.is_null() | test)


def not_empty(column: str, *, strip: bool = True) -> RowConstraint:
    """`column` must not be the empty string (NULL passes).

    An empty string is the null that survives a CSV round trip, and every value constraint
    written against the column passes it: it is in no forbidden set, matches no rejected
    pattern, and `not_null` says it is present. With `strip`, a value that is only
    whitespace counts as empty too, which is the form the same failure takes after a
    fixed-width export.

    Args:
        column: The text column to test.
        strip: Whether surrounding whitespace is ignored before measuring.

    Returns:
        The row constraint.
    """
    c = Col(column)
    value = c.str.strip_chars() if strip else c
    return RowConstraint(f"not_empty({column})", c.is_null() | (value.str.len_chars() > 0))
