"""Dotted table and namespace identifiers, and the glob a listing is filtered by.

Every catalog name is a dotted string: ``"t"``, ``"sales.t"``, ``"lake.sales.t"``. There is
no identifier object, because a string is what every engine a user arrives from accepts and
what a SQL statement already holds. The one thing a string cannot say is a dot *inside* a
name, and that is refused here rather than guessed at.
"""

from __future__ import annotations

import fnmatch

from batcher._internal.errors import PlanError

__all__ = ["filter_names", "join", "split"]


def split(name: str, *, what: str = "table") -> tuple[str, ...]:
    """Split a dotted identifier into its parts, refusing an empty part.

    Args:
        name: The identifier as the caller wrote it.
        what: What the identifier names, for the error message.

    Returns:
        The non-empty parts, in order.

    Raises:
        PlanError: `name` is not a string, or has an empty part (``"a..b"``, ``""``).
    """
    if not isinstance(name, str):
        raise PlanError(f"a {what} name must be a string, got {type(name).__name__}")
    parts = tuple(name.split("."))
    if not all(parts):
        raise PlanError(
            f"invalid {what} name {name!r}: every dot-separated part must be non-empty",
            hint="Write names such as 't', 'namespace.t' or 'catalog.namespace.t'.",
        )
    return parts


def join(*parts: str) -> str:
    """Join identifier parts back into a dotted name.

    Args:
        parts: The parts, outermost first.

    Returns:
        The dotted name.
    """
    return ".".join(parts)


def filter_names(names: list[str], pattern: str | None) -> list[str]:
    """Sort `names` and keep those matching the glob `pattern` (all of them when None).

    A glob rather than a regex because it is what Spark's ``listTables(pattern)`` and Daft's
    ``list_tables(pattern)`` users type: ``"sales.*"``, ``"*events*"``.

    Args:
        names: The candidate names.
        pattern: A shell-style glob, or None for no filter.

    Returns:
        The matching names, sorted.
    """
    kept = names if pattern is None else [n for n in names if fnmatch.fnmatchcase(n, pattern)]
    return sorted(kept)
