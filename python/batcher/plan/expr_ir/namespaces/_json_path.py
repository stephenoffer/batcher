"""Plan-time check of the JSONPath subset the engine's `.json` kernels navigate.

The subset is RFC 9535's *singular query*: a ``$``-rooted chain of member names and
array indices, each selecting at most one value -- ``$.a.b``, ``$.tags[0]``,
``$.a[-1]``, and a quoted name (``$."x.y"``, ``$['x.y']``, ``$["x.y"]``) for a key that
holds a ``.``. Every selector that picks *several* values -- wildcards, recursive
descent, slices, unions, filters -- is refused here with a `PlanError` naming it.

The engine used to skip what it did not understand, so ``$.a[0:1]`` read the whole
array: a plausible wrong answer. The grammar below is the same one
``crates/bc-expr/src/eval/str/json/path.rs`` parses; running it here turns a refused path
into an error at plan build rather than at the first batch. The Rust parser stays as the
defence for a path that only exists per row.
"""

from __future__ import annotations

import re

from batcher._internal.errors import PlanError

__all__ = ["check_json_path", "split_wildcard_tail"]

_RECURSIVE = "recursive descent (`..`) is not supported"
_WILDCARD = (
    "a wildcard (`*`) is not supported; read an array's elements with `.json.values(path)` instead"
)
_INDEX = re.compile(r"[+-]?[0-9]+")


def check_json_path(path: str) -> str:
    """Return `path` unchanged when the engine can navigate it, else raise `PlanError`.

    Args:
        path: The JSONPath a `.json` method or SQL JSON function was given.

    Returns:
        The same path, so a caller can validate inline.

    Raises:
        PlanError: The path uses a selector outside the supported subset, or is malformed.
    """
    if not isinstance(path, str):
        raise PlanError(f"a JSON path must be a string, got {type(path).__name__}")
    reason = _first_problem(path)
    if reason is not None:
        raise PlanError(f"unsupported JSONPath `{path}`: {reason}")
    return path


def split_wildcard_tail(path: str) -> str | None:
    """The path before a trailing ``[*]``, or None when the path does not end in one.

    SQL's ``json_extract(j, '$.a[*]')`` lists every element of ``$.a``; that one shape
    has a kernel of its own, so the SQL front-end peels the wildcard off and validates
    what is left. A wildcard anywhere else is still refused by `check_json_path`.

    Args:
        path: A JSONPath.

    Returns:
        The validated prefix, or None.
    """
    stripped = path.rstrip()
    match = re.search(r"\[\s*\*\s*\]$", stripped)
    if match is None:
        return None
    return check_json_path(stripped[: match.start()] or "$")


def _first_problem(path: str) -> str | None:
    """The reason `path` is refused, or None when every step parses."""
    body = path[1:] if path.startswith("$") else path
    i = 0
    if body and body[0] not in ".[":
        name, i = _read_name(body, 0)
        if (problem := _member(name)) is not None:
            return problem
    while i < len(body):
        ch = body[i]
        if ch == ".":
            i += 1
            if i >= len(body):
                return "a path cannot end with `.`"
            nxt = body[i]
            if nxt == ".":
                return _RECURSIVE
            if nxt == "[":
                return "`.` must be followed by a member name"
            if nxt in "\"'":
                end = _quoted_end(body, i)
                if end is None:
                    return "an unterminated quoted member name"
                i = end
                continue
            name, i = _read_name(body, i)
            if (problem := _member(name)) is not None:
                return problem
        elif ch == "[":
            result = _bracket(body, i)
            if isinstance(result, str):
                return result
            i = result
        else:
            return "expected `.` or `[` between two steps"
    return None


def _member(name: str) -> str | None:
    """Refuse the unquoted names that mean something other than a key."""
    if name == "*":
        return _WILDCARD
    if "]" in name:
        return "a `]` with no matching `[`"
    return None


def _read_name(body: str, start: int) -> tuple[str, int]:
    """The unquoted name at `start`: everything up to the next ``.`` or ``[``."""
    end = start
    while end < len(body) and body[end] not in ".[":
        end += 1
    return body[start:end], end


def _quoted_end(body: str, start: int) -> int | None:
    """Index just past the quoted name opening at `start`; a backslash escapes one char."""
    quote = body[start]
    i = start + 1
    while i < len(body):
        if body[i] == "\\":
            i += 2
            continue
        if body[i] == quote:
            return i + 1
        i += 1
    return None


def _bracket(body: str, start: int) -> int | str:
    """Index just past the ``[...]`` at `start`, or the reason it is refused."""
    i = start + 1
    while i < len(body) and body[i] == " ":
        i += 1
    if i < len(body) and body[i] in "\"'":
        end = _quoted_end(body, i)
        if end is None:
            return "an unterminated quoted member name"
        while end < len(body) and body[end] == " ":
            end += 1
        if end >= len(body) or body[end] != "]":
            return "a quoted name in `[...]` must be followed by `]`"
        return end + 1
    close = body.find("]", i)
    if close == -1:
        return "a `[` with no closing `]`"
    inner = body[i:close].strip()
    if _INDEX.fullmatch(inner):
        return close + 1 if -(2**63) <= int(inner) < 2**63 else "an array index out of range"
    if inner == "*":
        return _WILDCARD
    if inner.startswith("?"):
        return "a filter selector (`[?...]`) is not supported"
    if inner.startswith("#"):
        return "`[#-n]` is DuckDB's last-element syntax; write `[-n]` (`[-1]` is the last element)"
    if ":" in inner:
        return "an array slice (`[start:end]`) is not supported"
    if "," in inner:
        return "a union selector (`[a,b]`) is not supported"
    return "an array subscript must be an integer or a quoted member name"
