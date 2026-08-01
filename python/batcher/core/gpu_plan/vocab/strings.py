"""The string function family — one entry per named function the engine ships.

Split from `scalar_fns`, which is math and dates, on the seam that matters for reading it:
almost every entry here exists because the obvious translation is *wrong* in a way that returns
plausible text rather than an error. `substr` is 1-based, `lpad` truncates as well as pads,
`position` reports `0` rather than `-1` for "not found", `contains` matches a literal where both
libraries default to a regular expression, and the strip functions take a set of characters
rather than whitespace.

Each is spelled out and pinned by a case comparing it against the engine.
"""

from __future__ import annotations

from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.vocab.regex import REGEX_FNS, eval_regex

__all__ = ["eval_str"]


def _int64():
    import pyarrow as pa

    return pa.int64()


# String functions that are a no-argument `.str` method, named where the two differ. Spelled
# out rather than passed through by name: a name that happens to exist on a Series is not the
# same as one that means what the engine means (see the `trunc` case in `_MATH_FNS`), and a
# name that exists on cuDF but not pandas is a path nothing verifies.
_STR_METHODS = {"lower": "lower", "upper": "upper", "initcap": "title"}

# String functions taking a single `pattern` argument, mapped to their `.str` method. Both of
# these match literally on both libraries, which is what the engine does. `contains` is NOT here
# because it does not: it defaults to a regular expression and is handled explicitly.
_STR_PATTERN_METHODS = {
    "starts_with": "startswith",
    "ends_with": "endswith",
}


def eval_str(ir, df, be, eval_expr):
    x = be.column(eval_expr(ir["input"], df, be), df)
    fn = ir["fn"]
    if fn in _STR_METHODS:
        return getattr(x.str, _STR_METHODS[fn])()
    if fn in ("lpad", "rpad"):
        return _pad(x, int(ir["start"]), str(ir["pattern"]), left=fn == "lpad")
    if fn == "repeat":
        return x.str.repeat(int(ir["start"]))
    if fn == "right":
        return x.str.slice(-int(ir["start"]))
    if fn == "position":
        # SQL `position` is 1-based and reports 0 for "not found"; `find` is 0-based and -1.
        return (x.str.find(ir["pattern"]) + 1).astype(be.dtype(_int64()))
    if fn == "len":
        return x.str.len().astype(be.dtype(_int64()))
    if fn == "like":
        return _like(x, str(ir["pattern"]))
    if fn == "reverse":
        # A negative-step slice, which is the one spelling of "reversed" both libraries have —
        # neither exposes a `.str.reverse` that the other also does. It steps over characters,
        # not bytes, which is what the engine reverses too.
        return x.str.slice(step=-1)
    if fn == "contains":
        # The engine matches a **literal** substring, exactly as `replace` does. Both
        # libraries' `contains` defaults to a regular expression instead, so any pattern
        # carrying a metacharacter matches rows the engine does not — and `.` is in every path,
        # hostname, version string and email domain anyone filters on. `contains("a.b")` was
        # matching "axb"; `contains("a|b")` was matching everything.
        return x.str.contains(ir["pattern"], regex=False)
    if fn in _STR_PATTERN_METHODS:
        return getattr(x.str, _STR_PATTERN_METHODS[fn])(ir["pattern"])
    if fn == "replace":
        # The engine replaces every occurrence and treats the pattern as a literal.
        return x.str.replace(ir["pattern"], ir["replacement"], regex=False)
    if fn == "substr":
        # SQL's 1-based, inclusive `substring(s, start, length)`: a `start` below 1 spends
        # part of the length before the string begins, so `substr(s, 0, 1)` is the empty
        # string. Slicing from a 0-based `start` instead silently returns a shifted window.
        #
        # `length` is *optional* — `substr(s, 2)` runs to the end of the string, and that is
        # the form `capitalize` lowers to. Reading it unconditionally raised `KeyError`, which
        # is not an `Unsupported` and so reached the caller as a backend defect rather than as
        # a decline: every `.str.capitalize()` reported the GPU backend as broken.
        start = int(ir["start"])
        begin = max(start - 1, 0)
        if ir.get("length") is None:
            return x.str.slice(begin)
        return x.str.slice(begin, max(start + int(ir["length"]) - 1, 0))
    if fn in REGEX_FNS:
        return eval_regex(fn, x, ir, be)
    if fn in ("trim", "l_trim", "r_trim"):
        # The `pattern` is the *set of characters* to strip, and dropping it silently stripped
        # whitespace instead: `strip_chars("ax")` returned the string untouched wherever it had
        # no leading space. Absent, the pattern means whitespace, which is both libraries'
        # default — so it is passed through rather than defaulted here.
        method = {"trim": x.str.strip, "l_trim": x.str.lstrip, "r_trim": x.str.rstrip}[fn]
        chars = ir.get("pattern")
        return method() if chars is None else method(str(chars))
    raise Unsupported(f"str fn {fn}")


def _like(x, pattern: str):
    """SQL `LIKE`, for the patterns that reduce to a literal substring test.

    The engine classifies a `LIKE` pattern exactly this way and only reaches for a regex when
    it has to (`eval::str::like::LikeMatcher::classify`), so this is the same reduction rather
    than an approximation of it — `%foo%` is a substring scan on both sides, not two regex
    dialects that happen to agree.

    That matters because a regex is the one thing here that could not be checked: the engine
    compiles Rust's, the host backend Python's and the device cuDF's, and the three disagree on
    exactly the classes a test over ASCII data would never reach. `ILIKE` and any pattern
    carrying `_` are the cases the engine itself needs a regex for, and they are declined here
    for the same reason. A pattern with a literal in the middle (`a%b`) is declined too: the
    engine scans its segments in order, which no single `.str` call expresses.
    """
    if "_" in pattern:
        raise Unsupported("like with a single-character wildcard")
    parts = pattern.split("%")
    if len(parts) == 1:
        return x == pattern  # no wildcard at all: LIKE is equality
    prefix, suffix = parts[0], parts[-1]
    middles = [p for p in parts[1:-1] if p]  # `%%` constrains nothing
    if prefix and not suffix and not middles:
        return x.str.startswith(prefix)
    if suffix and not prefix and not middles:
        return x.str.endswith(suffix)
    if not prefix and not suffix and len(middles) == 1:
        # Literal, not a regular expression — the same reason `contains` is spelled this way.
        return x.str.contains(middles[0], regex=False)
    if not prefix and not suffix and not middles:
        # A pattern of nothing but `%` matches every row it is given, and a null is still not
        # a row it was given: `LIKE` on an unknown is unknown.
        return x.notna().where(x.notna(), None)
    raise Unsupported(f"like pattern {pattern!r}")


def _pad(x, width: int, fill: str, *, left: bool):
    """SQL `lpad`/`rpad`: pad to `width`, and **truncate** to it when the value is longer.

    The libraries' `rjust`/`ljust` only ever pad, so a value longer than the width came back
    unchanged where the engine returns its first `width` characters.
    """
    padded = x.str.pad(width, side="left" if left else "right", fillchar=fill)
    return padded.str.slice(0, width)
