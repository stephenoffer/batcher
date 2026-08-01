"""The regular-expression functions, for the patterns three regex engines agree on.

A regex is the one thing in this package that cannot be checked by construction: the engine
compiles Rust's `regex`, the verification backend compiles Python's `re`, and the device
compiles cuDF's own, and the three disagree on exactly the constructs a test over ASCII data
would never reach. That is why the whole family used to be declined — and declining it cost far
more than it looks like, because a *third* of the string surface lowers to one of these four
calls. `word_count`, `punctuation_ratio`, `has_html`, `slugify`, `remove_digits` and thirty
more are all a `regexp_count` or a `regexp_replace_all` underneath, and every one of them sent
its whole chain to the host.

So the pattern is classified rather than the family declined. `portable` accepts only what all
three implement identically and rejects everything else, which keeps the guarantee exact:

* **rejected — the shorthand classes** `\\w`, `\\s`, `\\d`, `\\b` and their negations. Rust's are
  Unicode by default, Python's are Unicode on `str`, and cuDF's are not, so they agree on ASCII
  and diverge on the first accented letter. That is the worst possible failure shape: correct in
  every test and wrong on real text;
* **rejected — `$`**. Python's matches before a trailing newline and Rust's does not, so
  `^[0-9]+$` disagrees about `"12\\n"`;
* **rejected — lookaround, backreferences, inline flags, POSIX and Unicode classes**, none of
  which all three even implement;
* **accepted** — literals, escaped metacharacters, `\\xNN`, explicit character classes and
  ranges, `.`, the quantifiers, alternation, and non-capturing groups. That is `[0-9]`,
  `[A-Za-z]`, `[^\\x00-\\x7F]`, `<[^>]+>` and `[^a-z0-9]+`: the patterns the engine's own text
  functions are built from.

A rejected pattern is an `Unsupported`, so the chain runs on the CPU engine and returns the
same rows. Nothing here approximates a pattern it does not fully understand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.core.gpu_plan.backend import Unsupported, call_or_decline

if TYPE_CHECKING:
    from batcher.core.gpu_plan.backend import DfBackend

__all__ = ["REGEX_FNS", "eval_regex", "portable"]

#: The regex functions translated here. Three are absent and stay that way. `regexp_extract_all`
#: and `regexp_split` return a *list* column, and the only spelling pandas has for either
#: materializes a Python list per row, which is a hot-path tuple touch rather than a translation.
#: `regexp_extract` is absent for a different reason: pandas' Arrow-backed `extract` is
#: `pyarrow.compute.extract_regex`, which accepts only **named** capture groups, and cuDF's
#: accepts only unnamed ones — so the verification backend cannot run the pattern the device
#: would, which is the one thing this package refuses to ship.
REGEX_FNS = frozenset({"regexp_count", "regexp_matches", "regexp_replace", "regexp_replace_all"})

#: Characters that are a literal when escaped, and mean the same literal in all three engines.
_LITERAL_ESCAPES = frozenset(".^$*+?()[]{}|\\/-tnrfv \"'#&~=<>,:;!@%_`")

#: Hexadecimal digits, for the `\xNN` escape — the one non-literal escape that is portable,
#: and the one `[^\x00-\x7F]` (the non-ASCII test) is built from.
_HEX = frozenset("0123456789abcdefABCDEF")


def portable(pattern: str) -> bool:
    """Whether every construct in `pattern` means the same thing in all three regex engines.

    Conservative by construction: the scanner accepts a known list and rejects everything else,
    so a construct nobody thought about is declined rather than assumed. It also over-rejects in
    one harmless place — a `$` inside a character class is a literal and is rejected anyway —
    because the cost of that is a fallback and the cost of the opposite is a wrong answer.

    Args:
        pattern: The regular expression from the plan.

    Returns:
        True when the pattern may be handed to either dataframe backend.
    """
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "\\":
            if i + 1 >= n:
                return False
            nxt = pattern[i + 1]
            if nxt == "x" and i + 3 < n and pattern[i + 2] in _HEX and pattern[i + 3] in _HEX:
                i += 4
                continue
            if nxt not in _LITERAL_ESCAPES:
                # `\w`, `\s`, `\d`, `\b`, `\p{...}`, `\1` — every one of them dialect-sensitive.
                return False
            i += 2
            continue
        if char == "$":
            return False
        if pattern.startswith("[:", i):
            # A POSIX class (`[[:alpha:]]`). Rust implements it, Python does not, and cuDF has
            # its own list — three different answers to the same bracket.
            return False
        if char == "(" and pattern.startswith("(?", i) and not pattern.startswith("(?:", i):
            # A lookaround or an inline flag group. cuDF implements neither.
            return False
        i += 1
    return True


def _checked(pattern: str) -> str:
    if not portable(pattern):
        raise Unsupported(f"regex pattern {pattern!r} is not portable across the backends")
    return pattern


def _checked_replacement(replacement: str) -> str:
    """A replacement string carrying no group reference, which the three engines spell three
    ways (`$1`, `\\1`, and a dedicated call on the device)."""
    if "\\" in replacement or "$" in replacement:
        raise Unsupported("regex replacement with a group reference")
    return replacement


def eval_regex(fn: str, x, ir: dict, be: DfBackend):
    """Evaluate one regular-expression function over the string column `x`.

    Args:
        fn: The `str` node's ``fn`` discriminator.
        x: The string column.
        ir: The node's JSON IR, carrying the pattern and any replacement.
        be: The dataframe backend to compute on.

    Returns:
        The result as a column of `x`'s length.

    Raises:
        Unsupported: For a function outside `REGEX_FNS`, or a pattern the three engines do not
            agree on.
    """
    import pyarrow as pa

    pattern = _checked(str(ir["pattern"]))
    if fn == "regexp_count":
        # Both libraries spell a regex count `.str.count`, and both count non-overlapping
        # matches, which is what the engine reports.
        return call_or_decline(x.str, "count", pattern).astype(be.dtype(pa.int64()))
    if fn == "regexp_matches":
        # A *search*, not a full match: the engine's `regexp_matches` is true when the pattern
        # occurs anywhere. Spelled with `regex=True` explicitly, because the same call means a
        # literal substring elsewhere in this package and the default differs between them.
        return call_or_decline(x.str, "contains", pattern, regex=True)
    if fn in ("regexp_replace", "regexp_replace_all"):
        replacement = _checked_replacement(str(ir["replacement"]))
        # `n=1` is "replace the first occurrence", which is what `regexp_replace` does; the
        # `_all` form replaces every one. Getting the two the wrong way round changes a value
        # rather than raising.
        count = 1 if fn == "regexp_replace" else -1
        return call_or_decline(x.str, "replace", pattern, replacement, n=count, regex=True)
    raise Unsupported(f"regex fn {fn}")
