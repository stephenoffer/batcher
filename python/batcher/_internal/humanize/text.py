"""Display-width-correct text measurement, truncation, padding, and terminal hyperlinks.

Every aligned thing Batcher prints — the progress line's fixed columns, the profile's
operator table, the log record's level field — is padded to a width. Padding with
`len()` is right only for the subset of strings that are pure ASCII and carry no escape
codes, and Batcher's are routinely neither:

* A **color escape** costs zero columns and several characters. `str.ljust` on a colored
  cell pads it by the length of the escape sequence, so a colored table is a table with
  ragged columns, and it is ragged only when color is on — which is to say, only for the
  human and never for the test.
* A **CJK or emoji character** costs two columns and one character. A dataset named
  ``注文明細`` clipped to an 18-column field overflows to 22 and shifts every column to
  its right. This is not exotic: it is what happens the first time the engine is used in
  Tokyo or Shanghai.
* A **combining mark** costs zero columns and one character.

So width is measured here, once, from `unicodedata` — no `wcwidth` dependency, because a
data engine should not make a user install a rendering library to see a progress bar, and
the stdlib carries the East Asian Width property this needs.

Layer 0 (`_internal`) so `observe`, `plan`, `io`, and `api` can all align against the same
measurement.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ELLIPSIS",
    "display_width",
    "fit",
    "hyperlink",
    "indent_block",
    "pad",
    "strip_ansi",
    "truncate",
    "wrap",
]

#: What a clipped string ends with. One column wide, and universally understood.
ELLIPSIS = "…"

# CSI/OSC escape sequences: the two families a terminal renderer actually emits. Matching
# both matters because an OSC-8 hyperlink wraps its label in sequences terminated by BEL or
# ST, which the CSI pattern alone would leave in the measured width.
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")

# Characters that occupy two terminal columns: the East Asian Wide and Fullwidth classes,
# plus the emoji ranges that terminals render double-wide but that `east_asian_width`
# reports as Neutral on older Unicode data files.
_WIDE_EAW = frozenset("WF")
_WIDE_RANGES = (
    (0x1F300, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x27BF),
)


def strip_ansi(text: str) -> str:
    """`text` with every ANSI escape sequence removed.

    Args:
        text: Possibly styled text.

    Returns:
        The same text with no escape sequences.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import strip_ansi
            >>> strip_ansi("\\x1b[1mbold\\x1b[0m")
            'bold'
    """
    return _ANSI.sub("", text) if "\x1b" in text else text


def _char_width(ch: str) -> int:
    """Terminal columns one character occupies: 0 for combining, 2 for wide, else 1."""
    if unicodedata.combining(ch):
        return 0
    category = unicodedata.category(ch)
    if category in ("Mn", "Me", "Cf"):
        return 0
    if category == "Cc":
        return 0
    if unicodedata.east_asian_width(ch) in _WIDE_EAW:
        return 2
    code = ord(ch)
    return 2 if any(lo <= code <= hi for lo, hi in _WIDE_RANGES) else 1


def display_width(text: str) -> int:
    """The number of terminal columns `text` occupies, ignoring escape sequences.

    Args:
        text: The text to measure, styled or not.

    Returns:
        Column count.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import display_width
            >>> display_width("abc"), display_width("\\u6ce8\\u6587")
            (3, 4)
            >>> display_width("\\x1b[1mx\\x1b[0m")
            1
    """
    plain = strip_ansi(text)
    if plain.isascii():
        return len(plain)
    return sum(_char_width(ch) for ch in plain)


def truncate(text: str, width: int, *, ellipsis: str = ELLIPSIS) -> str:
    """`text` clipped to at most `width` display columns, ending in an ellipsis if clipped.

    Never splits a double-width character across the boundary: if the last character that
    would fit is wide and only one column remains, it is dropped and the cell is padded
    instead, so the column after it still starts where it should.

    Args:
        text: The text to clip. Must not contain escape sequences — clip before styling.
        width: The maximum display width; ``<= 0`` yields ``""``.
        ellipsis: The marker appended when clipping happens.

    Returns:
        Text of at most `width` display columns.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import truncate
            >>> truncate("hash_join_probe", 9)
            'hash_joi\\u2026'
    """
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    marker_width = display_width(ellipsis)
    if width <= marker_width:
        return ellipsis[:width] if marker_width <= width else ""
    budget = width - marker_width
    out: list[str] = []
    used = 0
    for ch in text:
        step = _char_width(ch)
        if used + step > budget:
            break
        out.append(ch)
        used += step
    return "".join(out) + " " * (budget - used) + ellipsis


def pad(text: str, width: int, *, align: str = "left") -> str:
    """`text` padded with spaces to exactly `width` display columns, clipping if longer.

    The escape-aware replacement for `str.ljust`/`rjust`/`center`. Styled text pads
    correctly because only the visible columns are counted.

    Args:
        text: The text to place in the field.
        width: The field width in display columns.
        align: ``"left"``, ``"right"``, or ``"center"``.

    Returns:
        Text occupying exactly `width` columns.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import pad
            >>> pad("ab", 5) + "|"
            'ab   |'
            >>> pad("ab", 5, align="right") + "|"
            '   ab|'
    """
    current = display_width(text)
    if current > width:
        # Clipping a styled string would cut an escape in half; clip the plain form and let
        # the caller restyle. This is why every call site styles *after* fitting.
        text = truncate(strip_ansi(text), width)
        current = display_width(text)
    slack = width - current
    if align == "right":
        return " " * slack + text
    if align == "center":
        left = slack // 2
        return " " * left + text + " " * (slack - left)
    return text + " " * slack


def fit(text: str, width: int, *, align: str = "left") -> str:
    """Clip `text` to `width` and pad it back out — one call for a fixed-width cell.

    Args:
        text: The cell's content.
        width: The column width.
        align: ``"left"``, ``"right"``, or ``"center"``.

    Returns:
        Text occupying exactly `width` display columns.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import fit
            >>> fit("aggregate", 6)
            'aggre\\u2026'
    """
    return pad(truncate(text, width), width, align=align)


def hyperlink(label: str, url: str, *, enabled: bool = True) -> str:
    """`label` as an OSC-8 terminal hyperlink to `url`, or plain when not `enabled`.

    Modern terminals (iTerm2, WezTerm, Kitty, VTE, Windows Terminal) render this as a
    clickable link; every other terminal shows the label unchanged, because the escape is
    ignored rather than printed. That degradation is why the engine can put a documentation
    link on an error without a fallback branch — the worst case is the label alone.

    Args:
        label: The visible text.
        url: The target URL.
        enabled: `False` returns the label unchanged, for a non-TTY or a captured stream.

    Returns:
        The label, hyperlinked when enabled.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import hyperlink
            >>> hyperlink("docs", "https://example.invalid", enabled=False)
            'docs'
    """
    if not enabled or not url:
        return label
    return f"\x1b]8;;{url}\x1b\\{label}\x1b]8;;\x1b\\"


def wrap(text: str, width: int) -> list[str]:
    """`text` broken into lines of at most `width` display columns, on word boundaries.

    `textwrap` measures in characters, which is the same mistake `str.ljust` makes; this
    measures in columns, so a wrapped Japanese sentence stays inside the terminal.

    Args:
        text: The prose to wrap.
        width: The maximum display width per line.

    Returns:
        The wrapped lines; a single empty line for empty input.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import wrap
            >>> wrap("the quick brown fox", 9)
            ['the quick', 'brown fox']
    """
    if width <= 0:
        return [text]
    lines: list[str] = []
    current: list[str] = []
    used = 0
    for word in text.split():
        step = display_width(word)
        if current and used + 1 + step > width:
            lines.append(" ".join(current))
            current, used = [word], step
        else:
            used = step if not current else used + 1 + step
            current.append(word)
    lines.append(" ".join(current))
    return lines


def indent_block(text: str, prefix: str) -> str:
    """Every line of `text` prefixed with `prefix` — for nesting a block under a heading.

    Args:
        text: A possibly multi-line block.
        prefix: The prefix to apply to each line.

    Returns:
        The indented block.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import indent_block
            >>> indent_block("a\\nb", "  ")
            '  a\\n  b'
    """
    return "\n".join(prefix + line for line in text.split("\n"))
