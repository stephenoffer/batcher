"""Script and character-set composition metrics — corpus-level detectors of language drift.

When a model is meant to answer in one language, its output can silently drift: a Han
character slips into an English answer, a whole reply comes back in Cyrillic, or a burst of
emoji fills the text. Each metric here is a single mergeable aggregate over one output column
that turns that failure into one dashboard number, so a check over a million generations is
one scan and composes inside `group_by` to break the rate down per model, per prompt, or per
day.

Each rate is the fraction of outputs that contain (or, for `latin_only_rate`, avoid) a given
script or character class, detected with the Rust `regex` engine's Unicode script classes
(`\\p{Han}`, `\\p{Cyrillic}`, `\\p{Arabic}`) and codepoint ranges for emoji. A match anywhere
in the output flags it, so a single stray character is enough to catch drift.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "arabic_rate",
    "cjk_rate",
    "cyrillic_rate",
    "emoji_rate",
    "latin_only_rate",
]

# Emoji and pictographic symbol codepoint ranges: the Supplementary Symbols and Pictographs
# block plus the Miscellaneous Symbols and Dingbats range that covers older emoji like "✅".
_EMOJI_RE = r"[\x{1f300}-\x{1faff}\x{2600}-\x{27bf}]"

# Any non-ASCII character: the presence of one means the output is not clean ASCII/Latin text.
_NON_ASCII_RE = r"[^\x00-\x7f]"


def _rate(cond: Expr) -> Expr:
    """The fraction of outputs where a boolean condition holds, as a mergeable aggregate."""
    return count_if(cond) / count_if(lit(True))


def cjk_rate(text: IntoExpr) -> Expr:
    """The fraction of outputs containing at least one CJK (Han) character — a drift detector.

    Flags any output with a Chinese, Japanese kanji, or Korean hanja character, then reports the
    share of flagged outputs over the corpus. It is the cheapest catch for a model that slips
    into or toward CJK when it should answer in a Latin-script language. A single Han character
    is enough to flag an output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The CJK-character rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["hello 世界", "plain english", "привет"]})
            >>> ds.agg(m=bt.cjk_rate("o")).to_pydict()
            {'m': [0.3333333333333333]}
    """
    return _rate(_as_column(text).str.regexp_matches(r"\p{Han}"))


def cyrillic_rate(text: IntoExpr) -> Expr:
    """The fraction of outputs containing at least one Cyrillic character — a drift detector.

    Flags any output with a Cyrillic character, then reports the share of flagged outputs over
    the corpus. It is the direct catch for a model that drifts into Russian, Ukrainian, or
    another Cyrillic-script language when it should answer in Latin script. A single Cyrillic
    character is enough to flag an output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The Cyrillic-character rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["hello 世界", "plain english", "привет"]})
            >>> ds.agg(m=bt.cyrillic_rate("o")).to_pydict()
            {'m': [0.3333333333333333]}
    """
    return _rate(_as_column(text).str.regexp_matches(r"\p{Cyrillic}"))


def arabic_rate(text: IntoExpr) -> Expr:
    """The fraction of outputs containing at least one Arabic character — a drift detector.

    Flags any output with an Arabic-script character, then reports the share of flagged outputs
    over the corpus. It is the direct catch for a model that drifts into Arabic, Persian, or
    another Arabic-script language when it should answer in Latin script. A single Arabic
    character is enough to flag an output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The Arabic-character rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["hello 世界", "plain english", "привет"]})
            >>> ds.agg(m=bt.arabic_rate("o")).to_pydict()
            {'m': [0.0]}
    """
    return _rate(_as_column(text).str.regexp_matches(r"\p{Arabic}"))


def emoji_rate(text: IntoExpr) -> Expr:
    """The fraction of outputs containing at least one emoji or pictograph — a spam detector.

    Flags any output with a character in the emoji and pictographic codepoint ranges, then
    reports the share of flagged outputs over the corpus. It is the catch for a model that
    fills otherwise-plain text with emoji. A single emoji is enough to flag an output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The emoji rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["party 🎉", "no emoji here"]})
            >>> ds.agg(m=bt.emoji_rate("o")).to_pydict()
            {'m': [0.5]}
    """
    return _rate(_as_column(text).str.regexp_matches(_EMOJI_RE))


def latin_only_rate(text: IntoExpr) -> Expr:
    """The fraction of outputs that are pure ASCII/Latin text — the clean-English output rate.

    Flags an output only when it contains no non-ASCII character at all, then reports the share
    of clean outputs over the corpus. It is the positive complement of the script-drift metrics:
    a high rate means the corpus stayed in plain English/ASCII, and a drop signals that some
    other script or an emoji crept in. Any non-ASCII character disqualifies an output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The pure-ASCII/Latin rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["hello 世界", "plain english", "привет"]})
            >>> ds.agg(m=bt.latin_only_rate("o")).to_pydict()
            {'m': [0.3333333333333333]}
    """
    return _rate(~_as_column(text).str.regexp_matches(_NON_ASCII_RE))
