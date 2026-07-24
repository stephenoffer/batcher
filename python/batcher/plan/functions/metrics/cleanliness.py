"""Whitespace-hygiene metrics — the little output-cleanliness defects in LLM-generated text.

Stray whitespace is the quiet failure mode of generated text: a trailing space, a leading tab, a
doubled space, or a blank-line gap reads fine to a human but breaks a downstream parser, a diff, or
a renderer, and a prompt tweak can start emitting it without any visible change in the answer. These
measure the rate of each defect directly, as a corpus number, so a regression in output cleanliness
surfaces before a reader or a strict parser trips on it. Each is a single mergeable aggregate over
the string primitives and composes inside `group_by`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "blank_line_rate",
    "double_space_rate",
    "empty_or_whitespace_rate",
    "has_tab_rate",
    "leading_whitespace_rate",
    "trailing_whitespace_rate",
]


def trailing_whitespace_rate(text: IntoExpr) -> Expr:
    """The fraction of generations whose final character is whitespace — a trailing-whitespace rate.

    Matches ``\\s$``, which anchors at the absolute end of the string (the Rust regex ``$`` is not
    multiline here), so a generation counts when its last character is a space, a tab, or a
    newline. A trailing newline is therefore itself flagged as trailing whitespace. A rise between
    runs points at a prompt change that started leaving stray whitespace at the end of answers.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The trailing-whitespace rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["clean text", "trailing space ", "\\ttabbed"]})
            >>> round(ds.agg(t=bt.trailing_whitespace_rate("o")).to_pydict()["t"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"\s$")) / count_if(lit(True))


def leading_whitespace_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that begin with a whitespace char — a leading-whitespace rate.

    Matches ``^\\s``, one space, tab, or newline at the very start of the string. A model that
    indents or blank-pads the top of its answer produces output that misaligns when concatenated or
    re-wrapped, and this catches it as a corpus number rather than one row at a time.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The leading-whitespace rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["clean text", "trailing space ", "\\ttabbed"]})
            >>> round(ds.agg(le=bt.leading_whitespace_rate("o")).to_pydict()["le"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"^\s")) / count_if(lit(True))


def has_tab_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain at least one tab character — a tab-presence rate.

    Counts a generation when its tab count is positive, using the ``tab_count`` string primitive
    rather than a regex. Tabs render inconsistently across viewers and break column-aligned or
    tab-separated downstream parsing, so a nonzero rate is worth watching even when the text looks
    fine.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The tab-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["clean text", "trailing space ", "\\ttabbed"]})
            >>> round(ds.agg(h=bt.has_tab_rate("o")).to_pydict()["h"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.tab_count() > lit(0)) / count_if(lit(True))


def double_space_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a run of two or more spaces — a doubled-space rate.

    Matches ``  +`` (a space followed by one or more spaces) anywhere in the string. Collapsed or
    doubled spacing survives most rendering but corrupts fixed-width layout and shows up as noise in
    a diff, so this flags the outputs where the model doubled a space.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The doubled-space rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["a  b", "a b", "a   b"]})
            >>> round(ds.agg(d=bt.double_space_rate("o")).to_pydict()["d"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(r"  +")) / count_if(lit(True))


def blank_line_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a blank line — a blank-line-gap rate.

    Matches ``\\n[ \\t]*\\n``, two newlines separated only by optional spaces or tabs, so a fully
    empty line or a line of only whitespace between two content lines both count. Blank-line gaps
    inflate output and can split a record that a downstream reader expected to be contiguous.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The blank-line rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["para one\\n\\npara two", "single line", "no gap"]})
            >>> round(ds.agg(b=bt.blank_line_rate("o")).to_pydict()["b"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"\n[ \t]*\n")) / count_if(lit(True))


def empty_or_whitespace_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that are empty or contain only whitespace — a blank-output rate.

    Matches ``^\\s*$``, which requires the whole string to be zero or more whitespace characters
    from start to end, so both an empty string and one made only of spaces, tabs, or newlines
    count. This is the hard failure of output cleanliness: the model returned nothing usable at all,
    and it is distinct from an empty-generation check on a separate column.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The empty-or-whitespace-only rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["real answer", "", "   "]})
            >>> round(ds.agg(e=bt.empty_or_whitespace_rate("o")).to_pydict()["e"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(r"^\s*$")) / count_if(lit(True))
