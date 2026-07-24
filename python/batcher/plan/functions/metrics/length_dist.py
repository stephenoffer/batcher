"""Corpus-level length-distribution metrics for a generated-text column.

Mean output length hides the tails, and the tails are what break capacity planning: a p95 that
is triple the mean sizes a context window or a token budget, and the single longest generation is
the one that overflows a buffer. These functions summarize the *distribution* of per-row length
rather than its center, over character length and word count, so a slice of generations reports
its spread with one scan and breaks down per model or per day with `group_by`. The quantile
metrics ride a mergeable sketch, so the estimate is identical single-node and distributed but
approximate; the max/min/range metrics are exact.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.quantiles import approx_quantile

__all__ = [
    "char_length_quantile",
    "char_length_range",
    "max_char_length",
    "min_char_length",
    "word_count_quantile",
]


def char_length_quantile(text: IntoExpr, q: float) -> AggExpr:
    """Approximate ``q``-quantile of per-row character length across the corpus.

    Sizes a point in the length tail: ``q=0.95`` is the character length that 95% of generations
    stay under, the number you provision a context window or a truncation limit against. Character
    length comes from `str.len_chars`, and the quantile rides a mergeable sketch, so the estimate
    is identical single-node and distributed but approximate rather than exact.

    Args:
        text: The generated-text column (name or expression).
        q: The quantile to estimate, between 0 and 1.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> round(ds.agg(m=bt.char_length_quantile("o", 0.5)).to_pydict()["m"][0])
            4
    """
    return approx_quantile(_as_column(text).str.len_chars(), q)


def word_count_quantile(text: IntoExpr, q: float) -> AggExpr:
    """Approximate ``q``-quantile of per-row word count across the corpus.

    Sizes a point in the verbosity tail: ``q=0.95`` is the word count that 95% of generations stay
    under, useful when a token budget tracks words more closely than characters. Word count comes
    from `str.word_count`, and the quantile rides a mergeable sketch, so the estimate is identical
    single-node and distributed but approximate rather than exact.

    Args:
        text: The generated-text column (name or expression).
        q: The quantile to estimate, between 0 and 1.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["a b", "a b c", "a b c d e"]})
            >>> round(ds.agg(m=bt.word_count_quantile("o", 0.5)).to_pydict()["m"][0])
            3
    """
    return approx_quantile(_as_column(text).str.word_count(), q)


def max_char_length(text: IntoExpr) -> AggExpr:
    """Exact character length of the longest generation in the corpus.

    The worst-case output, the one that overflows a fixed buffer or blows a truncation limit. This
    is the exact maximum over per-row `str.len_chars`, so it is a single mergeable aggregate that
    breaks down per model or per day with `group_by`.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> ds.agg(m=bt.max_char_length("o")).to_pydict()["m"][0]
            6
    """
    return _as_column(text).str.len_chars().max()


def min_char_length(text: IntoExpr) -> AggExpr:
    """Exact character length of the shortest generation in the corpus.

    The floor of the distribution, where a near-empty or truncated output shows up as a suspiciously
    small value. This is the exact minimum over per-row `str.len_chars`, a single mergeable
    aggregate that breaks down per model or per day with `group_by`.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> ds.agg(m=bt.min_char_length("o")).to_pydict()["m"][0]
            2
    """
    return _as_column(text).str.len_chars().min()


def char_length_range(text: IntoExpr) -> Expr:
    """Exact spread of character length across the corpus — longest minus shortest.

    A single number for how wide the length distribution is: a large range means the corpus mixes
    terse and sprawling generations, a signal worth splitting by model. Character length is read
    once from `str.len_chars`, then its exact maximum and minimum are subtracted, so the result is
    mergeable and breaks down per group.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        An expression for the character-length range; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> ds.agg(m=bt.char_length_range("o")).to_pydict()["m"][0]
            4
    """
    length = _as_column(text).str.len_chars()
    return length.max() - length.min()
