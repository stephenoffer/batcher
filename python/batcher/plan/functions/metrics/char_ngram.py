"""Character n-gram overlap metrics — language-agnostic generation scoring (chrF-style).

The token-set metrics in `generation.py` split on whitespace, which is the wrong unit for a
language that does not put spaces between words (Chinese, Japanese, Thai) or one that inflects
heavily. These compare the *character* n-grams of the two strings instead, so they score overlap
without a tokenizer and without a space assumption. That is the idea behind chrF, the standard
character-F machine-translation metric.

They are *set*-based (a character n-gram is counted once, not by multiplicity), matching the
token-set family, so they answer "did it recover the right character shapes" stably rather than
reproducing the multiset chrF score exactly. Each is a per-row expression over two text columns
that aggregates to a corpus number in one scan and composes inside `group_by`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column

__all__ = [
    "char_ngram_f1",
    "char_ngram_jaccard",
    "char_ngram_precision",
    "char_ngram_recall",
]


def _char_ngrams(text: Expr, n: int) -> Expr:
    """The set of character n-grams of a case-folded, space-collapsed text column, as a list."""
    normalized = text.str.lower().str.normalize_whitespace().str.strip()
    return normalized.str.chunk(n, overlap=n - 1)


def _validate_n(n: int) -> None:
    from batcher._internal.errors import PlanError

    if n < 1:
        raise PlanError(f"char n-gram size must be >= 1, got {n}")


def char_ngram_precision(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram precision — predicted n-grams found in the reference.

    Over the *set* of character n-grams (each counted once): the intersection over the prediction's
    n-gram count, averaged per example. It is the tokenizer-free, language-agnostic counterpart of
    `token_set_precision`, and the right precision half of chrF for text without word boundaries.

    Args:
        prediction: The generated-text column (name or expression).
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram precision, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abcd"], "r": ["abce"]})
            >>> round(ds.agg(p=bt.char_ngram_precision("p", "r", n=2)).to_pydict()["p"][0], 4)
            0.6667
    """
    _validate_n(n)
    pred, gold = _char_ngrams(_as_column(prediction), n), _char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    size = pred.list.n_unique()
    return when(size > lit(0)).then(intersection / size).otherwise(lit(0.0)).mean()


def char_ngram_recall(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram recall — the reference n-grams the prediction produced.

    Over the *set* of character n-grams: the intersection over the reference's n-gram count,
    averaged per example. The tokenizer-free counterpart of `token_set_recall`, and the recall half
    of chrF. chrF weights recall more heavily than precision; use `char_ngram_f1` for the balanced
    combination or read this alongside `char_ngram_precision` to weight them yourself.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram recall, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abce"], "r": ["abcd"]})
            >>> round(ds.agg(r=bt.char_ngram_recall("p", "r", n=2)).to_pydict()["r"][0], 4)
            0.6667
    """
    _validate_n(n)
    pred, gold = _char_ngrams(_as_column(prediction), n), _char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    size = gold.list.n_unique()
    return when(size > lit(0)).then(intersection / size).otherwise(lit(0.0)).mean()


def char_ngram_f1(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram F1 — the chrF-style overlap score.

    The harmonic mean of character n-gram precision and recall per example, averaged over the
    corpus. It is the language-agnostic overlap metric for generation: unlike `token_set_f1` it
    needs no word boundaries, so it scores Chinese, Japanese, or inflected output fairly. This is a
    set-based F1 (β = 1); true chrF uses β = 2 to favor recall, so treat this as a stable chrF-style
    number rather than the exact statistic.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram F1, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abcd"], "r": ["abcd"]})
            >>> ds.agg(f=bt.char_ngram_f1("p", "r", n=2)).to_pydict()["f"][0]
            1.0
    """
    _validate_n(n)
    pred, gold = _char_ngrams(_as_column(prediction), n), _char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    precision = (
        when(pred.list.n_unique() > lit(0))
        .then(intersection / pred.list.n_unique())
        .otherwise(lit(0.0))
    )
    recall = (
        when(gold.list.n_unique() > lit(0))
        .then(intersection / gold.list.n_unique())
        .otherwise(lit(0.0))
    )
    denominator = precision + recall
    f1 = (
        when(denominator > lit(0))
        .then(lit(2.0) * precision * recall / denominator)
        .otherwise(lit(0.0))
    )
    return f1.mean()


def char_ngram_jaccard(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram Jaccard — intersection over union of the n-gram sets.

    The set-similarity view of character n-gram overlap: the shared n-grams over the total distinct
    n-grams across both strings, averaged per example. It is symmetric in the two columns (unlike
    precision and recall) and the tokenizer-free counterpart of `token_set_jaccard`.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram Jaccard index, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abcd"], "r": ["abce"]})
            >>> ds.agg(j=bt.char_ngram_jaccard("p", "r", n=2)).to_pydict()["j"][0]
            0.5
    """
    _validate_n(n)
    pred, gold = _char_ngrams(_as_column(prediction), n), _char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    union = pred.list.set_union(gold).list.len()
    return when(union > lit(0)).then(intersection / union).otherwise(lit(0.0)).mean()
