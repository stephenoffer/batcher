"""Repetition and degeneration metrics — corpus-level detectors of LLM text looping.

A common failure mode of a language model is *degeneration*: the decoder falls into a loop and
repeats a character n-gram, a word, or a whole line. Each metric here is a single mergeable
aggregate over one output column that turns that failure into one dashboard number, so a check
over a million generations is one scan and composes inside `group_by` to break the number down
per model, per prompt template, or per day.

The character-level metrics normalize case and whitespace before chunking, so ``"AB  ab"`` and
``"ab ab"`` collapse together. The word-level metric uses the SQuAD-normalized tokens (articles
and punctuation dropped, lowercased), matching the rest of the generation-metric family.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.generation import _tokens

__all__ = [
    "char_repetition_rate",
    "compression_ratio_proxy",
    "distinct_char_ngram_ratio",
    "repeated_line_rate",
    "word_type_token_ratio",
]


def _char_ngrams(text: Expr, n: int) -> Expr:
    """The character n-grams of a case-folded, space-collapsed text column, as a list."""
    normalized = text.str.lower().str.normalize_whitespace().str.strip()
    return normalized.str.chunk(n, overlap=n - 1)


def distinct_char_ngram_ratio(text: IntoExpr, n: int = 3) -> Expr:
    """The mean fraction of distinct character n-grams per output — a character-level Distinct-n.

    For each output, the count of unique character n-grams over the total count, averaged over the
    corpus. It is the tokenizer-free degeneration detector: healthy text scores near 1, and a model
    stuck repeating the same characters ("aaaa ...") collapses toward 0. Case and whitespace are
    normalized before chunking, so surface noise does not inflate the diversity.

    Args:
        text: The generated-text column (name or expression).
        n: The character n-gram size.

    Returns:
        The mean distinct character n-gram ratio over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["aaaa"]})
            >>> round(ds.agg(d=bt.distinct_char_ngram_ratio("o", n=2)).to_pydict()["d"][0], 4)
            0.3333
    """
    ngrams = _char_ngrams(_as_column(text), n)
    total = ngrams.list.len()
    ratio = when(total > lit(0)).then(ngrams.list.n_unique() / total).otherwise(lit(0.0))
    return ratio.mean()


def char_repetition_rate(text: IntoExpr, n: int = 3) -> Expr:
    """The mean fraction of character n-grams that are repeats — the complement of Distinct-n.

    For each output, one minus the distinct-ratio: the share of character n-grams that recur,
    averaged over the corpus. It is the direct "how repetitive is the text" number, near 0 for
    healthy text and climbing toward 1 as the model loops. On outputs with at least one n-gram it
    equals ``1 - distinct_char_ngram_ratio``; empty outputs are guarded to 0.

    Args:
        text: The generated-text column (name or expression).
        n: The character n-gram size.

    Returns:
        The mean character n-gram repetition rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["aaaa"]})
            >>> round(ds.agg(r=bt.char_repetition_rate("o", n=2)).to_pydict()["r"][0], 4)
            0.6667
    """
    ngrams = _char_ngrams(_as_column(text), n)
    total = ngrams.list.len()
    rate = when(total > lit(0)).then(lit(1.0) - ngrams.list.n_unique() / total).otherwise(lit(0.0))
    return rate.mean()


def word_type_token_ratio(text: IntoExpr) -> Expr:
    """The mean fraction of distinct word tokens per output — the word-level type-token ratio.

    For each output, the count of unique tokens over the total token count, averaged over the
    corpus. It is the word-level diversity signal: a healthy generation scores near 1, and a model
    repeating words ("the the the ...") collapses toward 0. Tokens are the SQuAD-normalized
    whitespace tokens, so casing, punctuation, and articles do not distort the count.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean word type-token ratio over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["the cat sat", "dog dog dog"]})
            >>> round(ds.agg(w=bt.word_type_token_ratio("o")).to_pydict()["w"][0], 4)
            0.6667
    """
    tokens = _tokens(_as_column(text))
    total = tokens.list.len()
    ratio = when(total > lit(0)).then(tokens.list.n_unique() / total).otherwise(lit(0.0))
    return ratio.mean()


def repeated_line_rate(text: IntoExpr) -> Expr:
    """The fraction of outputs that contain at least one duplicated line — a looping detector.

    Splits each output on newlines and flags it when it has fewer unique lines than total lines,
    then reports the share of flagged outputs over the corpus. It is the cheapest catch for the
    list- or paragraph-repetition failure, where a model emits the same line twice. An output with
    all-distinct lines does not count.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The repeated-line rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["a\\na", "x\\ny\\nz"]})
            >>> round(ds.agg(r=bt.repeated_line_rate("o")).to_pydict()["r"][0], 4)
            0.5
    """
    lines = _as_column(text).str.split("\n")
    has_repeat = lines.list.n_unique() < lines.list.len()
    return count_if(has_repeat) / count_if(lit(True))


def compression_ratio_proxy(text: IntoExpr, n: int = 3) -> Expr:
    """The mean ratio of total to distinct character n-grams — a gzip-ratio degeneration proxy.

    For each output, the total character n-gram count over the distinct count, averaged over the
    corpus, the reciprocal of the distinct-ratio. A value near 1 means diverse text; a large value
    means heavy repetition, the way a high gzip compression ratio flags a degenerate output. It is
    a cheap stand-in for running an actual compressor over every generation.

    Args:
        text: The generated-text column (name or expression).
        n: The character n-gram size.

    Returns:
        The mean compression-ratio proxy over the corpus, ``>= 1`` for non-empty outputs.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["aaaa"]})
            >>> round(ds.agg(c=bt.compression_ratio_proxy("o", n=2)).to_pydict()["c"][0], 4)
            3.0
    """
    ngrams = _char_ngrams(_as_column(text), n)
    unique = ngrams.list.n_unique()
    ratio = when(unique > lit(0)).then(ngrams.list.len() / unique).otherwise(lit(0.0))
    return ratio.mean()
