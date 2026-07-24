"""Corpus-level readability and lexical-complexity metrics for a generated-text column.

Where `text_quality.py` flags what a generation *should not do*, these summarize how *hard to
read* a slice of generations is, using the standard readability signals: the automated
readability index, words per sentence, characters per word, and the rate of long words. Each is a
single mergeable aggregate over the string primitives, so a score over a million outputs is one
scan and breaks down per model or per day with `group_by`. They are lexical heuristics, useful as
between-run regression detectors rather than judgments of a single row.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column

__all__ = [
    "automated_readability_index",
    "long_word_rate",
    "mean_chars_per_word",
    "mean_paragraph_count",
    "mean_words_per_sentence",
]


def _safe_div(num: Expr, den: Expr) -> Expr:
    """``num / den`` where ``den > 0``, else ``0.0`` — a per-row divide-by-zero guard."""
    return when(den > lit(0)).then(num / den).otherwise(lit(0.0))


def automated_readability_index(text: IntoExpr) -> Expr:
    """Mean automated readability index (ARI) over the corpus — a US-grade readability score.

    ARI scores a row as ``4.71 * chars/words + 0.5 * words/sentences - 21.43`` and approximates
    the US school grade needed to read it. A higher value means denser, harder text. Rows with no
    words or no sentences contribute a guarded 0 for that term, so an empty output cannot divide by
    zero. The corpus score is the mean of the per-row scores, mergeable across partitions.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean per-row ARI over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
            >>> round(ds.agg(m=bt.automated_readability_index("o")).to_pydict()["m"][0], 4)
            -0.34
    """
    col = _as_column(text)
    chars = col.str.len_chars()
    words = col.str.word_count()
    sentences = col.str.sentence_count()
    score = (
        lit(4.71) * _safe_div(chars, words)
        + lit(0.5) * _safe_div(words, sentences)
        - lit(21.43)
    )
    return score.mean()


def mean_words_per_sentence(text: IntoExpr) -> Expr:
    """Mean words per sentence over the corpus — the core readability driver in one number.

    For each row this divides the word count by the sentence count, guarding an unpunctuated row
    (zero sentences) with a 0, then averages across rows. Long sentences read as denser prose, so
    a rising value across runs flags a drift toward harder output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean of the per-row words-per-sentence ratio over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
            >>> round(ds.agg(m=bt.mean_words_per_sentence("o")).to_pydict()["m"][0], 4)
            4.5
    """
    col = _as_column(text)
    ratio = _safe_div(col.str.word_count(), col.str.sentence_count())
    return ratio.mean()


def mean_chars_per_word(text: IntoExpr) -> Expr:
    """Mean characters per word over the corpus — a tokenizer-free lexical-complexity signal.

    Each row contributes its average letters-per-word, and the corpus score is the mean of those.
    Longer words track denser vocabulary, so this is a cheap complementary signal to the sentence
    length driver.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean of the per-row average word length over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
            >>> round(ds.agg(m=bt.mean_chars_per_word("o")).to_pydict()["m"][0], 4)
            2.8889
    """
    return _as_column(text).str.avg_word_length().mean()


def long_word_rate(text: IntoExpr, min_length: int = 7) -> Expr:
    """Mean fraction of long words per row over the corpus — a lexical-complexity proxy.

    For each row this divides the count of words at least `min_length` characters by the total word
    count, guarding an empty row with a 0, then averages across rows. A high long-word fraction
    marks technical or ornate prose, so this tracks how advanced the vocabulary of a slice is.

    Args:
        text: The generated-text column (name or expression).
        min_length: The minimum character length for a word to count as long.

    Returns:
        The mean of the per-row long-word fraction over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["extraordinary complicated situations", "the cat sat"]})
            >>> round(ds.agg(m=bt.long_word_rate("o")).to_pydict()["m"][0], 4)
            0.5
    """
    col = _as_column(text)
    ratio = _safe_div(col.str.long_word_count(min_length), col.str.word_count())
    return ratio.mean()


def mean_paragraph_count(text: IntoExpr) -> Expr:
    """Mean number of paragraphs per generation over the corpus — a structural-density signal.

    Each row contributes its paragraph count (blank-line-separated blocks), and the corpus score is
    the mean of those. More paragraphs mark longer, more structured output, so a shift here flags a
    change in generation shape between runs.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean per-row paragraph count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["First para.\\n\\nSecond para.\\n\\nThird."]})
            >>> round(ds.agg(m=bt.mean_paragraph_count("o")).to_pydict()["m"][0], 4)
            3.0
    """
    return _as_column(text).str.paragraph_count().mean()
