"""Degeneration and diversity metrics — catching a model that has started repeating itself.

A language model failing at generation usually fails the same way: it loops. It repeats a word, a
character run, a line, or a whole paragraph, and the output stays fluent enough to pass a
spot-check while being worthless. The metrics here are the cheap corpus-level detectors for that,
at every granularity — token, character n-gram, and line — plus the blunter output-shape signals
(empty, truncated, refused) that catch a generation that failed outright rather than degenerated.

A healthy corpus scores near 1 on the ratio metrics and near 0 on the rate metrics. A sharp move in
either is the signal to look at the samples.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.text._text import char_ngrams, mean_ratio, tokens
from batcher.plan.functions.string import is_refusal

__all__ = [
    "char_repetition_rate",
    "compression_ratio_proxy",
    "distinct_char_ngram_ratio",
    "distinct_token_ratio",
    "empty_generation_rate",
    "mean_output_tokens",
    "refusal_rate",
    "repeated_line_rate",
    "truncation_rate",
]


def distinct_token_ratio(text: IntoExpr) -> Expr:
    """The mean fraction of distinct tokens per generation — the Distinct-1 diversity score.

    For each output, the count of unique tokens over the total token count, averaged over the
    corpus. It is the standard cheap detector of *degeneration*: a healthy generation scores near 1,
    and a
    model stuck in a repetition loop ("the the the ...") collapses toward 0. Tokens are the
    SQuAD-normalized whitespace tokens, so casing and punctuation do not inflate the count.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean distinct-token ratio over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["cat sat mat", "cat cat cat cat"]})
            >>> round(ds.agg(d=bt.distinct_token_ratio("o")).to_pydict()["d"][0], 4)
            0.625
    """
    toks = tokens(_as_column(text))
    return mean_ratio(toks.list.n_unique(), toks.list.len())


def mean_output_tokens(text: IntoExpr, chars_per_token: float = 4.0) -> Expr:
    """The mean estimated token length of a generation — the corpus verbosity number.

    The average of the per-row token estimate (characters over ``chars_per_token``). It tracks
    length drift between models or prompt versions and sizes the output-token bill before it
    arrives. It is an estimate, not a real tokenizer count, so compare it against itself across runs
    rather than treating it as an exact token total.

    Args:
        text: The generated-text column.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The mean estimated token count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["12345678", "1234"]})
            >>> ds.agg(t=bt.mean_output_tokens("o", chars_per_token=4.0)).to_pydict()["t"][0]
            1.5
    """
    return _as_column(text).str.estimate_tokens(chars_per_token).mean()


def empty_generation_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that are empty or whitespace only — a silent-failure rate.

    A model that returns an empty string on some inputs looks fine row by row and poisons any
    downstream aggregate. This counts the blank outputs as a corpus rate, so the failure is one
    number on a dashboard instead of a surprise null later. Whitespace-only counts as empty.

    Args:
        text: The generated-text column.

    Returns:
        The empty-generation rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["an answer", "   ", ""]})
            >>> round(ds.agg(e=bt.empty_generation_rate("o")).to_pydict()["e"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.is_blank()) / count_if(lit(True))


def refusal_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that read as a refusal — the corpus refusal rate.

    A safety-eval and helpfulness number: how often the model declined ("I'm sorry, I can't ...",
    "As an AI ...") instead of answering. It uses the same phrasing detector as :func:`is_refusal`,
    aggregated to a rate, so you can track it per model or per prompt category and catch an
    over-cautious release. It is a lexical heuristic, so treat it as a monitor, not a verdict.

    Args:
        text: The generated-text column.

    Returns:
        The refusal rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["I'm sorry, I can't help.", "The answer is 4."]})
            >>> ds.agg(r=bt.refusal_rate("o")).to_pydict()["r"][0]
            0.5
    """
    return count_if(is_refusal(_as_column(text))) / count_if(lit(True))


def truncation_rate(text: IntoExpr) -> Expr:
    """The fraction of non-empty generations that do not end in terminal punctuation.

    A generation cut off by a token limit usually stops mid-sentence, so an output that ends in
    something other than ``.``, ``!``, or ``?`` is a cheap proxy for truncation. It is only a proxy
    (a list or a code block legitimately ends without terminal punctuation), so read a rising rate
    between runs as a signal to raise the max-token budget, not as an exact truncation count. Empty
    outputs are excluded so they do not double-count with `empty_generation_rate`.

    Args:
        text: The generated-text column.

    Returns:
        The truncation-proxy rate over the non-empty corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["A full sentence.", "cut off here", ""]})
            >>> ds.agg(t=bt.truncation_rate("o")).to_pydict()["t"][0]
            0.5
    """
    value = _as_column(text)
    non_empty = ~value.str.is_blank()
    truncated = non_empty & ~value.str.ends_with_punctuation()
    return count_if(truncated) / count_if(non_empty)


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
    ngrams = char_ngrams(_as_column(text), n)
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
    ngrams = char_ngrams(_as_column(text), n)
    total = ngrams.list.len()
    rate = when(total > lit(0)).then(lit(1.0) - ngrams.list.n_unique() / total).otherwise(lit(0.0))
    return rate.mean()


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
    ngrams = char_ngrams(_as_column(text), n)
    unique = ngrams.list.n_unique()
    ratio = when(unique > lit(0)).then(ngrams.list.len() / unique).otherwise(lit(0.0))
    return ratio.mean()
