"""Generation-quality metrics that score an output column on its own, with no reference.

The reference-based metrics in `generation.py` need a gold column. These need only the model's
output, and they are the numbers a team running generation at scale watches on a dashboard: is
the text diverse or degenerating into repetition, how long is it, how often is it empty, a
refusal, or cut off mid-sentence. Each is a single mergeable aggregate over the existing string
primitives, so a check over a million generations is one scan and composes inside `group_by` to
break the number down per model, per prompt template, or per day.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics._text import mean_ratio, tokens
from batcher.plan.functions.string import is_refusal

__all__ = [
    "distinct_token_ratio",
    "empty_generation_rate",
    "mean_output_tokens",
    "refusal_rate",
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
