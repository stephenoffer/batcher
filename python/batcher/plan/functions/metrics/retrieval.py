"""Retrieval-grounding metrics — scoring a generated answer against its retrieved context.

Evaluating a retrieval-augmented generation (RAG) system means asking whether the answer the model
produced is actually supported by the context that was retrieved for it. The honest, model-free
signals for that are lexical-overlap measures between the answer's tokens and the context's tokens:
how much of the answer the context backs (groundedness), how much of the context the answer used
(utilization), and how many answer tokens have no support at all (a hallucination proxy). Each one
here is a per-row expression over two text columns that aggregates to a corpus score, so a RAG eval
over a million answer/context pairs is one scan through the engine rather than a Python loop.

The token metrics are *set*-based — they compare the sets of tokens, so a repeated token counts
once. Normalization (lowercasing, stripping punctuation and articles) follows the SQuAD convention
reused from `generation._tokens`, so an answer that differs from its context only in casing or a
trailing period still counts as supported. These are surface-overlap proxies, not a semantic
judge: they catch extractive grounding well and paraphrased grounding poorly, which is the right,
cheap first-pass filter before a more expensive model-based faithfulness check.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.generation import _tokens

__all__ = [
    "answer_groundedness",
    "citation_rate",
    "context_utilization",
    "fully_grounded_rate",
    "unsupported_token_rate",
]


def answer_groundedness(answer: IntoExpr, context: IntoExpr) -> Expr:
    """The mean fraction of an answer's tokens that also appear in its retrieved context.

    Over the *set* of tokens (repeats counted once): the answer/context intersection divided by the
    answer's token count, averaged per row over the corpus. It is token-set precision of the answer
    against the context, named for the RAG use — a high score means most of what the answer says is
    backed by the retrieved passage, a low score means the answer wandered off the evidence. Rows
    with an empty answer contribute zero rather than dividing by zero.

    Args:
        answer: The generated-answer column.
        context: The retrieved-context column.

    Returns:
        The mean answer-groundedness over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "a": ["the cat sat", "a dog barked loudly"],
            ...         "c": ["the cat sat on the mat", "the cat sat on the mat"],
            ...     }
            ... )
            >>> round(ds.agg(g=bt.answer_groundedness("a", "c")).to_pydict()["g"][0], 4)
            0.5
    """
    ans, ctx = _tokens(_as_column(answer)), _tokens(_as_column(context))
    intersection = ans.list.set_intersection(ctx).list.len()
    answer_size = ans.list.n_unique()
    ratio = when(answer_size > lit(0)).then(intersection / answer_size).otherwise(lit(0.0))
    return ratio.mean()


def context_utilization(answer: IntoExpr, context: IntoExpr) -> Expr:
    """The mean fraction of a retrieved context's tokens that the answer actually drew on.

    Over the *set* of tokens: the answer/context intersection divided by the context's token count,
    averaged per row over the corpus. It is the mirror of `answer_groundedness` — a low score means
    the retriever handed the model far more text than the answer used, which is a signal that the
    retrieved chunks are too coarse or the answer is too terse. Rows with an empty context
    contribute zero rather than dividing by zero.

    Args:
        answer: The generated-answer column.
        context: The retrieved-context column.

    Returns:
        The mean context-utilization over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["the cat sat"], "c": ["the cat sat on the mat"]})
            >>> round(ds.agg(u=bt.context_utilization("a", "c")).to_pydict()["u"][0], 4)
            0.5
    """
    ans, ctx = _tokens(_as_column(answer)), _tokens(_as_column(context))
    intersection = ans.list.set_intersection(ctx).list.len()
    context_size = ctx.list.n_unique()
    ratio = when(context_size > lit(0)).then(intersection / context_size).otherwise(lit(0.0))
    return ratio.mean()


def unsupported_token_rate(answer: IntoExpr, context: IntoExpr) -> Expr:
    """The mean fraction of an answer's tokens that are absent from its retrieved context.

    Over the *set* of tokens: the answer-minus-context difference divided by the answer's token
    count, averaged per row over the corpus. It is a hallucination proxy — the words the model
    produced that the evidence never mentioned. By construction it equals
    ``1 - answer_groundedness`` on any non-empty answer, but reads the opposite way (higher is
    worse) on a dashboard, so it
    earns its own name. Rows with an empty answer contribute zero rather than dividing by zero.

    Args:
        answer: The generated-answer column.
        context: The retrieved-context column.

    Returns:
        The mean unsupported-token rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": ["the cat ran fast"], "c": ["the cat sat on the mat"]})
            >>> round(ds.agg(h=bt.unsupported_token_rate("a", "c")).to_pydict()["h"][0], 4)
            0.6667
    """
    ans, ctx = _tokens(_as_column(answer)), _tokens(_as_column(context))
    unsupported = ans.list.set_difference(ctx).list.len()
    answer_size = ans.list.n_unique()
    ratio = when(answer_size > lit(0)).then(unsupported / answer_size).otherwise(lit(0.0))
    return ratio.mean()


def fully_grounded_rate(answer: IntoExpr, context: IntoExpr) -> Expr:
    """The fraction of rows whose every answer token appears in the retrieved context.

    A stricter, all-or-nothing sibling of `answer_groundedness`: a row counts only when the answer
    has at least one token and *none* of its tokens are missing from the context. It is the corpus
    rate of fully-supported answers, the number you quote when a single unsupported word is a
    failure — a compliance or citation-required setting rather than a graded-overlap one. An empty
    answer never counts as grounded.

    Args:
        answer: The generated-answer column.
        context: The retrieved-context column.

    Returns:
        The fraction of fully-grounded rows over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "a": ["the cat sat", "the cat flew"],
            ...         "c": ["the cat sat on the mat", "the cat sat on the mat"],
            ...     }
            ... )
            >>> ds.agg(f=bt.fully_grounded_rate("a", "c")).to_pydict()["f"][0]
            0.5
    """
    ans, ctx = _tokens(_as_column(answer)), _tokens(_as_column(context))
    unsupported = ans.list.set_difference(ctx).list.len()
    non_empty = ans.list.len() > lit(0)
    grounded = (unsupported == lit(0)) & non_empty
    return count_if(grounded) / count_if(lit(True))


def citation_rate(text: IntoExpr) -> Expr:
    """The fraction of rows whose text contains a bracketed citation marker such as ``[1]``.

    A single-column corpus rate: how often the generated text carries an inline numeric citation
    like ``[1]`` or ``[42]``. It does not check that the citation is correct, only that the model
    cited at all, which is the cheap first gate for a citation-required RAG system — a zero here
    means the answers are uncited regardless of how grounded they are.

    Args:
        text: The generated-text column to scan for citation markers.

    Returns:
        The fraction of rows containing a bracketed citation, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"t": ["see [1]", "no citation here", "also [12] and [3]"]})
            >>> round(ds.agg(c=bt.citation_rate("t")).to_pydict()["c"][0], 4)
            0.6667
    """
    has_citation = _as_column(text).str.regexp_matches(r"\[\d+\]")
    return count_if(has_citation) / count_if(lit(True))
