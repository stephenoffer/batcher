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

The phrase-level pair (`phrase_groundedness`, `unsupported_phrase_rate`) is the harder test, and
the one that catches a confident hallucination: an answer built from the context's own vocabulary,
rearranged into a claim the context never made, scores well on token overlap and badly on spans.

A final group scores the *retrieval* rather than the answer, over a list column of passages —
whether anything came back at all, whether the same chunk came back twice, and what the assembled
context is about to cost.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.text._text import mean_ratio, token_ngrams, tokens

__all__ = [
    "answer_groundedness",
    "citation_rate",
    "context_token_estimate",
    "context_utilization",
    "duplicate_context_rate",
    "empty_retrieval_rate",
    "fully_grounded_rate",
    "mean_retrieved_passages",
    "phrase_groundedness",
    "unsupported_phrase_rate",
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
    ans, ctx = tokens(_as_column(answer)), tokens(_as_column(context))
    intersection = ans.list.set_intersection(ctx).list.len()
    return mean_ratio(intersection, ans.list.n_unique())


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
    ans, ctx = tokens(_as_column(answer)), tokens(_as_column(context))
    intersection = ans.list.set_intersection(ctx).list.len()
    return mean_ratio(intersection, ctx.list.n_unique())


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
    ans, ctx = tokens(_as_column(answer)), tokens(_as_column(context))
    unsupported = ans.list.set_difference(ctx).list.len()
    return mean_ratio(unsupported, ans.list.n_unique())


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
    ans, ctx = tokens(_as_column(answer)), tokens(_as_column(context))
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


def phrase_groundedness(answer: IntoExpr, context: IntoExpr, n: int = 3) -> Expr:
    """The mean fraction of an answer's word n-grams that appear in its retrieved context.

    The phrase-level counterpart of `answer_groundedness`, and a much harder test. Token-set
    grounding scores an answer that reuses the context's vocabulary while rearranging it into a
    claim the context never made — which is exactly what a confident hallucination looks like.
    Requiring whole `n`-token spans to match catches that: the words are all there, the phrases
    are not.

    Read the two together. A high token groundedness with a low phrase groundedness is the
    signature of an answer assembled from the right material into the wrong statement.

    Args:
        answer: The generated-answer column.
        context: The retrieved-context column.
        n: The span length, in tokens, that must match.

    Returns:
        The mean phrase-groundedness over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "a": ["the cat sat quietly"],
            ...         "c": ["the cat sat quietly on the mat"],
            ...     }
            ... )
            >>> ds.agg(g=bt.phrase_groundedness("a", "c")).to_pydict()["g"][0]
            1.0
    """
    _require_span(n, "phrase_groundedness")
    ans = token_ngrams(_as_column(answer), n)
    ctx = token_ngrams(_as_column(context), n)
    return mean_ratio(ans.list.multiset_overlap(ctx), ans.list.len())


def unsupported_phrase_rate(answer: IntoExpr, context: IntoExpr, n: int = 3) -> Expr:
    """The mean fraction of an answer's word n-grams that its context does not contain.

    One minus `phrase_groundedness`, reported directly because it is the number a dashboard
    watches: it goes *up* when the system gets worse, so a threshold and an alert read the way
    you expect. Use it alongside `unsupported_token_rate`, which measures the same failure at
    the vocabulary level and is far more forgiving.

    Args:
        answer: The generated-answer column.
        context: The retrieved-context column.
        n: The span length, in tokens.

    Returns:
        The mean unsupported-phrase rate over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"a": ["the moon is cheese"], "c": ["the moon orbits the earth"]}
            ... )
            >>> ds.agg(u=bt.unsupported_phrase_rate("a", "c")).to_pydict()["u"][0]
            1.0
    """
    _require_span(n, "unsupported_phrase_rate")
    ans = token_ngrams(_as_column(answer), n)
    ctx = token_ngrams(_as_column(context), n)
    unsupported = ans.list.len() - ans.list.multiset_overlap(ctx)
    return mean_ratio(unsupported, ans.list.len())


def _require_span(n: int, func: str) -> None:
    """Reject a span length that cannot form an n-gram."""
    from batcher._internal.errors import PlanError

    if n < 1:
        raise PlanError(f"{func}: n must be at least 1, got {n}")


def empty_retrieval_rate(hits: IntoExpr) -> Expr:
    """The fraction of queries whose retriever returned nothing.

    The RAG failure that is invisible downstream: with no context the model answers from its
    parameters, fluently and with no citation, and every grounding metric above scores that row
    on an empty context rather than flagging it. A non-zero rate here explains a block of
    otherwise inexplicable hallucinations.

    Args:
        hits: A list column holding each query's retrieved passages.

    Returns:
        The empty-retrieval rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"hits": [["a passage"], []]})
            >>> ds.agg(e=bt.empty_retrieval_rate("hits")).to_pydict()["e"][0]
            0.5
    """
    column = _as_column(hits)
    return count_if(column.list.drop_nulls().list.len() == lit(0)) / count_if(lit(True))


def duplicate_context_rate(hits: IntoExpr) -> Expr:
    """The fraction of queries whose retrieved passages contain a duplicate.

    A chunk indexed twice, or overlapping chunks that both matched, spends the context window
    twice on the same text and biases the model toward whatever it repeats. It is easy to
    introduce during an index rebuild and hard to see one query at a time.

    Compares whole passages, so it catches an exact repeat rather than near-duplicates; for
    those, deduplicate on `str.minhash` before assembling the context.

    Args:
        hits: A list column holding each query's retrieved passages.

    Returns:
        The duplicate-passage rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"hits": [["a", "a", "b"], ["a", "b"]]})
            >>> ds.agg(d=bt.duplicate_context_rate("hits")).to_pydict()["d"][0]
            0.5
    """
    column = _as_column(hits).list.drop_nulls()
    return count_if(column.list.n_unique() < column.list.len()) / count_if(lit(True))


def mean_retrieved_passages(hits: IntoExpr) -> Expr:
    """The mean number of passages a query's retriever returned.

    Read it beside `empty_retrieval_rate`. A mean well below the `k` you asked for means the
    retriever is running out of candidates above its score threshold, which is a different
    problem from returning `k` irrelevant ones and is fixed in a different place.

    Args:
        hits: A list column holding each query's retrieved passages.

    Returns:
        The mean passage count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"hits": [["a", "b", "c"], ["a"]]})
            >>> ds.agg(k=bt.mean_retrieved_passages("hits")).to_pydict()["k"][0]
            2.0
    """
    return _as_column(hits).list.drop_nulls().list.len().mean()


def context_token_estimate(hits: IntoExpr, chars_per_token: float = 4.0) -> Expr:
    """The mean estimated tokens in a query's assembled context — the RAG cost driver.

    Retrieved context is usually the largest part of a RAG prompt and the part that grows
    silently: raise `k` from 5 to 10 and every request doubles its input bill. This sizes it
    before the run, from the passage text rather than from an assumed chunk size, so an
    oversized chunk shows up.

    The estimate divides characters by `chars_per_token` rather than running a tokenizer, which
    would be per-row Python on the hot path. Leave headroom.

    Args:
        hits: A list column holding each query's retrieved passages.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The mean estimated context length, in tokens.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"hits": [["12345678", "1234"]]})
            >>> ds.agg(t=bt.context_token_estimate("hits")).to_pydict()["t"][0]
            3.0
    """
    joined = _as_column(hits).list.drop_nulls().list.join("")
    return joined.str.estimate_tokens(chars_per_token).mean()
