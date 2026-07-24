"""Token and cost aggregates — sizing an LLM run before you pay for it.

Generation and embedding bills are token bills, and the questions that decide capacity are
aggregate: how many tokens will this corpus cost in total, what fraction of rows will overflow the
context window, and how long is the tail you must size the window for. Each is a single mergeable
aggregate over the tokenizer-free `estimate_tokens` heuristic, so a cost estimate over a hundred
million rows is one scan and breaks down per model or per tenant with `group_by`. The counts are
estimates (characters over ``chars_per_token``), so treat them as a planning number, not a bill.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.quantiles import approx_quantile

__all__ = [
    "token_budget_exceed_rate",
    "token_estimate_quantile",
    "total_token_estimate",
]


def total_token_estimate(text: IntoExpr, chars_per_token: float = 4.0) -> Expr:
    """The total estimated token count over a text column — the corpus cost number.

    The sum of the per-row token estimate, which is what a per-token bill is charged on. Run it on
    the prompt column to size an input cost and on the output column to size a generation cost, and
    put it inside `group_by` to attribute spend per model, per tenant, or per day. It is an estimate
    (characters over ``chars_per_token``), so use it to plan and compare, not to reconcile a bill.

    Args:
        text: The text column (name or expression) to size.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The total estimated token count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["12345678", "1234"]})
            >>> ds.agg(t=bt.total_token_estimate("o", chars_per_token=4.0)).to_pydict()["t"][0]
            3
    """
    return _as_column(text).str.estimate_tokens(chars_per_token).sum()


def token_budget_exceed_rate(text: IntoExpr, budget: int, chars_per_token: float = 4.0) -> Expr:
    """The fraction of rows whose estimated tokens exceed ``budget`` — the context-overflow rate.

    A row longer than the context window is silently truncated by most engines, losing its tail.
    This is the corpus rate of those rows, the number that tells you whether to raise the window,
    chunk the inputs, or accept the loss before a run rather than discovering it in the outputs.

    Args:
        text: The text column to size.
        budget: The token budget a row must exceed to count as over.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The over-budget rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["abcd", "abcdefghijkl"]})
            >>> ds.agg(r=bt.token_budget_exceed_rate("o", budget=2)).to_pydict()["r"][0]
            0.5
    """
    fits = _as_column(text).str.fits_token_budget(budget, chars_per_token)
    return count_if(~fits) / count_if(lit(True))


def token_estimate_quantile(text: IntoExpr, q: float, chars_per_token: float = 4.0) -> AggExpr:
    """The ``q``-quantile of the per-row estimated token count — the tail that sizes the window.

    The mean token length hides the tail, and it is the tail that overflows a context window. This
    is the approximate ``q``-quantile of the per-row estimate over a mergeable sketch, so at
    ``q = 0.95`` it is the window that covers 95% of inputs without truncation. Use it to pick a
    context size or a per-request cap from the data rather than a guess, sketch-approximate.

    Args:
        text: The text column to size.
        q: The quantile in ``[0, 1]`` (``0.95`` for the 95th percentile).
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The approximate ``q``-quantile of the per-row token estimate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["1234", "12345678", "123456789012"]})
            >>> round(ds.agg(p=bt.token_estimate_quantile("o", q=0.5)).to_pydict()["p"][0])
            2
    """
    return approx_quantile(_as_column(text).str.estimate_tokens(chars_per_token), q)
