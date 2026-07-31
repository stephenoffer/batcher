"""Fitting an assembled prompt into a context window, without a tokenizer.

A prompt that overruns the window is not an error you see: the serving stack truncates it and
the model answers from whatever survived, so the failure arrives as a quality regression on a
subset of rows. These functions size and trim a prompt before it is sent.

Every estimate here divides characters by a fixed `chars_per_token`, which is the tokenizer-free
approximation — around 4 for English prose under a byte-pair vocabulary, lower for code and much
lower for a language the vocabulary does not cover well. It is deliberately an estimate: running
a real tokenizer per row would put per-row Python on the hot path, which is the one thing the
control plane must never do. Leave headroom rather than targeting the window exactly, and
calibrate `chars_per_token` on a sample of your own text if the margin matters.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError, require_float, require_int
from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.string import concat

__all__ = [
    "fits_context",
    "prompt_token_estimate",
    "truncate_middle",
    "truncate_to_token_budget",
]


def _char_budget(budget: int, chars_per_token: float, func: str) -> int:
    """The character budget an estimated token budget corresponds to."""
    budget = require_int(budget, func=func, arg="budget", minimum=1)
    chars_per_token = require_float(chars_per_token, func=func, arg="chars_per_token")
    if chars_per_token <= 0:
        raise PlanError(f"{func}: chars_per_token must be positive, got {chars_per_token}")
    return int(budget * chars_per_token)


def truncate_to_token_budget(text: str | Expr, budget: int, chars_per_token: float = 4.0) -> Expr:
    """Trim a text column to fit an estimated token budget — keep a prompt inside the window.

    Cuts each value to ``budget * chars_per_token`` characters, the tokenizer-free way to keep an
    assembled prompt within a model's context window before generation truncates it silently. It is
    a character cut on an estimate, so leave headroom rather than targeting the exact window size.

    The cut takes the head and discards the tail, which is right when the important content leads.
    Use `truncate_middle` when the end matters too, such as a document whose conclusion answers
    the question.

    Args:
        text: The text column (name or expression) to trim.
        budget: The estimated token budget to fit within.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        A string expression trimmed to the estimated budget.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"t": ["hello world"]})
            >>> ds.select(r=bt.truncate_to_token_budget("t", budget=1)).to_pydict()
            {'r': ['hell']}
    """
    chars = _char_budget(budget, chars_per_token, "truncate_to_token_budget")
    return _as_column(text).str.truncate_chars(chars)


def truncate_middle(
    text: str | Expr,
    budget: int,
    chars_per_token: float = 4.0,
    marker: str = "\n...\n",
) -> Expr:
    """Trim a text column to a token budget by removing its *middle*, keeping both ends.

    A head-only cut throws away the end of every document, and for a contract, a transcript, or
    a log the answer is often in the last paragraph. This keeps roughly half the budget from the
    front and half from the back, with `marker` standing in for what was dropped, so a model
    reading it can tell the text is not continuous.

    Values already within the budget are returned unchanged, marker and all — the marker only
    appears where something was actually removed.

    Args:
        text: The text column (name or expression) to trim.
        budget: The estimated token budget to fit within.
        chars_per_token: The average characters per token used for the estimate.
        marker: The text standing in for the removed middle.

    Returns:
        A string expression trimmed from the middle to the estimated budget.

    Raises:
        PlanError: If the marker alone would not fit inside the budget.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"t": ["abcdefghijklmnop"]})
            >>> ds.select(r=bt.truncate_middle("t", budget=2, marker="~")).to_pydict()
            {'r': ['abcd~nop']}

            >>> # Short enough already: returned untouched.
            >>> bt.from_pydict({"t": ["abc"]}).select(
            ...     r=bt.truncate_middle("t", budget=4, marker="~")
            ... ).to_pydict()
            {'r': ['abc']}
    """
    chars = _char_budget(budget, chars_per_token, "truncate_middle")
    if len(marker) >= chars:
        raise PlanError(
            f"truncate_middle: marker of {len(marker)} chars does not fit a "
            f"{chars}-char budget; raise `budget` or shorten `marker`"
        )
    keep = chars - len(marker)
    head_chars = keep - keep // 2
    tail_chars = keep // 2
    column = _as_column(text)
    from batcher.plan.expr_ir.constructors import when

    shortened = concat(
        column.str.left(head_chars),
        Lit(marker),
        column.str.right(tail_chars),
    )
    return when(column.str.len() <= Lit(chars)).then(column).otherwise(shortened)


def prompt_token_estimate(*parts: IntoExpr, chars_per_token: float = 4.0) -> Expr:
    """The estimated token count of a prompt assembled from several columns.

    Sums the per-part estimates, which is what you want before an assembled prompt exists as a
    column: it costs one pass over the pieces instead of materializing the concatenation. Use it
    to price a run before making it, to route long rows to a larger-window model, or to sort a
    batch by length so a continuous-batching engine packs it well.

    Args:
        parts: The columns or values making up the prompt.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        An Int64 expression: the estimated total tokens.

    Raises:
        PlanError: If no parts are given.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"sys": ["Be brief."], "q": ["Why is the sky blue?"]})
            >>> ds.select(n=bt.prompt_token_estimate(bt.col("sys"), bt.col("q"))).to_pydict()
            {'n': [7]}
    """
    if not parts:
        raise PlanError("prompt_token_estimate: at least one part is required")
    chars_per_token = require_float(
        chars_per_token, func="prompt_token_estimate", arg="chars_per_token"
    )
    if chars_per_token <= 0:
        raise PlanError(
            f"prompt_token_estimate: chars_per_token must be positive, got {chars_per_token}"
        )
    total: Expr | None = None
    for part in parts:
        estimate = _as_column(part).str.estimate_tokens(chars_per_token)
        total = estimate if total is None else total + estimate
    return total.cast("int64")  # type: ignore[union-attr]


def fits_context(
    prompt: IntoExpr,
    window: int,
    *,
    reserve_output: int = 0,
    chars_per_token: float = 4.0,
) -> Expr:
    """True where a prompt leaves room for the reply it is supposed to get.

    The check that catches the failure a context window produces in practice. Overrunning the
    window rarely raises: the serving stack truncates the prompt, or leaves so few tokens that
    the answer stops mid-sentence, and the run finishes looking successful. `reserve_output` is
    what makes this different from a bare length filter — a prompt that fits in 8,192 tokens
    with none to spare cannot be answered at all.

    Args:
        prompt: The assembled prompt column (name or expression).
        window: The model's context window, in tokens.
        reserve_output: Tokens to keep free for the generation.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        A Boolean expression, true where the prompt fits with the reserve left over.

    Raises:
        PlanError: If the reserve is not smaller than the window.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["short", "x" * 400]})
            >>> ds.select(ok=bt.fits_context("p", window=100, reserve_output=50)).to_pydict()
            {'ok': [True, False]}
    """
    window = require_int(window, func="fits_context", arg="window", minimum=1)
    reserve_output = require_int(
        reserve_output, func="fits_context", arg="reserve_output", minimum=0
    )
    if reserve_output >= window:
        raise PlanError(
            f"fits_context: reserve_output ({reserve_output}) must be smaller than "
            f"window ({window}); nothing would be left for the prompt"
        )
    return _as_column(prompt).str.fits_token_budget(window - reserve_output, chars_per_token)
