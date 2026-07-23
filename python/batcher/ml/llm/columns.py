"""Building the columns a generation appends, from what the engine reported.

The generated text is the easy half. The other three columns — token usage, finish
reason, cumulative logprob — arrive through the `channels` side channels in *dispatch*
order, which is not row order, so each is un-permuted here with the same permutation the
outputs are. Getting that wrong never raises: every row keeps a plausible-looking number
that belongs to a different row.
"""

from __future__ import annotations

import json

from batcher.ml.llm.requests import _restore_order

__all__: list[str] = []


def _safe_json(text: str) -> object | None:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _aligned(reported: list | None, n: int, order: list | None) -> list:
    """`reported` put back into row order, or `n` nulls when it is missing or misshapen.

    A reported list whose length does not match the outputs cannot be attributed to rows
    at all, so it becomes nulls rather than a silent misalignment.
    """
    values = list(reported) if reported is not None else [None] * n
    if order is not None and len(values) == n:
        values = _restore_order(values, order)
    return values if len(values) == n else [None] * n


def _finish_reason_column(reported: list | None, n: int, order: list | None):
    """The per-row `finish_reason` string column, or all-null when the engine reports none.

    ``"length"`` means the model was cut off at `max_tokens` rather than finishing. Without
    this column that truncation is undetectable: the row holds a plausible prefix, and a
    downstream `parse_json` on a JSON object that stops mid-object silently yields null.
    Filter on it (``col("finish_reason") == "length"``) to count and re-run those rows.
    """
    import pyarrow as pa

    return pa.array(_aligned(reported, n, order), type=pa.string())


def _logprob_column(reported: list | None, n: int, order: list | None):
    """The per-row cumulative log-probability column, or all-null when none is reported.

    The model's confidence in its own output, summed over the generated tokens. It is the
    cheapest quality signal a bulk generation produces: sort by it to find the rows a
    model was least sure about, and route those to review or to a larger model instead of
    trusting a million generations uniformly.
    """
    import pyarrow as pa

    return pa.array(_aligned(reported, n, order), type=pa.float64())


def _usage_columns(engine: object, n: int, reported: list | None = None, order: list | None = None):
    """Per-row `(prompt_tokens, completion_tokens)` Int64 arrays, or all-null when the
    engine reports no usage.

    Two channels, in priority order. `reported` is what the engine pushed into this
    call's `usage_sink` — per call, thread-local, and impossible to misattribute. Falling
    back to the `engine.last_usage` attribute keeps the documented contract working for
    a user-written engine, at the cost of being a shared mutable read.

    `order` is the length-sort permutation the requests were dispatched under; the pairs
    come back in *dispatch* order and are un-permuted here. Without that every token
    count lands on the wrong row — plausible-looking and silent.
    """
    import pyarrow as pa

    if reported is None:
        reported = getattr(engine, "last_usage", None)
    pairs = list(reported) if reported is not None else [None] * n
    if order is not None and len(pairs) == n:
        pairs = _restore_order(pairs, order)
    if len(pairs) != n:
        # Otherwise this surfaces far from its cause, as an opaque "arrays must all be
        # the same length" from `RecordBatch.from_arrays`, with nothing naming the
        # engine. A mismatch means the engine reported usage for a different number of
        # requests than it returned outputs for, which would silently misalign every
        # token count against its row.
        from batcher._internal.errors import BackendError

        msg = (
            f"{type(engine).__name__}.last_usage reported {len(pairs)} usage pairs for "
            f"{n} generated outputs; they must correspond one-to-one and in prompt "
            "order. Pass usage=False to skip token accounting for this engine."
        )
        raise BackendError(msg)
    prompt = [p[0] if p else None for p in pairs]
    completion = [p[1] if p else None for p in pairs]
    return pa.array(prompt, type=pa.int64()), pa.array(completion, type=pa.int64())
