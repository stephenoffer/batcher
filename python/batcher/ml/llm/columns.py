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
    """Parse a generation as JSON, tolerating a fenced or prose-wrapped object → null."""
    return _loads_lenient(text)


def _loads_lenient(text: object) -> object | None:
    """Parse JSON an instruction-tuned model produced, or `None` if there is none.

    A model told to "reply with JSON" routinely wraps it in a ```json fence or a sentence
    ("Here is the JSON: {...}"). Raw `json.loads` rejects every such row, silently nulling
    it after the generation is already paid for. This tries the string as-is, then the
    contents of a Markdown code fence, then the first balanced ``{...}``/``[...]`` span, so
    the common wrappers parse while genuinely non-JSON output still falls through to null.
    """
    if not isinstance(text, str):
        return None
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


def _json_candidates(text: str):
    """Yield the substrings of `text` worth attempting to parse, most-literal first."""
    stripped = text.strip()
    yield stripped
    fenced = _strip_code_fence(stripped)
    if fenced != stripped:
        yield fenced
    span = _first_json_span(fenced)
    if span is not None and span != fenced:
        yield span


def _strip_code_fence(text: str) -> str:
    """The contents of the first Markdown code fence in `text`, or `text` unchanged.

    Handles ```` ```json ... ``` ```` and a bare ```` ``` ... ``` ````; a fence the model
    opened but never closed still yields its body so a truncated response can parse.
    """
    import re

    match = re.search(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)(?:```|$)", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _first_json_span(text: str) -> str | None:
    """The first balanced ``{...}`` or ``[...]`` in `text`, honoring strings, or `None`.

    A brace counter that skips over braces inside string literals (and their escapes), so
    a value like ``{"note": "a } brace"}`` is spanned correctly rather than cut short.
    """
    start = _first_of(text, "{[")
    if start is None:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _first_of(text: str, chars: str) -> int | None:
    """The index of the earliest of `chars` in `text`, or `None` if none appear."""
    positions = [text.index(c) for c in chars if c in text]
    return min(positions) if positions else None


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
