"""Typed columns out of an LLM — the AI-powered-ETL primitives.

`llm_generate` gives you a *string*. That string is not a column an analyst can filter,
join, or aggregate; turning it into one is the whole job of an AI-powered ETL step, and
it is where these pipelines break.

Two failure modes this module exists to remove:

* **Schema drift.** `parse_json=True` infers the struct type from whatever the model
  happened to emit *in that batch*. Ask for ``{label, score}`` and the model omits
  ``score`` on one batch, and the two batches carry incompatible struct types — the scan
  fails at concat time, after the GPU work is paid for. `extract` takes a **declared**
  schema, so every batch produces the same Arrow types no matter what the model says.
* **Unconstrained labels.** A classifier that answers ``"Positive."`` where you expected
  ``"positive"`` yields a category column with a long tail of near-duplicates. `classify`
  matches the output against the declared label set and nulls anything else, so the
  column has exactly the domain you asked for and bad rows are countable.

Both degrade per row, never per batch: an unparseable output or an off-menu label becomes
a null, so one bad generation cannot abort a scan over millions of rows.

Pair `extract` with ``vllm_engine(guided_json=json_schema(schema))`` — guided decoding
makes the output well-formed, and the declared schema makes it *typed*.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.llm.columns import _loads_lenient
from batcher.plan.types import CAST_DTYPES, DTYPE_REGISTRY

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.ml.llm.engines import Engine, EngineFactory

__all__ = ["json_schema", "llm_classify_udf", "llm_extract_udf"]

# The JSON Schema type each Batcher dtype maps to, for guided decoding.
_JSON_TYPES: dict[str, str] = {
    "int64": "integer",
    "int32": "integer",
    "float64": "number",
    "float32": "number",
    "bool": "boolean",
    "string": "string",
}

_EXTRACT_INSTRUCTION = (
    "Respond with a single JSON object and nothing else. "
    "It must have exactly these keys: {keys}. "
    "Use null for any value you cannot determine."
)


def _resolve_schema(schema: dict[str, str]) -> dict[str, pa.DataType]:
    """Validate the declared dtypes and resolve them to Arrow types."""
    if not schema:
        raise PlanError("extract(): schema must declare at least one field")
    unknown = {d for d in schema.values() if d not in CAST_DTYPES}
    if unknown:
        raise PlanError(
            f"extract(): unknown dtype(s) {sorted(unknown)}; use one of {sorted(CAST_DTYPES)}"
        )
    # A dtype `_coerce` cannot produce is rejected here rather than yielding an all-null
    # column, which is the worst possible outcome: the generation is already paid for, the
    # schema looks right, and nothing anywhere says the field was never extracted.
    uncoercible = sorted({d for d in schema.values() if not _is_extractable(DTYPE_REGISTRY[d])})
    if uncoercible:
        raise PlanError(
            f"extract(): dtype(s) {uncoercible} cannot be extracted from a model's JSON "
            "output. Declare a string/number/bool/date/time/timestamp/binary field instead, "
            "and cast it afterwards if you need another type."
        )
    return {name: DTYPE_REGISTRY[dtype] for name, dtype in schema.items()}


def _is_extractable(arrow_type: pa.DataType) -> bool:
    """Whether `_coerce` can produce a value of `arrow_type` from parsed JSON."""
    import pyarrow as pa

    return bool(
        pa.types.is_boolean(arrow_type)
        or pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_date(arrow_type)
        or pa.types.is_time(arrow_type)
        or pa.types.is_timestamp(arrow_type)
    )


def json_schema(schema: dict[str, str]) -> dict:
    """A JSON Schema for `schema`, to hand to ``vllm_engine(guided_json=...)``.

    Guided decoding constrains the model to emit exactly this shape, so every row parses.
    `extract` still types the result independently — this only removes the parse failures.

    Args:
        schema: Column name → Batcher dtype (``"string"``, ``"int64"``, ``"float64"``,
            ``"bool"``, …), the same mapping `extract` takes.

    Returns:
        A JSON Schema ``object`` with one property per field.

    Raises:
        PlanError: If a dtype is unknown or has no JSON Schema analogue.

    Examples:
        .. doctest::

            >>> from batcher.ml import json_schema
            >>> json_schema({"sentiment": "string", "score": "float64"})["properties"]
            {'sentiment': {'type': 'string'}, 'score': {'type': 'number'}}
    """
    _resolve_schema(schema)
    properties = {}
    for name, dtype in schema.items():
        canonical = DTYPE_REGISTRY[dtype]
        json_type = next(
            (j for d, j in _JSON_TYPES.items() if DTYPE_REGISTRY[d] == canonical), None
        )
        if json_type is None:
            raise PlanError(f"json_schema(): dtype {dtype!r} has no JSON Schema equivalent")
        properties[name] = {"type": json_type}
    return {"type": "object", "properties": properties, "required": list(schema)}


def _coerce(value: object, arrow_type: pa.DataType) -> object | None:
    """Coerce one parsed JSON value to `arrow_type`, or null if it cannot be.

    A model that returns ``"42"`` for an integer field, or ``"yes"`` for a boolean, is
    doing what models do; a per-value coercion recovers the row instead of losing it.
    """
    import pyarrow as pa

    if value is None:
        return None
    try:
        if pa.types.is_boolean(arrow_type):
            if isinstance(value, bool):
                return value
            return {"true": True, "yes": True, "false": False, "no": False}.get(
                str(value).strip().lower()
            )
        if pa.types.is_integer(arrow_type):
            return _coerce_integer(value)
        if pa.types.is_floating(arrow_type):
            if isinstance(value, bool):
                return None
            # `pa.array([1.5], type=pa.float16())` rejects a Python float outright with
            # "Expected np.float16 instance", so a half-precision field did not degrade to
            # null like every other mismatch here — it raised and failed the whole batch.
            if pa.types.is_float16(arrow_type):
                import numpy as np

                return np.float16(value)
            return float(value)
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return value if isinstance(value, str) else json.dumps(value)
        if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return value if isinstance(value, bytes) else str(value).encode()
        if pa.types.is_temporal(arrow_type):
            return _coerce_temporal(value, arrow_type)
    except (TypeError, ValueError):
        return None
    return None


def _coerce_temporal(value: object, arrow_type: pa.DataType) -> object | None:
    """Coerce one parsed JSON value to a date / time / timestamp, or null.

    A model asked for a date answers with an ISO string, because that is what dates look
    like in the text it was trained on — never with a `datetime` object, which JSON cannot
    carry anyway. Without this every temporal field of an extraction came back null for
    every row, after the generation had already been paid for.
    """
    import datetime as dt

    import pyarrow as pa

    if isinstance(value, (dt.date, dt.time)):  # dt.datetime is a dt.date
        parsed: object = value
    elif isinstance(value, str):
        parsed = _parse_iso(value.strip(), arrow_type)
    else:
        return None
    if parsed is None:
        return None
    # A `date` where a timestamp is declared (and the reverse) is the ordinary mismatch: the
    # model answered "2024-01-05" for a field the schema types as a timestamp. Normalize
    # rather than null, since the value the model gave is unambiguous.
    if pa.types.is_date(arrow_type) and isinstance(parsed, dt.datetime):
        return parsed.date()
    if pa.types.is_timestamp(arrow_type) and not isinstance(parsed, dt.datetime):
        if isinstance(parsed, dt.date):
            return dt.datetime(parsed.year, parsed.month, parsed.day)
        return None
    if pa.types.is_time(arrow_type) and not isinstance(parsed, dt.time):
        return parsed.time() if isinstance(parsed, dt.datetime) else None
    return parsed


def _parse_iso(text: str, arrow_type: pa.DataType) -> object | None:
    """An ISO-8601 date, time, or datetime out of `text`, or `None` when it is neither.

    ``fromisoformat`` accepts a trailing ``Z`` from Python 3.11 on, which is the spelling a
    model emits most often, so no pre-processing is needed beyond the strip the caller did.
    """
    import datetime as dt

    import pyarrow as pa

    if pa.types.is_time(arrow_type):
        parsers = (dt.time.fromisoformat, dt.datetime.fromisoformat)
    else:
        parsers = (dt.datetime.fromisoformat, dt.date.fromisoformat)
    for parse in parsers:
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def _coerce_integer(value: object) -> int | None:
    """Coerce `value` to an int, or null when the narrowing would be **lossy**.

    ``int(float("3.9"))`` is 3, which is indistinguishable from a model that genuinely
    answered 3 — a wrong number in a column that looks healthy. Only an exactly integral
    value converts; anything else degrades to null like any other uncoercible value, so
    the failures stay countable. An `int` is taken directly rather than through `float`,
    which would silently round past 2**53.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)  # exact, and keeps precision beyond 2**53
        except ValueError:
            pass  # not an exact integer -> fall through to the float parse below
        value = float(text)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    return None


def _apply_instruction(requests: list, suffix: str) -> list:
    """Append `suffix` to the text of every request, string or dict.

    A dict request (a vision or per-row-LoRA row) carries its text under ``"prompt"``.
    Skipping those dropped the "reply with JSON" / "answer with one label" instruction
    for exactly the rows that need it most, and the model was never told what to emit.
    """
    out = []
    for request in requests:
        if isinstance(request, dict):
            out.append({**request, "prompt": f"{request['prompt']}{suffix}"})
        else:
            out.append(f"{request}{suffix}")
    return out


def _dispatch_sorted(engine: Engine, requests: list) -> list:
    """Run `requests` through `engine` longest-prompt-first, returning outputs in row order.

    The same padding / prefix-cache throughput lever `generate` pulls (see
    `requests._length_sorted_order`): a batch that mixes a 4-token prompt with a 4000-token
    one otherwise pads every sequence to the longest. The results are un-permuted, so the
    extracted columns line up with the caller's rows exactly.
    """
    from batcher.ml.llm.requests import _length_sorted_order, _restore_order

    order = _length_sorted_order(requests)
    generated = list(engine([requests[i] for i in order]))
    if len(generated) != len(order):
        from batcher._internal.errors import BackendError

        raise BackendError(
            f"{type(engine).__name__} returned {len(generated)} outputs for {len(order)} "
            "requests; an engine must return exactly one string per request."
        )
    return _restore_order(generated, order)


def _extract_batch(
    engine: Engine,
    batch: pa.RecordBatch,
    *,
    fields: dict[str, pa.DataType],
    prompt_column: str | None,
    template: str | None,
    instruct: bool,
    adapter_column: str | None = None,
    image_column: str | None = None,
) -> pa.RecordBatch:
    """One batch through the engine, appending one typed column per declared field."""
    import pyarrow as pa

    from batcher.ml.llm.requests import GenerateSpec, _build_requests
    from batcher.ml.tabular.features import append_columns

    spec = GenerateSpec(
        prompt_column=prompt_column or "",
        template=template,
        adapter_column=adapter_column,
        image_column=image_column,
    )
    requests = _build_requests(spec, batch)
    if instruct:
        suffix = "\n\n" + _EXTRACT_INSTRUCTION.format(keys=", ".join(fields))
        requests = _apply_instruction(requests, suffix)

    parsed: list[dict] = []
    for out in _dispatch_sorted(engine, requests):
        # Lenient parse: a model told to reply with JSON routinely fences it or wraps it
        # in a sentence, which raw json.loads would reject — nulling every field of the row.
        obj = _loads_lenient(out)
        parsed.append(obj if isinstance(obj, dict) else {})

    extracted = {
        # The declared type, always — never inferred from what this batch happened to
        # contain. That is what keeps every batch's schema identical.
        name: pa.array([_coerce(row.get(name), arrow_type) for row in parsed], type=arrow_type)
        for name, arrow_type in fields.items()
    }
    # `append_columns` **replaces** a field name the batch already carries; building the
    # batch by hand appended it, and Arrow permits duplicate field names — so extracting
    # into a column you already have produced two of one name that `to_pydict()` and every
    # expression disagree about.
    return append_columns(batch, extracted)


def llm_extract_udf(
    engine_factory: EngineFactory,
    *,
    schema: dict[str, str],
    prompt_column: str | None = None,
    template: str | None = None,
    instruct: bool = True,
    adapter_column: str | None = None,
    image_column: str | None = None,
) -> type:
    """A load-once class UDF appending one **typed** column per `schema` field.

    Args:
        engine_factory: Zero-arg callable returning an `Engine`; called once per worker.
        schema: Output column name → Batcher dtype.
        prompt_column: The text column to send (ignored when `template` is set).
        template: A ``str.format`` template over the row's columns.
        instruct: Append a "reply with JSON having exactly these keys" instruction to
            each prompt. Turn it off when the engine already constrains decoding
            (``guided_json``) or the template says it itself.
        adapter_column: Optional column naming the **LoRA adapter** to use per row, so
            one engine serves many fine-tuned extractors. Pair with
            ``vllm_engine(lora_paths={name: path})``; a null uses the base model.
        image_column: Optional image column (raw bytes or an ``(H, W, 3)`` tensor) for a
            **vision** model, so fields can be extracted from an image (an invoice photo
            → ``{vendor, total}``). The engine must be vision-capable.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus one
        column per declared field.
    """
    fields = _resolve_schema(schema)

    class _LlmExtract:
        """Holds one engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return _extract_batch(
                self._engine,
                batch,
                fields=fields,
                prompt_column=prompt_column,
                template=template,
                instruct=instruct,
                adapter_column=adapter_column,
                image_column=image_column,
            )

    return _LlmExtract


_CLASSIFY_INSTRUCTION = "Answer with exactly one of these labels and nothing else: {labels}"


def _match_label(output: str, lookup: dict[str, str]) -> str | None:
    """Resolve a model's answer to a declared label, or null.

    Tolerates the two things a model reliably does to a label — changes its case, and
    wraps it in punctuation or a sentence — while refusing to guess at anything else.
    """
    if not isinstance(output, str):
        return None
    text = output.strip().strip(".\"'` \n").lower()
    if text in lookup:
        return lookup[text]
    # The label may sit inside a short sentence ("The sentiment is positive."). When the
    # declared labels nest ("positive" inside "very positive"), a correct answer matches
    # both keys; a plain uniqueness test called that ambiguous and nulled the row. The
    # longest matching key is the specific one the model actually said, so prefer it —
    # and only fall back to null when two *equally long* labels both appear, which is
    # genuine ambiguity ("could be positive, could be negative").
    hits = [key for key in lookup if key in text]
    if not hits:
        return None
    longest = max(len(key) for key in hits)
    finalists = {lookup[key] for key in hits if len(key) == longest}
    return finalists.pop() if len(finalists) == 1 else None


def llm_classify_udf(
    engine_factory: EngineFactory,
    *,
    labels: list[str],
    prompt_column: str | None = None,
    output_column: str = "label",
    template: str | None = None,
    instruct: bool = True,
    adapter_column: str | None = None,
    image_column: str | None = None,
) -> type:
    """A load-once class UDF appending a label column constrained to `labels`.

    Any output that does not resolve to exactly one declared label becomes null, so the
    column's domain is exactly `labels` and the failures are countable
    (``ds.filter(col("label").is_null()).count()``).

    Args:
        engine_factory: Zero-arg callable returning an `Engine`; called once per worker.
        labels: The permitted labels. Must be non-empty and case-insensitively distinct.
        prompt_column: The text column to classify (ignored when `template` is set).
        output_column: Name of the appended label column.
        template: A ``str.format`` template over the row's columns.
        instruct: Append the "answer with one of these labels" instruction to each prompt.
        adapter_column: Optional column naming the **LoRA adapter** to use per row, so
            one engine serves many fine-tuned classifiers. Pair with
            ``vllm_engine(lora_paths={name: path})``; a null uses the base model.
        image_column: Optional image column for a **vision** model, so a row can be
            classified from an image rather than text. The engine must be vision-capable.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the label.

    Raises:
        PlanError: If `labels` is empty or contains case-insensitive duplicates.
    """
    if not labels:
        raise PlanError("classify(): labels must be non-empty")
    lookup = {label.strip().lower(): label for label in labels}
    if len(lookup) != len(labels):
        raise PlanError(f"classify(): labels must be distinct ignoring case, got {labels}")

    class _LlmClassify:
        """Holds one engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return _classify_batch(
                self._engine,
                batch,
                labels=labels,
                lookup=lookup,
                prompt_column=prompt_column,
                output_column=output_column,
                template=template,
                instruct=instruct,
                adapter_column=adapter_column,
                image_column=image_column,
            )

    return _LlmClassify


def _classify_batch(
    engine: Engine,
    batch: pa.RecordBatch,
    *,
    labels: list[str],
    lookup: dict[str, str],
    prompt_column: str | None,
    output_column: str,
    template: str | None,
    instruct: bool,
    adapter_column: str | None = None,
    image_column: str | None = None,
) -> pa.RecordBatch:
    """One batch through the engine, appending the label column resolved against `lookup`.

    A module-level function rather than a closure body so the request construction —
    including the instruction suffix that dict requests used to lose — is reachable from
    a test without standing up the whole UDF.
    """
    import pyarrow as pa

    from batcher.ml.llm.requests import GenerateSpec, _build_requests
    from batcher.ml.tabular.features import append_columns

    spec = GenerateSpec(
        prompt_column=prompt_column or "",
        template=template,
        adapter_column=adapter_column,
        image_column=image_column,
    )
    requests = _build_requests(spec, batch)
    if instruct:
        suffix = "\n\n" + _CLASSIFY_INSTRUCTION.format(labels=", ".join(labels))
        requests = _apply_instruction(requests, suffix)
    resolved = [_match_label(o, lookup) for o in _dispatch_sorted(engine, requests)]
    # Replaces rather than appends when the batch already has `output_column` — see
    # `_extract_batch` for why a duplicate Arrow field name is the silent outcome.
    return append_columns(batch, {output_column: pa.array(resolved, type=pa.string())})
