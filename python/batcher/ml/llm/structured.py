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

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.llm.columns import _loads_lenient
from batcher.ml.llm.extract_schema import FieldSpec, coerce_row, json_schema, resolve_schema

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.ml.llm.engines import Engine, EngineFactory

__all__ = ["json_schema", "llm_classify_udf", "llm_extract_udf"]

_EXTRACT_INSTRUCTION = (
    "Respond with a single JSON object and nothing else. "
    "It must have exactly these keys: {keys}. "
    "Use null for any value you cannot determine."
)


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
    fields: dict[str, FieldSpec],
    prompt_column: str | None,
    template: str | None,
    instruct: bool,
    adapter_column: str | None = None,
    image_column: str | None = None,
    raw_column: str | None = None,
    diagnostics_column: str | None = None,
) -> pa.RecordBatch:
    """One batch through the engine, appending one typed column per declared field.

    With `raw_column` the model's own text is kept beside the typed columns, and with
    `diagnostics_column` each row carries the reasons its values did not fit the schema
    (null for a row that fit), so a parse or validation failure is distinguishable from a
    model that answered null.
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
        requests = _apply_instruction(requests, "\n\n" + _extract_instruction(fields))

    outputs = _dispatch_sorted(engine, requests)
    # Lenient parse: a model told to reply with JSON routinely fences it or wraps it in a
    # sentence, which raw json.loads would reject — nulling every field of the row.
    rows = [coerce_row(_loads_lenient(out), fields) for out in outputs]
    extracted: dict[str, Any] = {
        # The declared type, always — never inferred from what this batch happened to
        # contain. That is what keeps every batch's schema identical.
        name: pa.array([values[name] for values, _ in rows], type=field.arrow_type)
        for name, field in fields.items()
    }
    if raw_column is not None:
        extracted[raw_column] = pa.array(
            [None if out is None else str(out) for out in outputs], type=pa.string()
        )
    if diagnostics_column is not None:
        extracted[diagnostics_column] = pa.array(
            [problems or None for _, problems in rows], type=pa.list_(pa.string())
        )
    # `append_columns` **replaces** a field name the batch already carries; building the
    # batch by hand appended it, and Arrow permits duplicate field names — so extracting
    # into a column you already have produced two of one name that `to_pydict()` and every
    # expression disagree about.
    return append_columns(batch, extracted)


def _extract_instruction(fields: dict[str, FieldSpec]) -> str:
    """The "reply with JSON" instruction, with the nested shape spelled out when there is one.

    A flat schema keeps the original one-line instruction. A nested one also shows the
    shape, because a model told only the top-level keys has no way to know a struct's own.
    """
    text = _EXTRACT_INSTRUCTION.format(keys=", ".join(fields))
    if all(not (f.children or f.item or f.choices) for f in fields.values()):
        return text
    shape = ", ".join(f'"{name}": {_shape(field)}' for name, field in fields.items())
    return f"{text} The object has this shape: {{{shape}}}"


def _shape(field: FieldSpec) -> str:
    """A compact rendering of one field's expected JSON shape, for the instruction."""
    if field.children:
        inner = ", ".join(f'"{name}": {_shape(child)}' for name, child in field.children)
        return f"{{{inner}}}"
    if field.item is not None:
        return f"[{_shape(field.item)}, ...]"
    if field.choices:
        return "one of " + "|".join(field.choices)
    return str(field.arrow_type)


def extract_output_columns(
    schema: dict[str, Any], raw_column: str | None, diagnostics_column: str | None
) -> list[str]:
    """Every column `extract` appends, in order, refusing a name declared twice.

    Raises:
        PlanError: If `raw_column` or `diagnostics_column` repeats a schema field or each
            other, which would make one silently overwrite the other.
    """
    names = [str(n) for n in schema]
    for extra in (raw_column, diagnostics_column):
        if extra is None:
            continue
        if extra in names:
            raise PlanError(
                f"extract(): {extra!r} is already an output column; give raw_column= and "
                "diagnostics_column= names distinct from the schema fields and each other"
            )
        names.append(extra)
    return names


def llm_extract_udf(
    engine_factory: EngineFactory,
    *,
    schema: dict[str, Any],
    prompt_column: str | None = None,
    template: str | None = None,
    instruct: bool = True,
    adapter_column: str | None = None,
    image_column: str | None = None,
    raw_column: str | None = None,
    diagnostics_column: str | None = None,
) -> type:
    """A load-once class UDF appending one **typed** column per `schema` field.

    Args:
        engine_factory: Zero-arg callable returning an `Engine`; called once per worker.
        schema: Output column name to field declaration: a Batcher dtype, a nested
            ``dict`` (struct), a one-element ``list`` (list), a ``set`` of strings (enum),
            or a `pyarrow.DataType`.
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
        raw_column: Optional column keeping the model's unparsed text.
        diagnostics_column: Optional ``list<string>`` column of the row's validation
            failures, null for a row that matched the schema.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus one
        column per declared field.
    """
    fields = resolve_schema(schema)
    extract_output_columns(schema, raw_column, diagnostics_column)

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
                raw_column=raw_column,
                diagnostics_column=diagnostics_column,
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
