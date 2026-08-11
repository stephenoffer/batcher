"""Turning a `RecordBatch` into the per-row requests an engine receives.

Everything between "here is a columnar batch" and "here is a list of requests" lives
here: prompt templating, vision image decoding, the per-row tags (LoRA adapter, sampling
overrides), and the length-sorted dispatch order.

`GenerateSpec` is the single definition of what a generation *is*. Both entry points in
`generate` build one, so neither can grow an option the other lacks — the drift that
comes free with two separately-maintained keyword lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["GenerateSpec"]


@dataclass(frozen=True)
class GenerateSpec:
    """Every columnar choice a generation makes, in one immutable object.

    Args:
        prompt_column: the text column to send (ignored when `template` is set).
        output_column: name of the appended generated column.
        template: a ``str.format`` template over the row's columns.
        image_column: an image column for vision-language models.
        adapter_column: a column naming the per-row LoRA adapter.
        max_tokens_column: a column giving each row its own token budget.
        temperature_column: a column giving each row its own sampling temperature.
        few_shot: fixed ``(input, output)`` demonstration pairs prepended to every prompt.
        parse_json: parse each output as JSON into a struct column (null on error).
        usage: append ``prompt_tokens`` / ``completion_tokens`` columns.
        finish_reason: append a ``finish_reason`` column.
        logprobs: append a ``logprob`` column.
        dedup: send each distinct prompt to the engine only once and copy its result to
            every row that shares it. A throughput win for deterministic decoding over a
            corpus with repeated prompts; leave off when sampling (``temperature > 0``)
            and independent samples for identical prompts are wanted.
    """

    prompt_column: str
    output_column: str = "response"
    template: str | None = None
    image_column: str | None = None
    adapter_column: str | None = None
    max_tokens_column: str | None = None
    temperature_column: str | None = None
    few_shot: tuple[tuple[str, str], ...] | None = None
    parse_json: bool = False
    usage: bool = False
    finish_reason: bool = False
    logprobs: bool = False
    dedup: bool = False

    @property
    def appended_columns(self) -> list[str]:
        """The columns this spec appends to a batch, in the order they are appended."""
        names = [self.output_column]
        if self.usage:
            names += ["prompt_tokens", "completion_tokens"]
        if self.finish_reason:
            names.append("finish_reason")
        if self.logprobs:
            names.append("logprob")
        return names


def _cell(value: object) -> str:
    """One cell as prompt text — a null becomes ``""``, not the literal string ``"None"``.

    ``str(None)`` renders ``"None"``, so a null prompt cell used to inject the four-letter
    word ``None`` straight into the model's context. An empty string is the sane rendering.
    """
    return "" if value is None else str(value)


def _few_shot_prefix(few_shot: tuple[tuple[str, str], ...] | None) -> str:
    """A fixed demonstration block prepended to every prompt, or ``""`` when none.

    Rendered as ``Input: … / Output: …`` pairs — the plain, model-agnostic few-shot shape
    that works on both the completion and chat paths (the whole thing becomes one prompt)."""
    if not few_shot:
        return ""
    blocks = [f"Input: {inp}\nOutput: {out}" for inp, out in few_shot]
    return "\n\n".join(blocks) + "\n\n"


def _render(
    template: str | None,
    column: str,
    batch: pa.RecordBatch,
    few_shot: tuple[tuple[str, str], ...] | None = None,
) -> list[str]:
    """The prompt for each row: ``column`` verbatim, or `template` formatted with the
    row's columns (``"{system} Q: {question}"``-style ``str.format`` placeholders), with
    any `few_shot` demonstrations prepended."""
    prefix = _few_shot_prefix(few_shot)
    if template is None:
        return [prefix + _cell(v) for v in batch.column(column).to_pylist()]
    referenced = _validate_template(template, batch.schema.names)
    if not referenced:
        return [prefix + template] * batch.num_rows  # a constant prompt for every row
    # Materialize ONLY the columns the template names. `batch.to_pylist()` converts every
    # column of every row into Python objects, so a vision generation — whose batch carries an
    # image column precisely because `_decode_image_inputs` needs it — used to decode every
    # image into a Python `bytes` a second time just to format a text template, doubling the
    # batch's peak memory and adding O(rows) work over a column the prompt never mentions.
    cells = {name: batch.column(name).to_pylist() for name in referenced}
    # A null in any referenced column renders as "" rather than "None" (see `_cell`).
    return [
        prefix
        + template.format(
            **{
                name: _cell(values[i]) if values[i] is None else values[i]
                for name, values in cells.items()
            }
        )
        for i in range(batch.num_rows)
    ]


def _validate_template(template: str, names: list[str]) -> set[str]:
    """The batch columns `template` references, raising if it names one the batch lacks.

    Without the check a template referencing a missing column fails deep in the engine with a
    bare ``KeyError`` that names neither the template nor the column — and it fails the
    whole batch. Catching it here turns that into an actionable message at the UDF's edge.
    The referenced set is returned because `_render` needs exactly it to avoid materializing
    the columns the template does not mention.
    """
    import string

    referenced: set[str] = set()
    for _literal, field, _spec, _conv in string.Formatter().parse(template):
        if field:
            # ``{a.b}`` / ``{a[0]}`` reference column ``a``; take that root.
            referenced.add(field.split(".", 1)[0].split("[", 1)[0])
    known = set(names)
    resolved = {f for f in referenced if f and not f.isdigit()}
    missing = sorted(f for f in resolved if f not in known)
    if missing:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"generate template references column(s) not in the data: {missing}. "
            f"Available columns: {sorted(names)}."
        )
    return resolved


#: The request-dict keys carrying a per-row *sampling* override, mapped to the spec field
#: naming the column each comes from. An engine reads these off the request; an engine
#: that does not support them simply ignores the extra keys.
_PER_ROW_SAMPLING = {
    "max_tokens": "max_tokens_column",
    "temperature": "temperature_column",
}


def _build_requests(spec: GenerateSpec | str | None, *rest: object) -> list:
    """Per-row engine requests: plain prompt strings, or ``{prompt, ...}`` dicts.

    A dict is used as soon as any per-row column is set — an image (vision-language), an
    adapter (per-row LoRA), or a sampling override. A null in any of those columns drops
    that key, so the row falls back to the engine's own default: the base model, the
    engine's `max_tokens`, its temperature. That null convention is what lets one column
    override a handful of rows without restating the default for all the others.

    Accepts either a `GenerateSpec` plus the batch, or the legacy positional form
    ``(template, prompt_column, image_column, adapter_column, batch)`` — the seam an
    older test drives — which is turned into a spec here.
    """
    if isinstance(spec, GenerateSpec):
        (batch,) = rest  # type: ignore[assignment]
    else:
        template, prompt_column, image_column, adapter_column, batch = spec, *rest  # type: ignore[assignment]
        spec = GenerateSpec(
            prompt_column=prompt_column,  # type: ignore[arg-type]
            template=template,  # type: ignore[arg-type]
            image_column=image_column,  # type: ignore[arg-type]
            adapter_column=adapter_column,  # type: ignore[arg-type]
        )
    prompts = _render(spec.template, spec.prompt_column, batch, spec.few_shot)
    tags = _per_row_tags(spec, batch)
    if not tags:
        return prompts
    requests: list = []
    for i, prompt in enumerate(prompts):
        request: dict = {"prompt": prompt}
        for key, values in tags.items():
            if values[i] is not None:
                request[key] = values[i]
        requests.append(request)
    return requests


def _per_row_tags(spec: GenerateSpec, batch: pa.RecordBatch) -> dict[str, list]:
    """The per-row request keys this spec asks for, each as a column-length list.

    Empty when the spec names no per-row column at all, which is the signal to send plain
    prompt strings and skip the dict entirely.
    """
    tags: dict[str, list] = {}
    if spec.image_column is not None:
        tags["image"] = _decode_image_inputs(batch.column(spec.image_column))
    if spec.adapter_column is not None:
        tags["adapter"] = batch.column(spec.adapter_column).to_pylist()
    for key, field in _PER_ROW_SAMPLING.items():
        column = getattr(spec, field)
        if column is not None:
            tags[key] = batch.column(column).to_pylist()
    return tags


#: Longest edge, in pixels, a vision request's image is scaled down to. A vision model
#: tiles an image into tokens, so an unbounded one costs context proportional to its
#: area — a 4000x3000 photo can exhaust the window on its own, and the model resizes it
#: internally anyway. Bounding it here also keeps the decoded batch off the worker's heap.
_MAX_IMAGE_EDGE = 1024


def _decode_image_inputs(column: pa.Array) -> list:
    """A list of PIL images for a column of raw image bytes or decoded pixel tensors.

    Bytes → ``PIL.Image.open``; a fixed-shape-tensor ``(H, W, 3)`` → ``Image.fromarray``.
    Null rows yield ``None`` (the model sees a text-only request for that row).

    Every image is **bounded** before it leaves this function (see `_bound_image`): a
    lazily-opened, full-resolution, possibly-palettized image handed straight to a model
    is the shape that blows the context window and the worker's memory at once."""
    import io as _io

    from batcher._internal.optional import require
    from batcher.io.formats.ml.tensor import is_tensor_column

    Image = require("PIL", "Image", feature="Vision LLM input", provides="Pillow", extra="image")

    if is_tensor_column(column):
        if hasattr(column, "combine_chunks"):
            column = column.combine_chunks()
        return [_bound_image(Image.fromarray(row)) for row in column.to_numpy_ndarray()]
    return [
        None if b is None else _bound_image(Image.open(_io.BytesIO(b))) for b in column.to_pylist()
    ]


def _bound_image(image):
    """Materialize, normalize, and size-cap one decoded image.

    Three things, each of which is a real failure at scale. `Image.open` is **lazy**, so
    without `load` the file handle and the decode both survive into the engine call and
    the batch's memory is unbounded and unpredictable. A palette or RGBA image reaches a
    model expecting three channels and either errors or silently loses the alpha. And a
    full-resolution image costs vision tokens proportional to its area. `thumbnail` only
    ever shrinks, so a small image passes through untouched.
    """
    image.load()
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > _MAX_IMAGE_EDGE:
        image.thumbnail((_MAX_IMAGE_EDGE, _MAX_IMAGE_EDGE))
    return image


def _request_length(request: object) -> int:
    """The prompt length of a request, for length bucketing. A dict request carries its
    text under ``"prompt"``; an image contributes no characters, so a vision batch
    buckets on its text and its rows stay interchangeable."""
    text = request.get("prompt", "") if isinstance(request, dict) else request
    return len(str(text))


def _length_sorted_order(requests: list) -> list[int]:
    """Indices of `requests` ordered longest prompt first — the offline-vLLM throughput
    lever nothing here was pulling.

    A batch mixing a 4-token prompt with a 4000-token one pads every sequence in the
    step to the longest, so most of the tensor is padding, and neighbouring rows share
    no prefix so the prefix cache misses. Grouping similar lengths together fixes both.
    Longest-first additionally lets the scheduler discover its true memory ceiling on
    the first step rather than OOM-ing part way through.

    The sort is stable, so rows of equal length keep their input order and the dispatch
    stays deterministic. `_restore_order` puts the results back.
    """
    return sorted(range(len(requests)), key=lambda i: -_request_length(requests[i]))


def _restore_order(values: list, order: list[int]) -> list:
    """Invert the `_length_sorted_order` permutation, so results match the caller's rows.

    This is the half that makes bucketing invisible. Getting it wrong does not raise —
    it silently pairs every generation with the wrong row.
    """
    out: list = [None] * len(order)
    for position, original in enumerate(order):
        out[original] = values[position]
    return out
