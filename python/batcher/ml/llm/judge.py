"""Model-graded evaluation — scoring generations with a judge model, as typed columns.

The lexical metrics in `plan.functions.metrics.text` compare surface forms. They cannot tell a
correct paraphrase from a wrong answer, and for anything open-ended that is most of what you
need to know. The usual answer is to ask a stronger model, and the usual implementation is a
Python loop over examples with a hand-rolled parser for whatever the judge wrote back.

This is that pattern as a batch UDF over the same `Engine` contract everything else here uses,
so a judged eval over a million rows is one scan with the judge's answers already parsed into a
column you can filter and aggregate.

Three shapes, because they fail differently:

* `llm_score_udf` grades one output against a rubric on a numeric scale. The score is parsed
  and **range-checked**, so a judge that answers "8/10" or "high" yields null rather than
  poisoning a mean.
* `llm_pairwise_udf` compares two outputs and picks a winner. It is the shape to prefer when
  comparing two systems: a judge is far more consistent choosing between two answers than
  assigning either an absolute number, and the position bias that introduces is measurable
  (see `swap` below) where an absolute score's bias is not.
* `llm_verify_udf` asks a yes/no question about one row — is this grounded in the context, does
  it follow the instruction, is it safe. A boolean column is the one a data-quality gate wants.

Each appends one column and passes the rest through. To *filter or aggregate* on that column,
declare it through `map_batches(output_columns=[...])` — the plan above the stage cannot see
inside a UDF, so an undeclared column exists in the data and not in the schema.

**A judge is a model, so it is wrong sometimes, and its errors correlate with the thing being
judged.** It prefers longer answers, answers that look like its own, and whichever option came
first. Calibrate against human labels on a sample before trusting a number, and read a judged
score as a comparison between runs rather than as ground truth.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.ml.llm.engines import EngineFactory

__all__ = ["llm_pairwise_udf", "llm_score_udf", "llm_verify_udf"]

_SCORE_INSTRUCTION = (
    "Respond with a single number between {low} and {high} and nothing else. "
    "Do not explain, do not add units, do not use a fraction."
)

_PAIRWISE_INSTRUCTION = (
    "Respond with exactly one of A, B, or TIE and nothing else. "
    "Answer A if the first response is better, B if the second is better, "
    "TIE if they are equally good."
)

_VERIFY_INSTRUCTION = "Respond with exactly YES or NO and nothing else."

# A leading number, which is what the instruction asks for. Anchored at the start so a judge
# that ignores the instruction and writes prose is a null rather than a number pulled out of
# the middle of a sentence — "I would not give this more than a 2" must not read as 2.
_LEADING_NUMBER = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?")

_VERDICT = re.compile(r"^\s*(A|B|TIE)\b", re.IGNORECASE)
_YES_NO = re.compile(r"^\s*(YES|NO)\b", re.IGNORECASE)


def _parse_score(output: str, low: float, high: float) -> float | None:
    """The judge's number, or None when it did not answer with one inside the scale.

    Out-of-range is a null rather than a clamp: a judge answering 8 on a 1-5 scale has not
    understood the rubric, and clamping it to 5 would silently record a strong positive.
    """
    match = _LEADING_NUMBER.match(output or "")
    if match is None:
        return None
    value = float(match.group(0))
    return value if low <= value <= high else None


def _parse_verdict(output: str) -> str | None:
    """`"A"`, `"B"`, `"TIE"`, or None when the judge answered with something else."""
    match = _VERDICT.match(output or "")
    return match.group(1).upper() if match else None


def _parse_yes_no(output: str) -> bool | None:
    """True/False, or None when the judge answered with neither."""
    match = _YES_NO.match(output or "")
    return match.group(1).upper() == "YES" if match else None


def _swap_verdict(verdict: str | None) -> str | None:
    """The verdict as seen from the other presentation order."""
    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict


def _render(
    template: str, batch: pa.RecordBatch, extra: dict[str, list] | None = None
) -> list[str]:
    """One prompt per row, filling `{column}` slots from the batch.

    Uses `str.format_map` over a per-row view rather than `format`, so a template mentioning a
    column the batch does not have raises a clear `PlanError` naming it instead of a `KeyError`
    from inside the formatter.
    """
    columns = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
    if extra:
        columns.update(extra)
    names = set(re.findall(r"\{(\w+)\}", template))
    missing = sorted(names - set(columns))
    if missing:
        raise PlanError(
            f"judge template references column(s) not in the batch: {missing}; "
            f"available: {sorted(columns)}"
        )
    return [
        template.format_map({k: _cell(v[i]) for k, v in columns.items()})
        for i in range(batch.num_rows)
    ]


def _cell(value: Any) -> str:
    """A cell rendered for a prompt; a null reads as empty rather than the string 'None'."""
    return "" if value is None else str(value)


def _append(batch: pa.RecordBatch, name: str, values: list, dtype: Any) -> pa.RecordBatch:
    """The batch with one column appended, keeping every input column."""
    import pyarrow as pa

    arrays = [batch.column(i) for i in range(batch.num_columns)]
    arrays.append(pa.array(values, type=dtype))
    return pa.RecordBatch.from_arrays(arrays, names=[*batch.schema.names, name])


def _validate_scale(low: float, high: float) -> None:
    """Reject a scale that cannot express an ordering."""
    if not high > low:
        raise PlanError(f"llm_score_udf: high ({high}) must be greater than low ({low})")


def llm_score_udf(
    engine_factory: EngineFactory,
    *,
    template: str,
    output_column: str = "score",
    low: float = 1.0,
    high: float = 5.0,
    instruct: bool = True,
) -> type:
    """A load-once class UDF appending a judge model's numeric score for each row.

    The rubric lives in `template`, which is filled per row from the batch's columns. The
    judge's answer is parsed as a leading number and range-checked against ``[low, high]``;
    anything else becomes null, so a judge that wrote prose or answered off-scale is countable
    (``ds.filter(col("score").is_null()).count()``) rather than silently averaged in.

    Out-of-range answers are nulled rather than clamped on purpose. A judge answering 8 on a
    1-5 scale has not understood the rubric, and recording that as a 5 would turn a
    misunderstanding into a strong positive.

    Args:
        engine_factory: Zero-arg callable returning an `Engine`; called once per worker.
        template: The rubric prompt, with ``{column}`` slots filled from the row.
        output_column: Name of the appended score column.
        low: The lowest valid score.
        high: The highest valid score.
        instruct: Append the "answer with a single number" instruction to each prompt.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the score.

    Raises:
        PlanError: If `high` is not greater than `low`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import llm_score_udf
            >>> judge = lambda: (lambda prompts: ["4"] * len(prompts))
            >>> udf = llm_score_udf(judge, template="Rate 1-5: {answer}")
            >>> ds = bt.from_pydict({"answer": ["a fine answer"]})
            >>> ds.ml.map_batches(udf).to_pydict()["score"]
            [4.0]
    """
    _validate_scale(low, high)
    suffix = "\n\n" + _SCORE_INSTRUCTION.format(low=_number(low), high=_number(high))

    class _LlmScore:
        """Holds one judge engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            prompts = _render(template, batch)
            if instruct:
                prompts = [p + suffix for p in prompts]
            outputs = list(self._engine(prompts))
            scores = [_parse_score(o, low, high) for o in outputs]
            return _append(batch, output_column, scores, pa.float64())

    return _LlmScore


def _number(value: float) -> str:
    """A scale bound rendered for the instruction, without a pointless trailing ``.0``."""
    return str(int(value)) if float(value).is_integer() else str(value)


def llm_pairwise_udf(
    engine_factory: EngineFactory,
    *,
    template: str,
    a_column: str,
    b_column: str,
    output_column: str = "winner",
    swap: bool = True,
    instruct: bool = True,
) -> type:
    """A load-once class UDF appending which of two responses a judge preferred.

    The column holds ``"A"``, ``"B"``, ``"TIE"``, or null when the judge answered with
    something else. Prefer this to an absolute score when the question is which of two systems
    is better: judges are markedly more consistent choosing between two answers than assigning
    either one a number.

    `swap` is what makes the result trustworthy. Judges have a strong, well-documented
    preference for whichever response is presented first, so each row is judged twice with the
    order reversed. A row where the two runs disagree is a row where position decided the
    outcome, and it is recorded as ``"TIE"`` rather than as whichever answer won the coin flip.
    It doubles the judging cost, which is the price of a number that is not mostly position
    bias; turn it off only when you have measured that bias yourself.

    The template must mention both response columns by name.

    Args:
        engine_factory: Zero-arg callable returning an `Engine`; called once per worker.
        template: The comparison prompt, with ``{column}`` slots filled from the row.
        a_column: The column holding the first response.
        b_column: The column holding the second response.
        output_column: Name of the appended verdict column.
        swap: Judge each row in both orders and record a disagreement as a tie.
        instruct: Append the "answer with A, B, or TIE" instruction to each prompt.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the verdict.

    Raises:
        PlanError: If `a_column` and `b_column` are the same column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import llm_pairwise_udf
            >>> judge = lambda: (lambda prompts: ["A"] * len(prompts))
            >>> udf = llm_pairwise_udf(
            ...     judge,
            ...     template="Which is better?\\nFirst: {left}\\nSecond: {right}",
            ...     a_column="left",
            ...     b_column="right",
            ... )
            >>> ds = bt.from_pydict({"left": ["one"], "right": ["two"]})
            >>> ds.ml.map_batches(udf).to_pydict()["winner"]
            ['TIE']
    """
    if a_column == b_column:
        raise PlanError(
            f"llm_pairwise_udf: a_column and b_column must differ, both are {a_column!r}"
        )
    suffix = "\n\n" + _PAIRWISE_INSTRUCTION

    class _LlmPairwise:
        """Holds one judge engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            forward = _render(template, batch)
            if instruct:
                forward = [p + suffix for p in forward]
            verdicts = [_parse_verdict(o) for o in self._engine(forward)]
            if swap:
                # The same rows with the two responses exchanged. A judge that answers "A"
                # both times preferred the *position*, not the response.
                swapped_columns = {
                    a_column: batch.column(batch.schema.get_field_index(b_column)).to_pylist(),
                    b_column: batch.column(batch.schema.get_field_index(a_column)).to_pylist(),
                }
                reversed_prompts = _render(template, batch, swapped_columns)
                if instruct:
                    reversed_prompts = [p + suffix for p in reversed_prompts]
                mirrored = [
                    _swap_verdict(_parse_verdict(o)) for o in self._engine(reversed_prompts)
                ]
                verdicts = [f if f == m else "TIE" for f, m in zip(verdicts, mirrored, strict=True)]
            return _append(batch, output_column, verdicts, pa.string())

    return _LlmPairwise


def llm_verify_udf(
    engine_factory: EngineFactory,
    *,
    template: str,
    output_column: str = "passed",
    instruct: bool = True,
) -> type:
    """A load-once class UDF appending a judge model's yes/no verdict for each row.

    The boolean column a data-quality gate wants: is this answer grounded in its context, does
    it follow the instruction, is it safe to ship. Anything the judge answers that is not a
    leading YES or NO becomes null, so an unusable verdict is countable rather than quietly
    counting as a failure — which matters, because a null read as False would make a confused
    judge look like a failing dataset.

    Args:
        engine_factory: Zero-arg callable returning an `Engine`; called once per worker.
        template: The question prompt, with ``{column}`` slots filled from the row.
        output_column: Name of the appended boolean column.
        instruct: Append the "answer YES or NO" instruction to each prompt.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the verdict.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import llm_verify_udf
            >>> judge = lambda: (lambda prompts: ["YES"] * len(prompts))
            >>> udf = llm_verify_udf(judge, template="Is {answer} grounded in {context}?")
            >>> ds = bt.from_pydict({"answer": ["Paris"], "context": ["Paris is the capital."]})
            >>> ds.ml.map_batches(udf).to_pydict()["passed"]
            [True]
    """
    suffix = "\n\n" + _VERIFY_INSTRUCTION

    class _LlmVerify:
        """Holds one judge engine for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._engine = engine_factory()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            prompts = _render(template, batch)
            if instruct:
                prompts = [p + suffix for p in prompts]
            verdicts = [_parse_yes_no(o) for o in self._engine(prompts)]
            return _append(batch, output_column, verdicts, pa.bool_())

    return _LlmVerify
