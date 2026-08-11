"""Check the device result against the CPU engine — the device tier's two oracles.

The device tier is a *second implementation of the same semantics*. cuDF has no maintained
Rust binding, so unlike every other tier it cannot share `bc_expr::Expr` — it is a translator
from the same JSON IR onto a dataframe library, and the two can drift. The Cranelift JIT has
exactly this hazard and answers it with a bit-for-bit parity requirement against Tier-0; this
module is that requirement for the device tier, which had no equivalent.

Two checks live here, and they cost very different things:

**The schema contract (`enforce_schema_contract`) — free, and on for every device run.** The
engine can say what columns a plan returns *without running it*: `LogicalPlan.available_schema`
is the same static analysis `Dataset.schema` is answered from, and it is pure plan inspection,
no rows and no engine call. So every device result can be held against the engine's own
declared column types on every query, at the cost of one walk over a field list. A result whose
columns disagree is refused and the query uses the CPU engine.

That check exists because of what the tier has actually shipped. **Every device-tier defect on
record is a column-*type* defect with correct values**: a DATE returning `date32` under pandas
and `timestamp[ms]` on a real device (`core/gpu_plan/backend.py::remember_date_alias`), an
integer `abs` widening to double, an empty cuDF string column converting as Arrow `null`. Each
was right in CI and wrong on hardware, because the translator's suite runs on **pandas** — CI
has no GPU — and a pandas stand-in is structurally unable to see a cuDF-only type. A schema the
engine declared before either backend ran *is* able to see it, on the device, with no GPU in CI
and nothing switched on. Every one of those three would have declined to the CPU engine instead
of returning a wrong column.

**The shadow re-run (`shadow_verify`) — expensive, and off by default.** The contract above
sees types; only running the plan twice sees values. That doubles the work, so it stays behind
`distributed.gpu_shadow_verify` for benchmark and staging runs. It compares schema first for
the reason above, then values.

`shadow_verify` walks the result in Python, which is ordinarily forbidden on a hot path
(`.claude/rules/architecture.md`). It is allowed there because the mode exists to be slow and
certain, and because it never runs unless an operator asks for it. The schema contract touches
no rows at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.plan.logical import LogicalPlan

__all__ = [
    "DeviceDivergence",
    "compare_results",
    "declared_schema",
    "enforce_schema_contract",
    "schema_contract_violation",
    "shadow_verify",
]

#: Relative tolerance for a floating-point value difference. A device sums in a different
#: order from the CPU engine, so the last bits of a `SUM` over millions of rows legitimately
#: differ; a divergence that matters is orders of magnitude larger than this. Tight enough
#: that a genuinely wrong kernel cannot hide under it.
_FLOAT_RTOL = 1e-9


class DeviceDivergence(Exception):
    """The device tier and the CPU engine disagreed on the same plan.

    A defect, never a decline: the tier's whole contract is that the device changes *where* a
    plan runs and never *what* it computes. Raised only to be handed to `note_gpu_failure`,
    which logs it at warning level — the query still returns the CPU engine's answer.
    """


def _schema_mismatch(gpu: pa.Schema, want: pa.Schema, source: str) -> str | None:
    """A description of the first schema difference, or `None` when the schemas agree.

    Compares names and types only. Nullability and schema metadata are deliberately not part
    of the contract: the engine's static analysis marks every inferred field nullable, and a
    device result's flags come from whatever the library happened to build, so comparing them
    would report a difference on every query while saying nothing about the answer.

    Args:
        gpu: The schema the device produced.
        want: The schema it is being held to.
        source: What produced `want`, for the message ("the CPU engine", "the engine").

    Returns:
        A one-line description of the first difference, or `None`.
    """
    if gpu.names != want.names:
        return f"columns {gpu.names} vs {source} {want.names}"
    for field, want_field in zip(gpu, want, strict=True):
        if field.type != want_field.type:
            return (
                f"column {field.name!r} is {field.type} on the device, "
                f"{want_field.type} from {source}"
            )
    return None


def declared_schema(plan: LogicalPlan) -> pa.Schema | None:
    """The columns and types the engine says `plan` returns, without running it, or `None`.

    `available_schema` is all-or-nothing by design — one column whose type cannot be inferred
    makes the whole schema unknown — so `None` means "the engine has no opinion here", not
    "the plan has no schema". The caller then has nothing to check against and lets the result
    through, which is the same position the tier was in before this existed.

    Args:
        plan: The optimized logical plan the device ran.

    Returns:
        The declared Arrow schema, or `None` when the engine cannot infer every output type.
    """
    try:
        inferred = plan.available_schema()
    except Exception as exc:  # pragma: no cover - analysis must never break a query
        from batcher._internal.logging import note_suppressed

        note_suppressed("api", "infer the engine's declared schema for the GPU result", exc)
        return None
    return None if inferred is None else inferred.arrow


def schema_contract_violation(result: pa.Table, plan: LogicalPlan) -> str | None:
    """How `result`'s columns differ from what the engine declares for `plan`, or `None`.

    Args:
        result: The table the device tier produced.
        plan: The optimized logical plan it ran.

    Returns:
        A one-line description of the first difference, or `None` when they agree or when the
        engine declares nothing.
    """
    declared = declared_schema(plan)
    return None if declared is None else _schema_mismatch(result.schema, declared, "the engine")


def enforce_schema_contract(result: pa.Table, plan: LogicalPlan) -> pa.Table | None:
    """`result` when its columns are what the engine declares, else `None` — use the CPU engine.

    The device tier's contract is that a GPU changes *where* a plan runs and never *what* it
    computes, and "what" includes every column's type. This is the cheap half of holding it:
    the check reads a field list, touches no rows, needs no second execution and no device, so
    it runs on every device result rather than under a diagnostic flag.

    A violation is a **defect**, not a decline — the translator produced an answer it should
    not have — so it is reported through `note_gpu_failure` at warning level, and the query
    falls back to the CPU engine and returns the right columns.

    Args:
        result: The table the device tier produced.
        plan: The optimized logical plan it ran.

    Returns:
        `result` when the columns agree, else `None`.
    """
    difference = schema_contract_violation(result, plan)
    if difference is None:
        return result
    from batcher.api.terminal.gpu_backend.failure import note_gpu_failure

    note_gpu_failure(
        "return the GPU result: its columns are not the ones the engine declares for this "
        "plan; using the CPU engine",
        DeviceDivergence(difference),
    )
    return None


def _sort_key(row: tuple) -> tuple:
    """A total order over result rows that tolerates nulls and mixed types.

    Only ever used to put two multisets in the same order before comparing them, so it has to
    be total and deterministic rather than meaningful.
    """
    return tuple((value is None, str(type(value)), str(value)) for value in row)


def _values_mismatch(gpu: pa.Table, cpu: pa.Table) -> str | None:
    """A description of the first value difference, or `None` when the rows agree.

    Compared as a multiset: a group-by does not fix its output order on either backend, so
    order is not part of the contract unless the plan ends in a sort — and a sort's order is
    already pinned by the row sequence within the sorted key.
    """
    if gpu.num_rows != cpu.num_rows:
        return f"{gpu.num_rows} rows on the device, {cpu.num_rows} on CPU"
    gpu_rows = sorted((tuple(r.values()) for r in gpu.to_pylist()), key=_sort_key)
    cpu_rows = sorted((tuple(r.values()) for r in cpu.to_pylist()), key=_sort_key)
    for index, (got, want) in enumerate(zip(gpu_rows, cpu_rows, strict=True)):
        for column, (a, b) in enumerate(zip(got, want, strict=True)):
            if _differs(a, b):
                name = gpu.schema.names[column]
                return f"row {index}, column {name!r}: device {a!r} vs CPU {b!r}"
    return None


def _differs(a: Any, b: Any) -> bool:
    """Whether two cell values disagree beyond floating-point accumulation order."""
    if a is None or b is None:
        return a is not b
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # NaN == NaN, for this purpose
            return False
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) > _FLOAT_RTOL * scale
    return a != b


def compare_results(gpu: pa.Table, cpu: pa.Table) -> str | None:
    """The first disagreement between a device result and the CPU engine's, or `None`.

    Args:
        gpu: The table the device tier produced.
        cpu: The table the CPU engine produced for the same plan.

    Returns:
        A one-line description of the first difference found, schema before values, or `None`
        when the two agree.
    """
    return _schema_mismatch(gpu.schema, cpu.schema, "CPU") or _values_mismatch(gpu, cpu)


def shadow_verify(plan, sources, columns, gpu_result: pa.Table) -> pa.Table:
    """Re-run `plan` on the CPU engine and return whichever result is trustworthy.

    On agreement the device result is returned unchanged, so a verified run differs from an
    unverified one only in cost. On disagreement the **CPU result** is returned — the mode is
    for finding drift, and returning the answer known to match the oracle is what makes it
    safe to leave on in staging — and the difference is reported through `note_gpu_failure` as
    a backend defect.

    Args:
        plan: The optimized logical plan the device ran.
        sources: The plan's bound sources.
        columns: The terminal op's requested output columns.
        gpu_result: The table the device tier produced.

    Returns:
        The device result when the two agree, else the CPU engine's result.
    """
    from batcher import core
    from batcher.api import executors
    from batcher.api.terminal.gpu_backend.failure import note_gpu_failure

    try:
        cpu_result = executors.select(plan, distributed=False).execute(
            plan, sources, core.ExecutionContext(columns=columns, hub=core.default_hub())
        )
    except Exception as exc:
        # The oracle itself failed. That is not a divergence and must not be reported as one,
        # but it does mean this run verified nothing — say so rather than implying a pass.
        note_gpu_failure("shadow-verify the GPU result (the CPU engine raised)", exc)
        return gpu_result

    difference = compare_results(gpu_result, cpu_result)
    if difference is None:
        return gpu_result
    note_gpu_failure(
        "shadow-verify the GPU result; returning the CPU engine's",
        DeviceDivergence(difference),
    )
    return cpu_result
