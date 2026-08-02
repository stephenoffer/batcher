"""Run the device result against the CPU engine and report where they disagree.

The device tier is a *second implementation of the same semantics*. cuDF has no maintained
Rust binding, so unlike every other tier it cannot share `bc_expr::Expr` — it is a translator
from the same JSON IR onto a dataframe library, and the two can drift. The Cranelift JIT has
exactly this hazard and answers it with a bit-for-bit parity requirement against Tier-0; this
module is that requirement for the device tier, which had no equivalent.

**Why a runtime mode rather than a test.** The translator's suite runs it on **pandas**, never
on cuDF, because CI has no GPU. That stand-in is structurally unable to catch the divergences
that actually shipped, and the package's own docstrings record two of them: a DATE column
returning `date32` under pandas and `timestamp[ms]` on a real device
(`core/gpu_plan/backend.py::remember_date_alias`), and an integer `abs` widening to double.
Both were right in CI and wrong on hardware. Until there is a GPU CI lane, the only place the
device is observable is a real run — so the check lives here, off by default, and is switched
on for benchmark and staging runs.

**Schema first, and it is not a formality.** Both recorded bugs were *column type* bugs with
correct values. A comparison that comes back "same rows" while the schema moved is exactly the
failure mode: a sharded fan-out concatenates its shards' partials, and a shard that fell back
to the CPU engine contributes `int64` beside a device shard's `int32`.

This walks the result in Python, which is ordinarily forbidden on a hot path
(`.claude/rules/architecture.md`). It is allowed here because the mode exists to be slow and
certain, and because it never runs unless an operator asks for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["DeviceDivergence", "compare_results", "shadow_verify"]

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


def _schema_mismatch(gpu: pa.Table, cpu: pa.Table) -> str | None:
    """A description of the first schema difference, or `None` when the schemas agree."""
    if gpu.schema.names != cpu.schema.names:
        return f"columns {gpu.schema.names} vs CPU {cpu.schema.names}"
    for field, cpu_field in zip(gpu.schema, cpu.schema, strict=True):
        if field.type != cpu_field.type:
            return f"column {field.name!r} is {field.type} on the device, {cpu_field.type} on CPU"
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
    return _schema_mismatch(gpu, cpu) or _values_mismatch(gpu, cpu)


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
