"""A bare scan needs no engine — the reader has already produced the plan's output.

``bt.read.parquet(path).collect()`` optimizes to a plan that is a single `Scan`. The
source reader then does all of the work: it decodes the files, and it applies the
projection Kyber pushed into it, so the batches it hands back **are** the query's result.
Running them through the engine anyway does nothing except move them:

    Rust decode → export every array to Python → import every array back into Rust →
    pass through a no-op Scan operator → export every array to Python again → concatenate

The Arrow C Data Interface makes each of those crossings zero-*copy*, but not zero-*cost*:
a 20M-row, 16-column table at the native read's batch size is ~300 batches, so a round trip
is ~10,000 array exports and imports plus the Python objects to wrap them. Measured on a
1.6 GB file, that pass-through cost **189 ms on a 709 ms read — a quarter of the wall clock
to accomplish nothing**, and it is paid by every plain scan, which is the single most common
thing anyone asks a data engine to do.

So: recognize the shape, and skip the engine.

## Why this is safe, and exactly when it applies

The result must be **bit-identical** to what the engine would have returned, which pins down
three conditions. `scan_only_result` returns `None` — deferring to the engine — unless all
three hold.

1. **The plan is exactly a `Scan`.** Any operator above it is real work the engine must do.
2. **No predicate was pushed to the source.** A pushed predicate prunes *row groups*, which
   is a superset filter — the engine keeps its `Filter` to finish the job. A plan whose only
   node is a `Scan` has no `Filter`, so it cannot have a predicate either; the check is a
   belt-and-braces guard against a future pushdown rule that drops one.
3. **The schema is normalized the way the FFI boundary would have normalized it.** Crossing
   into the engine widens narrow numerics (Int8/16/32 → Int64, Float16/32 → Float64); a
   result that skipped the crossing must be widened identically or a `read().collect()` would
   report a different dtype than `read().filter(...).collect()` on the same file. `plan.types.widen`
   is the same rule `bc_py::widen_to` applies, and `Scan.available_schema()` already states
   the widened schema the plan promises.

Columns are then reordered to the plan's own column order, because a Parquet reader returns
leaves in *file* order regardless of the order the projection asked for them in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.plan.logical import Scan

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from batcher.plan.logical import LogicalPlan

__all__ = ["scan_only_result"]


def scan_only_result(
    plan: LogicalPlan,
    resolved: list[list[pa.RecordBatch]],
    source_predicates: Mapping[int, Any] | None = None,
) -> pa.Table | None:
    """The result of `plan` when it is a bare scan, or None to run the engine.

    Args:
        plan: The optimized *logical* plan.
        resolved: Each source's already-read batches, indexed by `Scan.source_id`.
        source_predicates: Predicates Kyber pushed into each source, if any.

    Returns:
        The scan's rows as a table, or None when this shortcut does not apply.
    """
    scan = _scan_under_column_selection(plan)
    if scan is None:
        return None
    if (source_predicates or {}).get(scan.source_id) is not None:
        return None  # a pushed predicate only prunes row groups; the engine must still filter
    if scan.source_id >= len(resolved):
        return None

    schema = plan.available_schema()
    if schema is None:
        return None
    return _as_table(resolved[scan.source_id], schema.arrow)


def _scan_under_column_selection(plan: LogicalPlan) -> Scan | None:
    """The `Scan` this plan is, or that a pure column *selection* sits directly on.

    A bare `Scan` is the shape this module was written for, and it left the far commoner
    one on the slow path: `SELECT a, b FROM <file>` optimizes to `Project(Scan)`, and the
    `Project` — once Kyber has pushed its column list into the reader — asks for exactly
    the columns the reader already produced. It is an identity, and the round trip it was
    paying for is the whole cost. Measured on 32M rows x 16 `int64` columns of local
    parquet, selecting one column: **97.7 ms against 33.7 ms** for the same query written
    as a read-time projection, at the same CPU — the extra 64 ms is the crossing.

    "Pure column selection" is exact and narrow: every output is a bare `Col` whose name is
    its own alias. A rename cannot take this path (the batches carry the *source* names and
    nothing here renames them), and neither can a computed expression, a literal, or a
    duplicate of one input column under two names — each is real work, and each falls back.

    Column *order* may still differ, which is fine: `_as_table` restates the plan's order
    with a metadata-only `select`, exactly as it already does for a reader that returns
    leaves in file order.
    """
    from batcher.plan.expr_ir.nodes import Col
    from batcher.plan.logical import Project

    if isinstance(plan, Scan):
        return plan
    if not isinstance(plan, Project) or not isinstance(plan.input, Scan):
        return None
    names = [item.alias for item in plan.items]
    if len(set(names)) != len(names):
        return None
    identity = all(
        isinstance(item.expr, Col) and item.expr.name == item.alias for item in plan.items
    )
    return plan.input if identity else None


def _as_table(batches: list[pa.RecordBatch], schema: pa.Schema) -> pa.Table | None:
    """`batches` as one table with exactly `schema` — the widened schema the plan promises.

    Returns None rather than guessing if the batches do not carry every column the plan
    expects; the engine is always the correct fallback, so an unexpected shape costs a little
    time and never a wrong answer.
    """
    if not batches:
        return schema.empty_table()
    table = pa.Table.from_batches(batches)
    if not set(schema.names).issubset(table.column_names):
        return None

    # A Parquet reader returns leaves in *file* order, not in the order the projection asked
    # for them — so the plan's column order has to be restated. Metadata-only; no data moves.
    table = table.select(schema.names)
    if table.schema.equals(schema, check_metadata=False):
        return table
    return _widen(table, schema)


def _widen(table: pa.Table, schema: pa.Schema) -> pa.Table | None:
    """Cast only the columns whose type differs — the FFI boundary's numeric widening.

    Casting the whole table would copy every column, including the ones already correct
    (which is all of them, for the Parquet types anyone actually stores). Cast per column so
    the common case moves no data at all.
    """
    columns = []
    for field in schema:
        column = table.column(field.name)
        if column.type.equals(field.type):
            columns.append(column)
            continue
        try:
            columns.append(column.cast(field.type))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return None  # a cast the boundary would have done differently → let the engine do it
    return pa.Table.from_arrays(columns, schema=schema)
