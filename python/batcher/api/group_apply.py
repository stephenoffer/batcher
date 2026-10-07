"""Per-group Python callbacks: the machinery behind `GroupBy.map_groups`.

Handing a whole group to a Python function is the shape of most per-entity feature work —
a user's session sequence, a device's time series, a document's chunks — and it is what
pandas `groupby().apply()`, Polars `group_by().map_groups()`, and Spark `applyInPandas`
spell. `map_batches` does not: it sees whatever batches the engine produces, and a group is
not confined to one of them. Measured over 50,000 rows and 20 keys, every key spanned more
than one batch, so a callback written that way silently ran on fragments.

The fix uses what the engine already guarantees rather than a new operator. ``array_agg``
is a **mergeable** aggregate, so it emits exactly one row per key however many partitions
the input has, and that row carries the whole group as list columns. `map_groups` therefore
lowers to an aggregation followed by an ordinary `map_batches` over one-row-per-group
batches: the *grouping* is the aggregation's guarantee, not something this module has to
re-establish, and no group can straddle a batch boundary at any partition count.

**What that does not settle.** The resulting plan is a `map_batches` sitting above a
relational breaker, and the distributed executor's embarrassingly-parallel route
(`dist.executors.plan_analysis.is_map_prefix`) covers only a breaker-free chain down to the
scan. Whether the *multi-worker* path handles this shape is therefore exactly the question
for ``ds.group_by(...).agg(...).map_batches(fn)``, which is pre-existing and was not
measurable when this was written (the shared cluster was saturated), so it is written down
rather than assumed. The **fallback** path is settled: probing this shape is what surfaced
`_single_node` being unable to run any UDF plan at all, which is fixed and covered by
`tests/integration/test_dist_single_node_fallback.py`.

Rebuilding a group's `RecordBatch` out of those list columns is a slice of the list's child
array — no per-row Python and no copy of the values. The control-plane work is
``O(groups)``, which is the cost the caller asked for by wanting a callback per group.

This lives beside `groupby` rather than inside it because `GroupBy` is already an oversized
fluent builder, and because the adapter has to be a module-level class to survive being
pickled to a distributed worker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["GroupApply", "build_map_groups"]


class GroupApply:
    """Call `fn` once per group, given a batch holding one aggregated row per group.

    Each row of the incoming batch is one group: the key columns hold that group's key and
    every other column is a list of that group's values. `__call__` rebuilds the group's
    original `RecordBatch`, in the source's column order, and hands it to `fn`.

    A module-level class rather than a closure so it pickles to a distributed worker.
    """

    #: Marks this as a per-group adapter so a profile can tell a `map_groups` stage from a
    #: plain `map_batches` one. They are the same operator to the engine, and the per-group
    #: Python cost is the thing a profile most needs to attribute.
    batcher_group_adapter = True

    __slots__ = (
        "batch_format",
        "columns",
        "fn",
        "keys",
        "max_group_bytes",
        "max_group_rows",
        "output_schema",
    )

    def __init__(
        self,
        fn: Callable,
        keys: tuple[str, ...],
        columns: tuple[str, ...],
        batch_format: str = "pyarrow",
        *,
        output_schema: pa.Schema | None = None,
        max_group_rows: int | None = None,
        max_group_bytes: int | None = None,
    ) -> None:
        self.fn = fn
        self.keys = keys
        self.columns = columns
        self.batch_format = batch_format
        self.output_schema = output_schema
        self.max_group_rows = max_group_rows
        self.max_group_bytes = max_group_bytes

    def __repr__(self) -> str:
        name = getattr(self.fn, "__qualname__", repr(self.fn))
        return f"<map_groups {name} by {list(self.keys)}>"

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        from batcher.core.udf.call import _coerce_udf_result

        out: list[pa.RecordBatch] = []
        for row in range(batch.num_rows):
            # The group is also the reference schema: a non-Arrow `batch_format` carries a
            # string column with no non-null value in it as an untyped `object`, and without
            # something to restore the type from it comes back as neither `string` nor even
            # `null` here -- the group reassembly reads the type-less column as a list. The
            # `fn` sees this schema, so it is the one the result is held to.
            group = self._group(batch, row)
            self._check_size(batch, row, group)
            reference = self.output_schema if self.output_schema is not None else group.schema
            out.extend(_coerce_udf_result(self._call_one(group), reference))
        if self.output_schema is not None:
            return self._conform(out)
        if not out:
            # No groups in this batch, and no declared schema to report. A zero-column empty
            # table unifies with the real batches of the same stage, the way an empty
            # per-row result does in `dataset._udf.rows.rows_to_table` -- but when *every*
            # batch is empty the result has no columns, which is what `output_schema` fixes.
            return pa.table({})
        return pa.Table.from_batches(out)

    def _check_size(self, batch: pa.RecordBatch, row: int, group: pa.RecordBatch) -> None:
        """Refuse a group over `max_group_rows` / `max_group_bytes` before `fn` sees it.

        The engine has already assembled the group by now, so this guards the Python side --
        the conversion to `batch_format` and whatever `fn` builds from it -- not engine memory.
        """
        size, limit, unit = group.num_rows, self.max_group_rows, "rows"
        if limit is None or size <= limit:
            size, limit, unit = group.nbytes, self.max_group_bytes, "bytes"
            if limit is None or size <= limit:
                return
        from batcher._internal.errors import ExecutionError

        key = {k: batch.column(k)[row].as_py() for k in self.keys}
        raise ExecutionError(
            f"map_groups: group {key} holds {size:,} {unit}, over the limit of {limit:,}. "
            "Reduce it with agg(...) or a window (over(...)) instead of a per-group callback, "
            f"or raise max_group_{unit}= if the group really fits."
        )

    def _conform(self, out: list[pa.RecordBatch]) -> pa.Table:
        """Hold every result batch to the declared `output_schema`, in its column order."""
        from batcher._internal.errors import ExecutionError

        schema = self.output_schema
        assert schema is not None  # only called when a schema was declared
        tables = []
        for b in out:
            if set(b.schema.names) != set(schema.names):
                raise ExecutionError(
                    f"map_groups: the function returned columns {b.schema.names}, but "
                    f"output_schema declares {schema.names}"
                )
            try:
                tables.append(pa.Table.from_batches([b]).select(schema.names).cast(schema))
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
                raise ExecutionError(
                    f"map_groups: a result does not match output_schema ({exc})"
                ) from exc
        return pa.concat_tables(tables) if tables else schema.empty_table()

    def _call_one(self, group: pa.RecordBatch) -> Any:
        """Call `fn` on one group, reframed to `batch_format`.

        The conversion has to happen **here**, per group, not on the stage as a whole. The
        batch `map_batches` hands this adapter is the *aggregated* one — a row of list
        columns per group — so converting that to pandas would give `fn` a frame of lists
        rather than the group's rows, which is the opposite of what `applyInPandas` means.
        """
        if self.batch_format == "pyarrow":
            return self.fn(group)
        from batcher.interop.formats import result_to_arrowable, to_format

        return result_to_arrowable(self.fn(to_format(group, self.batch_format)), self.batch_format)

    def _group(self, batch: pa.RecordBatch, row: int) -> pa.RecordBatch:
        """One group's rows, rebuilt from the aggregated `row` of `batch`.

        A value column is that row's list, flattened — which is a view on the list's child
        buffer, not a copy. A key column is constant across the group, so it is broadcast by
        `take`-ing the single key value `n` times, which runs in Arrow rather than as a
        Python list of `n` repeats (the difference matters: a group can be millions of rows).
        """
        values = {name: batch.column(name).slice(row, 1).flatten() for name in self.columns}
        length = max((len(v) for v in values.values()), default=0)
        index = pa.array([0] * length, type=pa.int32())
        for name in self.keys:
            values[name] = batch.column(name).slice(row, 1).take(index)
        ordered = [*self.columns, *self.keys]
        return pa.RecordBatch.from_arrays([values[n] for n in ordered], names=ordered)


def build_map_groups(
    source: Dataset,
    keys: tuple[str, ...],
    fn: Callable,
    options: dict[str, Any],
    output_schema: pa.Schema | None = None,
    *,
    max_group_rows: int | None = None,
    max_group_bytes: int | None = None,
) -> Dataset:
    """Lower ``group_by(keys).map_groups(fn)`` to ``agg(array_agg) -> map_batches``.

    Args:
        source: The dataset being grouped.
        keys: The group-key column names.
        fn: The per-group callback.
        options: `map_batches` options to forward (`batch_format`, `num_gpus`, ...).
        output_schema: The declared result schema, also the result of an input with no
            groups; ``None`` leaves the result's schema to whatever `fn` returns.
        max_group_rows: Refuse a group with more rows than this before calling `fn`.
        max_group_bytes: Refuse a group larger than this many bytes before calling `fn`.

    Returns:
        A new lazy `Dataset` holding what `fn` returned for each group, concatenated.

    Raises:
        PlanError: if every column is a group key, leaving nothing to hand the function,
            or an option is malformed.
    """
    from batcher._internal.errors import PlanError
    from batcher.plan.expr_ir import Col

    key_set = set(keys)
    columns = tuple(c for c in source.columns if c not in key_set)
    if not columns:
        raise PlanError(
            "map_groups needs at least one non-key column to hand the function; every "
            f"column of this dataset is a group key ({list(keys)}). Group by fewer columns, "
            "or use .agg(n=bt.count()) if you only need the group sizes."
        )
    _check_options(options, output_schema, max_group_rows, max_group_bytes)
    # `input_columns`/`preserves_columns` describe the stage's input, and this stage's input
    # is the *aggregated* relation, not the user's. Forwarding them would prune or push a
    # predicate against a schema the caller never saw, so refuse rather than misapply them.
    for name in ("input_columns", "preserves_columns"):
        if options.get(name) is not None:
            raise PlanError(
                f"map_groups does not take {name}: it describes the columns of the stage's "
                "input, and this stage runs over the aggregated one-row-per-group relation. "
                "Select the columns you need before group_by instead."
            )
        options.pop(name, None)
    # The reframing happens per group inside the adapter, so the outer stage stays Arrow.
    adapter = GroupApply(
        fn,
        keys,
        columns,
        options.pop("batch_format", "pyarrow"),
        output_schema=output_schema,
        max_group_rows=max_group_rows,
        max_group_bytes=max_group_bytes,
    )
    collected = source.group_by(*keys).agg(**{name: Col(name).array_agg() for name in columns})
    return collected.map_batches(adapter, **options)


def _check_options(
    options: dict[str, Any],
    output_schema: pa.Schema | None,
    max_group_rows: int | None,
    max_group_bytes: int | None,
) -> None:
    """Validate the size limits and `output_schema`, which also sets `output_columns`."""
    from batcher._internal.errors import PlanError

    for name, limit in (("max_group_rows", max_group_rows), ("max_group_bytes", max_group_bytes)):
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise PlanError(f"map_groups({name}=...) must be a positive integer, got {limit!r}")
    if output_schema is not None:
        if not isinstance(output_schema, pa.Schema):
            raise PlanError(
                f"map_groups(output_schema=...) must be a pyarrow.Schema, got "
                f"{type(output_schema).__name__}"
            )
        declared = options.get("output_columns")
        if declared is not None and list(declared) != output_schema.names:
            raise PlanError(
                f"map_groups: output_columns {list(declared)} disagree with output_schema "
                f"{output_schema.names}; pass output_schema alone"
            )
        options["output_columns"] = list(output_schema.names)
