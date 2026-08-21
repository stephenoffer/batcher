"""Ordered-bucket offsetting: the algebra that makes a *global* window splittable.

A global (no ``PARTITION BY``) window has one partition over every row, so it has no
per-partition seam to cut along -- which is why it is the one window shape with neither a
grace-spill path nor, until this module was wired into the dispatcher, a distributed one.

It does have a seam, just a different one. **Range**-partition the rows by the single
``ORDER BY`` key into buckets that are ordered relative to each other, and equal keys land
in one bucket, so no peer group and no frame ever spans a boundary. Each bucket can then be
windowed independently, and the prior buckets contribute to it exactly one constant (or, for
the running extremes, one element-wise) shift:

* ``row_number`` / ``rank`` -- plus the number of rows in prior buckets.
* ``dense_rank`` -- plus the number of *distinct* order keys in prior buckets.
* running ``sum`` / ``count`` -- plus the prior buckets' total.
* running ``min`` / ``max`` -- element-wise against the prior buckets' running extreme.
* ``avg`` -- not a constant shift itself, but its ``sum`` and its ``count`` each are, so it
  is offset through the pair (which is why `inject_avg_helpers` asks the kernel for them).
* ``first_value`` -- the first bucket's first value, broadcast.

`lag` / `lead` / `last_value` / `ntile` / `percent_rank` / `cume_dist` are **not** offsettable
this way (each reads rows the bucket does not hold, or divides by a partition total the
bucket does not know), so `supports_ordered_bucket_offsets` refuses them and the caller keeps
the materializing kernel -- still correct, just not split.

Both consumers of this algebra live one directory up from here in spirit and one import away
in fact: `stream` runs the buckets one at a time on a single node under a memory envelope,
and `flight` runs them on different machines at the same time. They share this module rather
than each spelling the offsets out, because two statements of the same arithmetic is exactly
how a distributed result drifts from its single-node oracle.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.plan.expr_ir import Col
from batcher.plan.logical import Window

__all__ = [
    "OrderedBucketOffsets",
    "bucket_order",
    "inject_avg_helpers",
    "supports_ordered_bucket_offsets",
]

#: Window functions whose global value is recovered from the within-bucket value plus a
#: constant/element-wise per-bucket offset (so per-bucket compute + offset == single-node).
#: `avg` qualifies through its running `sum` and `count`, each of which is a constant shift.
_OFFSETTABLE = frozenset({"row_number", "rank", "dense_rank", "sum", "count", "min", "max", "avg"})
_NEEDS_COL_INPUT = frozenset({"sum", "count", "min", "max", "avg", "first_value"})
_UNSET = object()


def supports_ordered_bucket_offsets(window: Window) -> bool:
    """Whether `window` is a global window the ordered-bucket-offset algebra covers.

    Requires: no partition keys (global); exactly one plain-column order key (the column the
    range partitioner cuts on) *of a type that partitioner can cut*; every function
    offsettable or `first_value`, with no explicit frame; and aggregate/`first_value` inputs
    are plain columns.

    The type test is the one this predicate was missing while its sort sibling
    (`supports_spilling_sort`) had it, and both guard the same range partitioner. Without it
    a `rank()` over a Boolean column passed the shape test, collected correctly, and then
    raised a bare ``RuntimeError: range-partition key must be a numeric column`` the moment
    the identical plan was streamed — a query that worked in batch failing in streaming,
    which is precisely what one execution model is supposed to rule out. Declining here costs
    memory (the materializing kernel runs instead), never correctness.

    The key's type comes from `available_schema`, the plan layer's own static inference, so
    the check needs no sources and no zero-row execution. An uninferable schema, or a derived
    key absent from it, declines for the same reason `supports_spilling_sort` declines an
    unknown key: stay out of the range partition rather than fail inside it.

    Args:
        window: The window operator to classify.

    Returns:
        True when every bucket can be windowed independently and corrected by an offset.
    """
    if window.rank_limit is not None or window.partition_keys:
        return False
    if len(window.order_keys) != 1 or not isinstance(window.order_keys[0].expr, Col):
        return False
    if not _single_source_input(window):
        return False
    if not _key_type_partitionable(window):
        return False
    for fn in window.functions:
        if fn.frame is not None:
            return False
        if fn.func not in _OFFSETTABLE and fn.func != "first_value":
            return False
        if fn.func in _NEEDS_COL_INPUT and not isinstance(fn.input, Col):
            return False
    return True


def _single_source_input(window: Window) -> bool:
    """Whether the window's input names exactly one source.

    `stream_spilling_global_window` reaches `_relabel_single_source`, which **raises** on a
    multi-source input rather than declining -- so a global window above a join answered
    `collect()` and died under `collect(spill=True)` with `PlanError: expected a
    single-source subplan to relabel`. Same defect, same fix and same reasoning as
    `supports_spilling_window` and `supports_spilling_sort`: a predicate that answers
    *whether* a path applies must never raise when the answer is "no".

    Imported inside the function for the reason `_key_type_partitionable` gives: this module
    is on `dist.executor`'s eager-import budget and `executors.plan_analysis` is not.
    """
    from batcher.dist.executors.plan_analysis import _single_source

    return _single_source(window.input)


def _key_type_partitionable(window: Window) -> bool:
    """Whether the order key's statically-inferred type is one the partitioner can cut.

    Imported inside the function on purpose: this module is the one part of
    `dist.global_window` that `dist.executor` imports eagerly (see the package docstring on
    the 0.44 s `import ray` that eager submodule loading used to cost), and
    `executors.partition_io` is not on that budget.
    """
    from batcher.dist.executors.partition_io import range_partitionable

    schema = window.input.available_schema()
    if schema is None:
        return False
    index = schema.arrow.get_field_index(window.order_keys[0].expr.name)
    if index < 0:
        return False
    return range_partitionable(schema.arrow.field(index).type)


def inject_avg_helpers(window: Window, win_ir: dict) -> dict[str, tuple[str, str]]:
    """Append a running `sum` and `count` to `win_ir` for every `avg` function.

    The private aliases carry a prefix that cannot collide with a user column (window aliases
    are validated against the input schema, which never contains one), so the kernel computes
    them for free next to the average and `OrderedBucketOffsets` divides one offset running
    total by the other. Both are dropped again before any row is yielded.

    Args:
        window: The window whose `avg` functions need helper columns.
        win_ir: The window IR to append to. Mutated in place; the caller owns a copy.

    Returns:
        A mapping from each `avg` alias to its ``(sum_alias, count_alias)`` pair.
    """
    helpers: dict[str, tuple[str, str]] = {}
    for fn in window.functions:
        if fn.func == "avg":
            sa, ca = f"__ws_sum::{fn.alias}", f"__ws_cnt::{fn.alias}"
            helpers[fn.alias] = (sa, ca)
            inp = fn.input.to_ir()
            win_ir["functions"].append({"func": "sum", "alias": sa, "offset": 1, "input": inp})
            win_ir["functions"].append({"func": "count", "alias": ca, "offset": 1, "input": inp})
    return helpers


def bucket_order(n_buckets: int, descending: bool) -> range:
    """The bucket ids in *global sort order*, so the offsets accumulate correctly.

    Args:
        n_buckets: How many ordered buckets the range partitioner produced.
        descending: Whether the window's order key sorts descending.

    Returns:
        The bucket ids to visit, lowest key first (reversed when descending).
    """
    return range(n_buckets - 1, -1, -1) if descending else range(n_buckets)


class OrderedBucketOffsets:
    """Turns each bucket's within-bucket window result into the global one.

    Stateful and **order-dependent by contract**: feed it the buckets in `bucket_order`,
    exactly once each, and every row it hands back carries the value the single-node kernel
    would have produced for it.

    Args:
        window: The window being computed, whose functions decide which offsets apply.
        avg_helpers: The `inject_avg_helpers` mapping, empty when there is no `avg`.
    """

    def __init__(self, window: Window, avg_helpers: dict[str, tuple[str, str]]) -> None:
        self._window = window
        self._avg = avg_helpers
        self._prior_rows = 0
        aliases = [f.alias for f in window.functions]
        self._dense = dict.fromkeys(aliases, 0)
        self._sum: dict[str, float] = dict.fromkeys(aliases, 0)
        # Whether any prior bucket held a non-null input. `_sum` alone cannot answer that:
        # a genuine total of 0 and "nothing seen yet" are the same number, and a running
        # `sum` must stay NULL only in the second case.
        self._seen = dict.fromkeys(aliases, False)
        self._count = dict.fromkeys(aliases, 0)
        self._min: dict[str, object] = dict.fromkeys(aliases)
        self._max: dict[str, object] = dict.fromkeys(aliases)
        self._first: dict[str, object] = dict.fromkeys(aliases, _UNSET)

    def apply(self, wt: pa.Table) -> pa.Table:
        """Correct one bucket's window columns to their global values.

        Args:
            wt: The bucket's windowed rows, as returned by the window kernel.

        Returns:
            The same rows with every window column shifted to its global value, and the
            private `avg` helper columns dropped.
        """
        import pyarrow.compute as pc

        n = wt.num_rows
        for fn in self._window.functions:
            idx = wt.schema.get_field_index(fn.alias)
            col = wt.column(idx)
            alias = fn.alias
            if fn.func in ("row_number", "rank"):
                col = pc.add(col, self._prior_rows)
            elif fn.func == "dense_rank":
                bucket_distinct = pc.max(col).as_py() or 0
                col = pc.add(col, self._dense[alias])
                self._dense[alias] += bucket_distinct
            elif fn.func == "sum":
                # The kernel's within-bucket running sum is NULL until this bucket's first
                # non-null input — but if a prior bucket held one, the *global* running sum
                # at those rows is defined and equals the prior total. Adding to the NULL
                # (`NULL + prior == NULL`) silently dropped every such row's value, which
                # only ever showed on a bucket that opens with nulls and is not the first.
                # Where nothing non-null has been seen at all, NULL is the right answer and
                # the column is left as it is.
                if self._seen[alias]:
                    col = pc.add(pc.fill_null(col, 0), self._sum[alias])
                s = pc.sum(wt.column(fn.input.name)).as_py()
                if s is not None:
                    self._sum[alias] += s
                    self._seen[alias] = True
            elif fn.func == "count":
                col = pc.add(col, self._count[alias])
                self._count[alias] += pc.count(wt.column(fn.input.name)).as_py()
            elif fn.func == "min":
                col = self._extreme(col, wt, fn, self._min, pc.min_element_wise, min, pc.min)
            elif fn.func == "max":
                col = self._extreme(col, wt, fn, self._max, pc.max_element_wise, max, pc.max)
            elif fn.func == "avg":
                col = self._avg_column(wt, fn)
            elif fn.func == "first_value":
                if self._first[alias] is _UNSET:
                    self._first[alias] = col[0].as_py() if n else None
                else:
                    col = pa.array([self._first[alias]] * n, type=col.type)
            wt = wt.set_column(idx, wt.schema.field(idx), col)
        if self._avg:
            # Drop the private sum/count columns the avg offset borrowed.
            hidden = {a for pair in self._avg.values() for a in pair}
            wt = wt.select([c for c in wt.column_names if c not in hidden])
        self._prior_rows += n
        return wt

    def _extreme(self, col, wt, fn, state, element_wise, pick, reduce_fn):
        """Offset a running `min`/`max` against the prior buckets' running extreme."""
        alias = fn.alias
        if state[alias] is not None:
            col = element_wise(col, pa.scalar(state[alias], col.type))
        bucket = reduce_fn(wt.column(fn.input.name)).as_py()
        if bucket is not None:
            state[alias] = bucket if state[alias] is None else pick(state[alias], bucket)
        return col

    def _avg_column(self, wt, fn):
        """Offset a running `avg` through its injected running `sum` and `count`."""
        import pyarrow.compute as pc

        alias = fn.alias
        sa, ca = self._avg[alias]
        # The kernel's running sum is NULL until the first non-null input, but for the
        # offset that means a 0 contribution — coalesce before adding the prior buckets'
        # total. The running count is 0 (never null) there.
        within_sum = pc.fill_null(pc.cast(wt.column(sa), pa.float64()), 0.0)
        tot_sum = pc.add(within_sum, float(self._sum[alias]))
        tot_cnt = pc.add(wt.column(ca), self._count[alias])
        # No non-null value through this row (globally) ⇒ the mean is NULL, and the 0/0 the
        # divide would produce there is discarded by the mask.
        col = pc.if_else(
            pc.equal(tot_cnt, 0),
            pa.scalar(None, pa.float64()),
            pc.divide(tot_sum, pc.cast(tot_cnt, pa.float64())),
        )
        bs = pc.sum(wt.column(fn.input.name)).as_py()
        self._sum[alias] += bs if bs is not None else 0
        self._count[alias] += pc.count(wt.column(fn.input.name)).as_py()
        return col
