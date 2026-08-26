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

#: The running *associative folds*: a window function whose value at a row is
#: ``identity OP x0 OP x1 OP ... OP x_row`` over the non-null inputs, and which is therefore
#: offset by folding the prior buckets' accumulated value in on the left. Each entry is
#: ``(identity, pyarrow-compute op, numpy ufunc)``: the identity fills the within-bucket
#: nulls (a running fold is NULL until its first non-null input, where the correct global
#: value is exactly the prior accumulation), the compute op folds the prior accumulation into
#: every row, and the ufunc reduces this bucket's own inputs to the value the *next* bucket
#: carries. The reduce reads the **input** column, never the running one: the kernel returns a
#: bucket's rows in arrival order, not sort order, so the running column's last cell is an
#: arbitrary row's prefix rather than the bucket's total.
#:
#: `sum` is the member this table was generalized from — it had the arithmetic written out
#: inline, and the bitwise and boolean folds the engine computes were declined by
#: `supports_ordered_bucket_offsets` purely because nobody had written theirs. Declining costs
#: a *distributed* global window its split: the materializing kernel runs the whole relation
#: on one node instead. The identity is the only thing that differs between them, so stating
#: it as data rather than as a branch is what makes adding the sixth cost one line.
#:
#: **`product` is deliberately absent**, and is the one running fold the engine computes that
#: is not here. Reassociation is tolerated for a float reduction — `combine` is associative in
#: exact arithmetic and IEEE addition is not, so a `sum` differs in its last bits with the
#: partition count. `product` does not fail in its last bits. Over a few thousand values it
#: overflows to `inf` and underflows to `0`, and the two orders then disagree on `inf * 0`,
#: which is `NaN` one way and `0` the other: measured on 4,000 rows, single-node returned
#: `-0.0` where the seven-bucket split returned `NaN`. The other six folds are exact integer
#: or boolean arithmetic, so their reassociation is not merely bounded but nil.
_FOLDS: dict[str, tuple[object, str, str]] = {
    "sum": (0, "add", "add"),
    "bit_and": (-1, "bit_wise_and", "bitwise_and"),
    "bit_or": (0, "bit_wise_or", "bitwise_or"),
    "bit_xor": (0, "bit_wise_xor", "bitwise_xor"),
    "bool_and": (True, "and_", "logical_and"),
    "bool_or": (False, "or_", "logical_or"),
}

#: Window functions whose global value is recovered from the within-bucket value plus a
#: constant/element-wise per-bucket offset (so per-bucket compute + offset == single-node).
#: `avg` qualifies through its running `sum` and `count`, each of which is a constant shift.
_OFFSETTABLE = frozenset(
    {"row_number", "rank", "dense_rank", "count", "min", "max", "avg", *_FOLDS}
)
#: Functions whose offset reads the *input* column out of the bucket, so the input must be a
#: plain column the kernel also emits alongside the running one.
_NEEDS_COL_INPUT = frozenset({"count", "min", "max", "avg", "first_value", *_FOLDS})
_UNSET = object()


def supports_ordered_bucket_offsets(window: Window) -> bool:
    """Whether `window` is a global window the ordered-bucket-offset algebra covers.

    Requires: no partition keys (global); a **leading** order key that is a plain column (the
    column the range partitioner cuts on) *of a type that partitioner can cut*; every function
    offsettable or `first_value`, with no explicit frame; and aggregate/`first_value` inputs
    are plain columns.

    Only the leading key is constrained, and the further keys may be anything. The bucket
    argument survives them intact: a peer group under a multi-key `ORDER BY` is a set of rows
    equal on *every* key, so it is contained in a set of rows equal on the leading key, which
    the range partitioner puts in one bucket — no peer group and no frame straddles a cut. And
    the buckets stay ordered relative to each other, because a row in an earlier bucket has a
    strictly smaller leading key and therefore sorts before every row in a later one whatever
    the trailing keys say. Each bucket is then windowed by its full key list, which is what
    the reducer already receives (`unary_task_ir(window)` carries every key).

    This said `len(order_keys) != 1` and refused the rest, which was not merely conservative:
    a global window is not a `_split_at` pass-through, so nothing carried it up and
    `ORDER BY a, b` **raised** `PlanError` on distributed data rather than declining to a
    slower path. All three drivers already cut on `order_keys[0]` alone — the sort states the
    same rule for itself ("only the leading key drives the partitioning, the rest are
    evaluated by each reducer's local sort").

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
    if not window.order_keys or not isinstance(window.order_keys[0].expr, Col):
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
        # The prior buckets' accumulation for each running fold (`_FOLDS`), or `_UNSET` when
        # no prior bucket has held a non-null input. A sentinel rather than the op's identity:
        # a genuine accumulation *equal to* the identity (a `sum` of 0, a `bit_or` of 0, a
        # `bool_and` of True) and "nothing seen yet" are the same value, and a running fold
        # must stay NULL only in the second case.
        self._fold: dict[str, object] = dict.fromkeys(aliases, _UNSET)
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
            elif fn.func in _FOLDS:
                col = self._fold_column(col, wt, fn)
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

    def _fold_column(self, col, wt, fn):
        """Offset a running associative fold (`sum`/`product`/bitwise/boolean) by the prior
        buckets' accumulation.

        The kernel's within-bucket value is NULL until this bucket's first non-null input —
        but if a prior bucket held one, the *global* fold at those rows is defined and equals
        the prior accumulation. Folding into the NULL directly (`NULL OP prior == NULL` under
        every one of these ops) silently dropped every such row's value, which only ever
        showed on a bucket that opens with nulls and is not the first. Filling with the op's
        **identity** first is what makes those rows come back as the prior accumulation, and
        is the one thing that differs between the seven folds.

        The bucket's contribution to the next bucket is a reduce over its **input** column,
        not the running column's last cell: the kernel hands a bucket's rows back in arrival
        order rather than sort order, so that cell is some arbitrary row's prefix. Reducing
        the input is order-independent, which is the property that makes the fold mergeable
        in the first place.
        """
        import numpy as np
        import pyarrow.compute as pc

        alias = fn.alias
        identity, op, ufunc = _FOLDS[fn.func]
        prior = self._fold[alias]
        if prior is not _UNSET:
            col = getattr(pc, op)(pc.fill_null(col, identity), pa.scalar(prior, col.type))
        values = wt.column(fn.input.name).drop_null()
        if values.length():
            fold = getattr(np, ufunc)
            bucket = fold.reduce(values.to_numpy(zero_copy_only=False)).item()
            self._fold[alias] = bucket if prior is _UNSET else fold(prior, bucket).item()
        return col

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
