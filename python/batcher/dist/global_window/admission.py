"""Which global windows the ordered-bucket algebra covers, and why the rest are refused.

`offsets` holds the walk that corrects each bucket; this module holds the admission half it
rests on: the function families that algebra has an offset for (`_FOLDS`, `_MOMENTS`,
`_ASSEMBLED`, `_OFFSETTABLE`), the two running frames it carries (`is_running_last_value`,
`is_running_nth_value`), and the predicates that classify a window against them. They are
split from the walk because every caller that only *routes* -- `dist.executor` deciding
whether a global window has a distributed path -- needs this half and never the other.
"""

from __future__ import annotations

from batcher.plan.expr_ir import Col
from batcher.plan.logical import Window

__all__ = [
    "is_running_last_value",
    "is_running_nth_value",
    "supports_ordered_bucket_offsets",
    "unoffsettable_functions",
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

#: The running *moments*. Neither `var` nor `stddev` is a constant shift, and neither is
#: recoverable from its own within-bucket value alone -- but a bucket's `(count, mean, M2)`
#: triple combines with the prior buckets' by **Chan's parallel formula**, the same one
#: `bc-runtime`'s mergeable variance uses, so the same arithmetic that makes `var` distributable
#: as an *aggregate* makes it offsettable as a running *window*. The kernel is asked for a
#: running `count` and a running `avg` beside each one, which with the running `var` is exactly
#: that triple: `M2 == var * (count - 1)`.
_MOMENTS = frozenset({"var", "stddev"})

#: Window functions whose global value is not determined until the last bucket has been
#: walked, because what they need is a scalar no bucket knows on its own: the relation's total
#: row count (which is simply how many rows the walk has seen once it ends), or the final
#: bucket's last value. They are corrected by `finalize` over the assembled result rather than
#: bucket by bucket, so a driver that yields buckets as it goes cannot carry them -- see the
#: `assembled` argument to `supports_ordered_bucket_offsets`.
_ASSEMBLED = frozenset({"percent_rank", "cume_dist", "ntile", "last_value"})


def is_running_last_value(fn) -> bool:
    """Whether `fn` is a `last_value` over the *default* frame, which needs no correction.

    `_ASSEMBLED` carries `last_value` on the premise -- written into this module's opening
    paragraph -- that its frame is the whole partition, "the final bucket's value", which is
    what `finalize` substitutes into every row. That is `last_value(v) OVER (... ROWS BETWEEN
    UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)`, and it is **not** what the API builds:
    `("last_value", col("v"))` carries `WindowFrame(None, 0, "range")` -- UNBOUNDED PRECEDING
    TO CURRENT ROW -- whose value at each row is that row's *peer group's* last, not the
    relation's.

    Those are different answers, and `finalize` applied to the second would overwrite every
    row with the relation's last value. The blanket `fn.frame is not None` refusal beside it
    is what kept that from happening, so `last_value` has been unreachable on this route
    since `aad9d59a` added it -- the entry and the guard that hides it landed together.

    The default frame needs no offset and no finalize *at all*: the range partitioner puts
    equal keys in one bucket, as this module's opening paragraph states, so a row's peer
    group is whole inside its own bucket and the per-bucket `last_value` is already the
    global one. It is admitted here and excluded from both correction paths, rather than
    corrected by an algebra written for the other function.
    """
    frame = getattr(fn, "frame", None)
    if frame is None or fn.func != "last_value":
        return False
    return (frame.start, frame.end, frame.units) == (None, 0, "range")


def is_running_nth_value(fn) -> bool:
    """Whether `fn` is an `nth_value` over the *default* running frame, respecting nulls.

    Over ``RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`` the value at a row is the
    relation's k-th row in order once the frame (the rows through this row's peer group)
    holds at least k rows, and NULL before that -- one global value, reached at one point in
    the order. So the walk carries the first k input values it has seen: a bucket whose prior
    buckets already hold k rows takes the k-th from the carry, and one that does not takes its
    own ``(k - prior)``-th row, located by a bucket-local `row_number` helper because the
    kernel returns a bucket's rows in arrival order. `ignore_nulls` is excluded: it skips
    rows, so "the k-th row" is no longer a position the helper can name.
    """
    frame = getattr(fn, "frame", None)
    if frame is None or fn.func != "nth_value" or getattr(fn, "ignore_nulls", False):
        return False
    return (frame.start, frame.end, frame.units) == (None, 0, "range")


def _is_running_first_value(fn) -> bool:
    """Whether `fn` is a `first_value` over the default running frame, respecting nulls.

    Kyber rewrites `nth_value(x, 1)` to `first_value(x)` and keeps the frame, so this is the
    shape `nth_value`'s commonest spelling arrives in. Every running frame starts at the
    relation's first row, so its value is that row's everywhere -- which is what the
    frameless `first_value` branch of the walk already broadcasts.
    """
    frame = getattr(fn, "frame", None)
    if frame is None or fn.func != "first_value" or getattr(fn, "ignore_nulls", False):
        return False
    return (frame.start, frame.end, frame.units) == (None, 0, "range")


def _running_frame_ok(fn) -> bool:
    """Whether `fn`'s explicit frame is one of the running frames this algebra carries."""
    return is_running_last_value(fn) or is_running_nth_value(fn) or _is_running_first_value(fn)


#: Window functions whose global value is recovered from the within-bucket value plus a
#: constant/element-wise per-bucket offset (so per-bucket compute + offset == single-node).
#: `avg` qualifies through its running `sum` and `count`, each of which is a constant shift;
#: `var`/`stddev` through the moment triple above.
_OFFSETTABLE = frozenset(
    {"row_number", "rank", "dense_rank", "count", "min", "max", "avg", "lag", *_MOMENTS, *_FOLDS}
)
#: Functions whose offset reads the *input* column out of the bucket, so the input must be a
#: plain column the kernel also emits alongside the running one.
_NEEDS_COL_INPUT = frozenset(
    {
        "count",
        "min",
        "max",
        "avg",
        "first_value",
        "last_value",
        "nth_value",
        "lag",
        *_MOMENTS,
        *_FOLDS,
    }
)


def supports_ordered_bucket_offsets(window: Window, *, assembled: bool = False) -> bool:
    """Whether `window` is a global window the ordered-bucket-offset algebra covers.

    Requires: no partition keys (global); a **leading** order key that is a plain column (the
    column the range partitioner cuts on) *of a type that partitioner can cut*; every function
    offsettable or `first_value`, with no explicit frame; and aggregate/`first_value` inputs
    are plain columns.

    `assembled` says the caller will hold every bucket's corrected rows before it returns any
    of them, and will call `OrderedBucketOffsets.finalize` on the assembly. That admits the
    four functions of `_ASSEMBLED`, whose global value depends on the relation's total row
    count (or on its last row) and so is not known while the walk is still running. The
    distributed drivers concatenate and can pass it; the single-node streaming driver yields
    as it goes and cannot. Defaulting it to False is what keeps a caller that has not been
    taught to finalize from silently emitting a `percent_rank` divided by a bucket count.

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
        if fn.frame is not None and not _running_frame_ok(fn):
            return False
        allowed = fn.func in _OFFSETTABLE or fn.func == "first_value" or _running_frame_ok(fn)
        if not allowed and not (assembled and fn.func in _ASSEMBLED):
            return False
        if fn.func in _NEEDS_COL_INPUT and not isinstance(fn.input, Col):
            return False
    return True


def unoffsettable_functions(window: Window, *, assembled: bool = False) -> list[str]:
    """The functions in `window` this algebra has no offset for, for an error message.

    `supports_ordered_bucket_offsets` answers *whether*; this answers *which*, from the same
    tables, so a refusal names the function at fault rather than every function present. The
    message that used to be built at the call site listed them all — a `row_number` beside a
    `lag` was reported as equally unsupported, which sends the reader to rewrite the wrong half
    of their query.

    Args:
        window: The global window that found no distributed route.
        assembled: As for `supports_ordered_bucket_offsets` — whether the caller finalizes.

    Returns:
        The offending function names, sorted and deduplicated. Empty when the shape is
        supported and something else (the order key, the sources) is what declined.
    """
    bad = set()
    for fn in window.functions:
        if _running_frame_ok(fn):
            continue
        if fn.frame is not None:
            bad.add(fn.func)
        elif fn.func in _OFFSETTABLE or fn.func == "first_value":
            continue
        elif not (assembled and fn.func in _ASSEMBLED):
            bad.add(fn.func)
    return sorted(bad)


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
