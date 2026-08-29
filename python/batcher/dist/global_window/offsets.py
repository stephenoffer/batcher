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
  is offset through the pair (which is why `inject_window_helpers` asks the kernel for them).
* ``first_value`` -- the first bucket's first value, broadcast.

Four more are offsettable only *after* the last bucket, because what they need is a scalar
no single bucket knows: `percent_rank` and `cume_dist` divide by the relation's total row
count, `ntile` cuts the same total into tiles, and `last_value` (whose frame is the whole
partition) is the final bucket's value. Each is computed from a helper the kernel is asked for
alongside it -- a running `rank`, a running row count, a running `row_number` -- which the
ordinary per-bucket algebra offsets during the walk; `finalize` then closes them out over the
assembled result, where the total row count is simply how many rows walked past. A driver that
*assembles* its buckets (`disk`, `flight`) can do that; one that yields them as it goes
(`stream`) cannot, which is what the `assembled` argument to
`supports_ordered_bucket_offsets` asks about.

`var` and `stddev` need no second pass but no constant shift either: they combine by Chan's
parallel formula over `(count, mean, M2)`, the same one the mergeable aggregate uses, so the
kernel is asked for a running `count` and a running `avg` beside each and the triple is
reconstructed per row (`M2 == var * (count - 1)`).

`lag` / `lead` / `median` / `count_distinct` / the fills and the EWM series are still **not**
offsettable this way (each reads rows the bucket does not hold, in an order the kernel does
not return them in), so `supports_ordered_bucket_offsets` refuses them and the caller keeps
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
    "inject_window_helpers",
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

#: What each function asks the kernel to compute beside it, as `role -> window function`. The
#: helper columns ride the ordinary per-bucket algebra during the walk (each of these is a
#: running row count under a different name, so each offsets by the prior buckets' rows) and
#: are consumed and dropped by the pass that closes the function out.
#:
#: Stating the helpers as data rather than as a branch per function is what keeps the *reason*
#: each one is needed in one place. `avg` was the first and had its pair written out inline.
_HELPERS: dict[str, tuple[tuple[str, str], ...]] = {
    "avg": (("sum", "sum"), ("cnt", "count")),
    "var": (("cnt", "count"), ("avg", "avg")),
    "stddev": (("cnt", "count"), ("avg", "avg")),
    "percent_rank": (("rank", "rank"),),
    "cume_dist": (("rows", "count"),),
    "ntile": (("rn", "row_number"),),
}

#: Helper roles that are a count of *rows* rather than of the function's input, so their
#: kernel input is a literal the row count cannot be null at. `cume_dist` is the numerator of
#: "rows through my peer group over rows in the relation", which counts null-keyed rows too;
#: a `count` over the order key would silently drop them.
_ROW_COUNT_ROLES = frozenset({"rows"})

#: Helper roles taking no input at all (the ranking functions).
_RANKING_HELPERS = frozenset({"rank", "row_number"})

#: Window functions whose global value is recovered from the within-bucket value plus a
#: constant/element-wise per-bucket offset (so per-bucket compute + offset == single-node).
#: `avg` qualifies through its running `sum` and `count`, each of which is a constant shift;
#: `var`/`stddev` through the moment triple above.
_OFFSETTABLE = frozenset(
    {"row_number", "rank", "dense_rank", "count", "min", "max", "avg", *_MOMENTS, *_FOLDS}
)
#: Functions whose offset reads the *input* column out of the bucket, so the input must be a
#: plain column the kernel also emits alongside the running one.
_NEEDS_COL_INPUT = frozenset(
    {"count", "min", "max", "avg", "first_value", "last_value", *_MOMENTS, *_FOLDS}
)
_UNSET = object()


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
        if fn.frame is not None:
            return False
        allowed = fn.func in _OFFSETTABLE or fn.func == "first_value"
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


def inject_window_helpers(window: Window, win_ir: dict) -> dict[str, dict[str, str]]:
    """Append the helper running functions `_HELPERS` names to `win_ir`.

    Six of the offsettable functions are not offset from their own within-bucket value: an
    `avg` is offset through its running sum and count, a `var` through its running count and
    mean, and the three total-dividing rankings through a running rank or row count. Each such
    helper is an ordinary window function the kernel computes for free beside the one that
    needs it, under a private alias carrying a prefix that cannot collide with a user column
    (window aliases are validated against the input schema, which never holds one). Every
    helper is dropped again before any row is returned, so the output schema is unchanged.

    Args:
        window: The window whose functions need helper columns.
        win_ir: The window IR to append to. Mutated in place; the caller owns a copy.

    Returns:
        A mapping from each function's alias to its ``{role: helper alias}``, empty for a
        window whose functions all offset from their own value.
    """
    from batcher.plan.expr_ir import Lit

    helpers: dict[str, dict[str, str]] = {}
    for fn in window.functions:
        roles = _HELPERS.get(fn.func)
        if not roles:
            continue
        mine: dict[str, str] = {}
        for role, helper_fn in roles:
            alias = f"__wh_{role}::{fn.alias}"
            mine[role] = alias
            spec: dict = {"func": helper_fn, "alias": alias, "offset": 1}
            if helper_fn not in _RANKING_HELPERS:
                # A row count must not be null anywhere, so it counts a literal rather than
                # the function's input (which `cume_dist` does not even have).
                spec["input"] = Lit(1).to_ir() if role in _ROW_COUNT_ROLES else fn.input.to_ir()
            win_ir["functions"].append(spec)
        helpers[fn.alias] = mine
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

    Two of the corrections cannot be made during the walk, and both are closed out by
    `finalize` over the assembled result: `_ASSEMBLED`'s three total-dividing rankings need the
    relation's total row count, which is `_prior_rows` only once the last bucket has gone past,
    and `last_value` needs the last bucket's value. Until then their helper column carries the
    offset running value and the output column is untouched. A caller that never calls
    `finalize` must not have been given those functions in the first place, which is what
    `supports_ordered_bucket_offsets(..., assembled=...)` decides.

    Args:
        window: The window being computed, whose functions decide which offsets apply.
        helpers: The `inject_window_helpers` mapping, empty when every function offsets from
            its own within-bucket value.
    """

    def __init__(self, window: Window, helpers: dict[str, dict[str, str]]) -> None:
        self._window = window
        self._helpers = helpers
        self._prior_rows = 0
        aliases = [f.alias for f in window.functions]
        #: The last bucket-so-far's `last_value`, whose frame is the whole partition — so the
        #: value standing here when the walk ends is the relation's, and every row takes it.
        self._last: dict[str, object] = dict.fromkeys(aliases, _UNSET)
        #: Chan state per moment function: `(count, mean, M2)` over every prior bucket.
        self._moment: dict[str, tuple[int, float, float]] = dict.fromkeys(aliases, (0, 0.0, 0.0))
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
            elif fn.func in _MOMENTS:
                col = self._moment_column(wt, fn)
            elif fn.func in _ASSEMBLED:
                # Corrected by `finalize`; here the helper is carried forward instead. Each of
                # the three helpers is a running count of rows under a different name, so each
                # takes the same offset the row count does. `last_value` has no helper: its
                # bucket value is already the bucket's last, and the walk keeps the newest.
                if fn.func == "last_value":
                    if n:
                        self._last[alias] = col[0].as_py()
                else:
                    wt = self._offset_helper(wt, alias)
            elif fn.func == "first_value":
                if self._first[alias] is _UNSET:
                    self._first[alias] = col[0].as_py() if n else None
                else:
                    col = pa.array([self._first[alias]] * n, type=col.type)
            wt = wt.set_column(idx, wt.schema.field(idx), col)
        spent = self._helper_aliases(_MOMENTS | {"avg"})
        if spent:
            # Drop the private columns the one-pass offsets borrowed. The `_ASSEMBLED` helpers
            # stay: `finalize` has not run yet and they are the only record of the running
            # value it needs.
            wt = wt.select([c for c in wt.column_names if c not in spent])
        self._prior_rows += n
        return wt

    def finalize(self, table: pa.Table) -> pa.Table:
        """Close out the corrections that needed the whole relation, on the assembled result.

        Call once, after every bucket has been through `apply`, on the concatenation of what
        `apply` returned. `_prior_rows` is by then the relation's total row count, which is
        what the three total-dividing rankings divide by, and `_last` holds the final bucket's
        `last_value`. A window with none of those functions gets its table back unchanged.

        Args:
            table: Every bucket's corrected rows, concatenated in any order.

        Returns:
            The same rows with the `_ASSEMBLED` columns at their global values and every
            remaining helper column dropped.
        """
        total = self._prior_rows
        for fn in self._window.functions:
            if fn.func not in _ASSEMBLED:
                continue
            idx = table.schema.get_field_index(fn.alias)
            field = table.schema.field(idx)
            if fn.func == "last_value":
                value = self._last[fn.alias]
                col = pa.array(
                    [None if value is _UNSET else value] * table.num_rows, type=field.type
                )
            else:
                helper = table.column(self._helpers[fn.alias][_HELPERS[fn.func][0][0]])
                col = self._ranking_column(fn, helper, total)
            table = table.set_column(idx, field, col)
        left = self._helper_aliases(_ASSEMBLED)
        if left:
            table = table.select([c for c in table.column_names if c not in left])
        return table

    def _ranking_column(self, fn, helper, total: int):
        """One total-dividing ranking's global column, from its offset helper and the total."""
        import numpy as np
        import pyarrow.compute as pc

        if fn.func == "percent_rank":
            # (rank - 1) / (total - 1); a one-row relation has no spread, and SQL calls it 0.
            if total <= 1:
                return pa.array([0.0] * len(helper), type=pa.float64())
            return pc.divide(pc.cast(pc.subtract(helper, 1), pa.float64()), float(total - 1))
        if fn.func == "cume_dist":
            return pc.divide(pc.cast(helper, pa.float64()), float(total))
        # ntile: `total` rows into `fn.offset` tiles, the first `total % tiles` of them one row
        # larger — so a row's tile follows from its global row number and nothing else.
        tiles = max(1, fn.offset)
        base, rem = divmod(total, tiles)
        rows = np.asarray(helper.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        head = rem * (base + 1)
        wide = -(-rows // (base + 1))  # ceil, integer-only
        narrow = rem + -(-(rows - head) // max(base, 1))
        return pa.array(np.where(rows <= head, wide, narrow).astype(np.int64), type=pa.int64())

    def _offset_helper(self, wt: pa.Table, alias: str) -> pa.Table:
        """Shift a running-row-count helper by the prior buckets' rows, in place in `wt`."""
        import pyarrow.compute as pc

        (helper,) = self._helpers[alias].values()
        idx = wt.schema.get_field_index(helper)
        return wt.set_column(idx, wt.schema.field(idx), pc.add(wt.column(idx), self._prior_rows))

    def _helper_aliases(self, funcs) -> set[str]:
        """Every helper alias belonging to a function in `funcs`."""
        return {
            alias
            for fn in self._window.functions
            if fn.func in funcs
            for alias in self._helpers.get(fn.alias, {}).values()
        }

    def _moment_column(self, wt: pa.Table, fn):
        """Offset a running `var`/`stddev` by combining the prior buckets' moments into it.

        The kernel's running `var` at a row, with the running `count` and `avg` asked for
        beside it, *is* that row's within-bucket `(n, mean, M2)` triple — `M2 == var * (n - 1)`
        — so the global value at the row is Chan's combination of the prior buckets' triple
        with it. Writing the combination out rather than reusing the aggregate's is the one
        duplication this module cannot avoid: `bc-runtime` holds it for a whole-relation
        reduction over Arrow, and this is a per-row correction over one bucket's columns.

        A row whose within-bucket count is 0 (every input null so far in this bucket) carries
        `mean = 0` and `M2 = 0` through the formula, which leaves the prior triple untouched —
        the right answer, and the reason the fills below are 0 rather than null.
        """
        import numpy as np
        import pyarrow.compute as pc

        alias = fn.alias
        roles = self._helpers[alias]
        cnt = np.asarray(
            wt.column(roles["cnt"]).combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        mean = np.asarray(
            pc.fill_null(pc.cast(wt.column(roles["avg"]), pa.float64()), 0.0)
            .combine_chunks()
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        running = np.asarray(
            pc.fill_null(pc.cast(wt.column(alias), pa.float64()), 0.0)
            .combine_chunks()
            .to_numpy(zero_copy_only=False),
            dtype=np.float64,
        )
        # The kernel's column holds what the *function* returns, so `stddev` has to be squared
        # back into a variance before it is a moment. Reading it as one directly is a defect
        # that looks right on a bucket of one (where both are 0) and is wrong everywhere else.
        variance = running * running if fn.func == "stddev" else running
        m2 = np.where(cnt >= 2, variance * (cnt - 1.0), 0.0)

        pn, pmean, pm2 = self._moment[alias]
        total = pn + cnt
        delta = mean - pmean
        safe = np.where(total > 0, total, 1.0)
        combined = pm2 + m2 + delta * delta * pn * cnt / safe
        defined = total >= 2
        out = np.where(defined, combined / np.maximum(total - 1.0, 1.0), 0.0)
        if fn.func == "stddev":
            # Guarded: a combined M2 can land a hair below zero on cancellation, and an
            # unguarded sqrt would turn that into a NaN the mask does not cover.
            out = np.sqrt(np.maximum(out, 0.0))
        col = pa.array(out, type=pa.float64(), mask=~defined)

        self._moment[alias] = _chan(self._moment[alias], _bucket_moment(wt.column(fn.input.name)))
        return col

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
        roles = self._helpers[alias]
        sa, ca = roles["sum"], roles["cnt"]
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


def _bucket_moment(column) -> tuple[int, float, float]:
    """One bucket's `(count, mean, M2)` over its non-null inputs."""
    import numpy as np

    values = column.combine_chunks().drop_null()
    if not len(values):
        return (0, 0.0, 0.0)
    data = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)
    mean = float(data.mean())
    return (int(data.size), mean, float(((data - mean) ** 2).sum()))


def _chan(a: tuple[int, float, float], b: tuple[int, float, float]) -> tuple[int, float, float]:
    """Chan's parallel combination of two `(count, mean, M2)` triples."""
    na, ma, sa = a
    nb, mb, sb = b
    if not nb:
        return a
    if not na:
        return b
    n = na + nb
    delta = mb - ma
    return (n, ma + delta * nb / n, sa + sb + delta * delta * na * nb / n)
