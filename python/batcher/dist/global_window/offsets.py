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

`lag` is offsettable in a third way again: it reads rows a bucket does not hold, but only the
`k` immediately before it, so a **boundary exchange** of that bounded tail recovers it. That
lives in `boundary`, because unlike everything above it needs a neighbouring bucket rather than
a running scalar, and because identifying "the first `k` rows" of a bucket the kernel returns
in arrival order takes a helper of its own.

`lead` / `median` / `count_distinct` / the fills and the EWM series are still **not**
offsettable (each reads rows the bucket does not hold in a direction or an order no bounded
exchange recovers -- `lead` reads the bucket the walk has not reached yet), so
`supports_ordered_bucket_offsets` refuses them and the caller keeps the materializing kernel --
still correct, just not split.

Both consumers of this algebra live one directory up from here in spirit and one import away
in fact: `stream` runs the buckets one at a time on a single node under a memory envelope,
and `flight` runs them on different machines at the same time. They share this module rather
than each spelling the offsets out, because two statements of the same arithmetic is exactly
how a distributed result drifts from its single-node oracle.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.dist.global_window.admission import (
    _ASSEMBLED,
    _FOLDS,
    _MOMENTS,
    is_running_last_value,
    is_running_nth_value,
    supports_ordered_bucket_offsets,
    unoffsettable_functions,
)
from batcher.dist.global_window.boundary import TrailingValues, lag_across_buckets
from batcher.plan.logical import Window

__all__ = [
    "OrderedBucketOffsets",
    "bucket_order",
    "inject_window_helpers",
    "supports_ordered_bucket_offsets",
    "unoffsettable_functions",
]

#: What each function asks the kernel to compute beside it, as `role -> window function`. The
#: helper columns ride the ordinary per-bucket algebra during the walk (each of these is a
#: running row count under a different name, so each offsets by the prior buckets' rows) and
#: are consumed and dropped by the pass that closes the function out.
#:
#: Stating the helpers as data rather than as a branch per function is what keeps the *reason*
#: each one is needed in one place. `avg` was the first and had its pair written out inline.
_HELPERS: dict[str, tuple[tuple[str, str], ...]] = {
    "avg": (("sum", "sum"), ("cnt", "count")),
    # `lrn` is the bucket-LOCAL row number and is deliberately never offset: what identifies a
    # row as one of the `k` the previous bucket has to lend to is its position inside its own
    # bucket, and the kernel returns a bucket's rows in arrival order, so nothing else does.
    "lag": (("lrn", "row_number"),),
    "var": (("cnt", "count"), ("avg", "avg")),
    "stddev": (("cnt", "count"), ("avg", "avg")),
    "percent_rank": (("rank", "rank"),),
    "cume_dist": (("rows", "count"),),
    "ntile": (("rn", "row_number"),),
    # Both bucket-LOCAL and never offset: `rows` is how far this row's frame reaches into the
    # bucket (peer-inclusive, as the running frame is), `lrn` names the bucket's i-th row.
    "nth_value": (("rows", "count"), ("lrn", "row_number")),
}

#: Helper roles that are a count of *rows* rather than of the function's input, so their
#: kernel input is a literal the row count cannot be null at. `cume_dist` is the numerator of
#: "rows through my peer group over rows in the relation", which counts null-keyed rows too;
#: a `count` over the order key would silently drop them.
_ROW_COUNT_ROLES = frozenset({"rows"})

#: Helper roles taking no input at all (the ranking functions).
_RANKING_HELPERS = frozenset({"rank", "row_number"})

_UNSET = object()


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
        #: The rolling boundary tail each `lag` reads back across a cut (`boundary`).
        self._trailing = {
            f.alias: TrailingValues(f.offset) for f in window.functions if f.func == "lag"
        }
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
        #: The first `k` input values the walk has seen, in order, per running `nth_value`.
        self._head: dict[str, list[object]] = {a: [] for a in aliases}

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
            elif fn.func == "lag":
                col = lag_across_buckets(
                    wt,
                    col,
                    self._helpers[alias]["lrn"],
                    fn.input.name,
                    self._trailing[alias],
                )
            elif is_running_nth_value(fn):
                col = self._nth_value_column(wt, fn, col.type)
            elif is_running_last_value(fn):
                # Nothing to carry: this row's peer group is whole in this bucket, so the
                # kernel's per-bucket answer is already the global one.
                pass
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
                    # An empty bucket holds no first value, so it must leave the carry unset
                    # rather than record NULL for every later bucket to broadcast.
                    if n:
                        self._first[alias] = col[0].as_py()
                else:
                    col = pa.array([self._first[alias]] * n, type=col.type)
            wt = wt.set_column(idx, wt.schema.field(idx), col)
        spent = self._helper_aliases(_MOMENTS | {"avg", "lag", "nth_value"})
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
            if fn.func not in _ASSEMBLED or is_running_last_value(fn):
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

    def _nth_value_column(self, wt: pa.Table, fn, typ: pa.DataType):
        """One bucket's running `nth_value`, from the carried head and the bucket-local helpers.

        A row is NULL while its frame -- the prior buckets' rows plus this bucket's through its
        peer group -- holds fewer than `k` rows, and the relation's k-th value after. While
        the prior buckets hold fewer than `k` rows they are all in the head, so topping it up
        from this bucket's first rows (in bucket-local order) makes its k-th entry the
        relation's k-th row, whichever bucket that row is in.
        """
        import numpy as np
        import pyarrow.compute as pc

        k = max(1, fn.offset)
        head = self._head[fn.alias]
        roles = self._helpers[fn.alias]
        rows = np.asarray(
            wt.column(roles["rows"]).combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        lrn = np.asarray(
            wt.column(roles["lrn"]).combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        values = wt.column(fn.input.name).combine_chunks()
        need = k - len(head)
        if need > 0:
            first = np.flatnonzero(lrn <= need)
            head.extend(values.take(pa.array(first[np.argsort(lrn[first])])).to_pylist())
        if len(head) < k:
            return pa.nulls(wt.num_rows, type=typ)
        reached = pa.array((rows + self._prior_rows) >= k, type=pa.bool_())
        return pc.if_else(reached, pa.scalar(head[k - 1], type=typ), pa.scalar(None, type=typ))

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
        # Clamp below at zero while letting a NaN through, because that is what the kernel
        # does: `bc_runtime::window::agg::Moments::variance` clamps with a comparison, so a
        # variance poisoned by a NaN input is NaN (Polars' `rolling_var` answer, and the
        # `GROUP BY` variance's). It used to clamp with `f64::max(_, 0.0)`, which returns the
        # non-NaN operand and answered `0.0`; this translation matched that, and follows the
        # kernel's fix rather than improving on it — `bc-interp` is the oracle here.
        # `np.where` on the comparison keeps NaN, where `np.maximum` would too but reads as the
        # thing the old clamp was not.
        out = np.where(out < 0.0, 0.0, out)
        if fn.func == "stddev":
            out = np.sqrt(out)
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
        """Offset a running `min`/`max` against the prior buckets' running extreme.

        The kernel orders floats by a **total** order in which NaN is the greatest value, so a
        running `max` that has seen a NaN stays NaN for the rest of the partition while a
        running `min` never picks one up. Arrow's `max`/`max_element_wise` instead *skip* NaN,
        which is the same answer on every input that has none and a different one on every
        input that does: measured over 40,000 rows with a NaN every 23rd, the split path
        returned a real number for 32,229 of the rows the kernel gave NaN. `_nan_aware` is
        that difference and nothing else; `min` needs no such care, because a value that is
        greatest can never win a minimum.
        """
        alias = fn.alias
        if state[alias] is not None:
            col = self._nan_aware(col, state[alias], fn, element_wise)
        bucket = self._bucket_extreme(wt.column(fn.input.name), fn, reduce_fn)
        if bucket is not None:
            state[alias] = bucket if state[alias] is None else pick(state[alias], bucket)
        return col

    @staticmethod
    def _nan_aware(col, prior, fn, element_wise):
        """`element_wise(col, prior)` with the kernel's NaN-is-greatest order for `max`."""
        import pyarrow.compute as pc

        scalar = pa.scalar(prior, col.type)
        if fn.func != "max" or not pa.types.is_floating(col.type):
            return element_wise(col, scalar)
        if isinstance(prior, float) and prior != prior:
            # A NaN already carried in is the greatest value there is, so it wins every row —
            # including the rows where this bucket's own running max is still null.
            return pa.array([prior] * len(col), type=col.type)
        # `is_nan` is null where the running value is (no non-null input through that row yet),
        # and a null condition selects neither arm, so it is filled before it is a condition.
        keep = pc.fill_null(pc.is_nan(col), False)
        return pc.if_else(keep, col, element_wise(col, scalar))

    @staticmethod
    def _bucket_extreme(column, fn, reduce_fn):
        """This bucket's contribution to the running extreme, under the same total order."""
        import pyarrow.compute as pc

        if fn.func == "max" and pa.types.is_floating(column.type):
            has_nan = pc.any(pc.fill_null(pc.is_nan(column), False)).as_py()
            if has_nan:
                return float("nan")
        return reduce_fn(column).as_py()

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
