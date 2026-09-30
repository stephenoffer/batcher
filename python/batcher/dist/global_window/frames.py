"""The explicit ``ROWS`` frames the ordered-bucket algebra carries, and the correction for them.

`offsets` was written for the default frame, and an explicit frame on a global window was
refused outright -- which refused the two frames every cumulative and rolling helper builds:
`cum_sum` and its siblings are ``ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW``, and
`rolling_sum(n)` is ``ROWS BETWEEN n - 1 PRECEDING AND CURRENT ROW``. A distributed global
window over either raised `PlanError`, and a spilled or streamed one materialized.

Both are recoverable, for two different reasons.

**The running ``ROWS`` frame** differs from the default ``RANGE`` frame only on peers: a
``RANGE`` running value covers the current row's whole peer group, a ``ROWS`` one stops at
the row. The range partitioner never splits a peer group across a cut, so every prior bucket
holds rows strictly *before* the current one under either frame, and the prior buckets'
contribution is the same accumulated scalar the default frame is offset by. The existing
per-function correction applies unchanged; only the helper columns (`avg`'s running sum and
count, `var`'s count and mean) must be computed over the same frame, which is what
`frame_for_helpers` says.

**The trailing ``ROWS`` frame** (``p PRECEDING`` to ``CURRENT ROW``) reads at most `p` rows
back, so like `lag` it needs a **boundary exchange** rather than a running scalar: the last
`p` input values of the prior buckets, in window order. A row at bucket-local position `r`
has a frame reaching `p - r + 1` rows into that tail when `r <= p`, and none at all past it,
so only a bucket's first `p` rows are corrected, each by folding a *suffix* of the tail into
the kernel's truncated within-bucket value. The exchange is bounded by the frame, not the
data, and the fold over it is vectorized over the tail's `p` values.

That frame is corrected for `sum`, `count`, `min`, `max` and `avg` over numeric inputs (and
`count` over any input). Everything else a frame can bound -- a ``FOLLOWING`` edge, which
reads the bucket the walk has not reached, the value-based ``RANGE`` offsets, the moments --
stays declined, and the caller keeps the materializing kernel.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.dist.global_window.boundary import TrailingValues, _tail_in_order

__all__ = [
    "TRAILING_FUNCS",
    "TrailingFrame",
    "frame_for_helpers",
    "is_rows_running_frame",
    "trailing_rows_distance",
]

#: The running aggregates whose ``ROWS UNBOUNDED PRECEDING AND CURRENT ROW`` frame takes the
#: default frame's correction as is. Stated here rather than derived from `admission`'s tables
#: so this module does not import the one that imports it; `admission` checks each entry is
#: one of its own offsettable functions (`test_rows_running_funcs_are_offsettable`).
ROWS_RUNNING_FUNCS = frozenset(
    {"sum", "count", "min", "max", "avg", "var", "stddev"}
    | {"bit_and", "bit_or", "bit_xor", "bool_and", "bool_or"}
)

#: The aggregates a bounded trailing ``ROWS`` frame is corrected for by `TrailingFrame`.
TRAILING_FUNCS = frozenset({"sum", "count", "min", "max", "avg"})


def is_rows_running_frame(fn) -> bool:
    """Whether `fn` is a running aggregate over ``ROWS UNBOUNDED PRECEDING .. CURRENT ROW``.

    Args:
        fn: A `WindowFuncSpec`.

    Returns:
        True for the frame `cum_sum` and its siblings build, on a function in
        `ROWS_RUNNING_FUNCS`.
    """
    frame = getattr(fn, "frame", None)
    if frame is None or fn.func not in ROWS_RUNNING_FUNCS:
        return False
    return (frame.start, frame.end, frame.units) == (None, 0, "rows")


def trailing_rows_distance(fn) -> int | None:
    """How many rows back `fn`'s trailing ``ROWS`` frame reaches, or None if it is not one.

    Args:
        fn: A `WindowFuncSpec`.

    Returns:
        `p` for ``ROWS BETWEEN p PRECEDING AND CURRENT ROW`` on a function in
        `TRAILING_FUNCS`; None for every other frame, including ``0 PRECEDING`` (which is
        ``CURRENT ROW`` and reads no prior bucket at all, so the kernel's value stands).
    """
    frame = getattr(fn, "frame", None)
    if frame is None or fn.func not in TRAILING_FUNCS or frame.units != "rows":
        return None
    if frame.end != 0 or frame.start is None or frame.start >= 0:
        return None
    return -frame.start


def frame_for_helpers(fn):
    """The frame a helper column must be computed over for `fn`'s correction to be exact.

    The helpers `offsets.inject_window_helpers` asks for (`avg`'s sum and count, the moments'
    count and mean) are what the correction reconstructs the function from, so they must span
    exactly the rows the function does. Under the default frame they carry none; under one of
    this module's frames they carry `fn`'s. The ranking helpers take no frame at all.

    Args:
        fn: A `WindowFuncSpec`.

    Returns:
        `fn.frame` when it is one of this module's frames, else None.
    """
    if is_rows_running_frame(fn) or trailing_rows_distance(fn) is not None:
        return fn.frame
    return None


class TrailingFrame:
    """The boundary exchange for one trailing-framed aggregate, and the fold that reads it.

    Stateful and order-dependent by contract, exactly as `TrailingValues` is: `correct` one
    bucket at a time, in `bucket_order`.

    Args:
        fn: The trailing-framed `WindowFuncSpec`.
    """

    def __init__(self, fn) -> None:
        self._fn = fn
        self._p = trailing_rows_distance(fn)
        self._tail = TrailingValues(self._p)

    def correct(self, wt: pa.Table, col, helpers: dict[str, str]):
        """One bucket's column, with its first `p` rows extended into the prior buckets.

        Args:
            wt: The bucket's windowed rows, in the kernel's arrival order.
            col: The kernel's within-bucket value, framed but truncated at the bucket start.
            helpers: This function's `{role: helper alias}`; `lrn` is the bucket-local
                `row_number`, and an `avg` also has its framed `sum` and `cnt`.

        Returns:
            The column at its global value. The tail is advanced past this bucket.
        """
        import numpy as np

        name = self._fn.input.name
        ranks = np.asarray(
            wt.column(helpers["lrn"]).combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        head = ranks <= self._p
        out = col
        if head.any():
            prior = self._suffixes(wt.schema.field(name).type)
            # Row at local rank r reads the last p - r + 1 tail values: suffix index p - r.
            index = np.where(head, self._p - ranks, 0)
            out = self._combine(wt, col, helpers, head, index, prior)
        self._tail.absorb(
            _tail_in_order(wt, wt.column(helpers["lrn"]), helpers["lrn"], name, self._p)
        )
        return out

    def _suffixes(self, dtype: pa.DataType) -> dict[str, object]:
        """Folds over every suffix of the tail, indexed by how many tail rows the frame needs.

        Entry `i` covers the last `i + 1` values of the tail (clamped to what the tail holds,
        which is fewer than `p` only at the very start of the relation), so a row whose frame
        needs `s` prior rows reads entry `s - 1`.
        """
        import numpy as np

        values = self._tail.values()
        present = np.array([v is not None for v in values[::-1]], dtype=bool)
        integral = pa.types.is_integer(dtype) or pa.types.is_boolean(dtype)
        kind = np.int64 if integral else np.float64
        numeric = pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
        rev = (
            np.array([0 if v is None else v for v in values[::-1]], dtype=kind)
            if numeric
            else np.zeros(len(values), dtype=np.int64)
        )
        size = self._p
        held = len(values)

        def padded(acc: np.ndarray, fill) -> np.ndarray:
            # Suffixes longer than the tail are the whole tail: repeat its last fold.
            if held == 0:
                return np.full(size, fill, dtype=acc.dtype if acc.size else kind)
            return np.concatenate([acc, np.full(size - held, acc[-1], dtype=acc.dtype)])

        counts = padded(np.cumsum(present.astype(np.int64)), 0)
        folds: dict[str, object] = {"count": counts}
        if numeric:
            folds["sum"] = padded(np.cumsum(np.where(present, rev, 0)), 0)
            if integral:
                hi, lo = np.iinfo(np.int64).max, np.iinfo(np.int64).min
                folds["min"] = padded(np.minimum.accumulate(np.where(present, rev, hi)), hi)
                folds["max"] = padded(np.maximum.accumulate(np.where(present, rev, lo)), lo)
            else:
                # The kernel's float order makes NaN the greatest value: a `max` that has seen
                # one is NaN (`np.maximum` propagates it) and a `min` never picks one up while a
                # real value exists (`np.fmin` skips it) -- `offsets._extreme` states the same.
                folds["min"] = padded(np.fmin.accumulate(np.where(present, rev, np.inf)), np.inf)
                folds["max"] = padded(
                    np.maximum.accumulate(np.where(present, rev, -np.inf)), -np.inf
                )
        return folds

    def _combine(self, wt, col, helpers, head, index, prior):
        """Fold the prior rows' suffix into the kernel's value on the bucket's head rows."""
        import numpy as np
        import pyarrow.compute as pc

        func = self._fn.func
        seen = prior["count"][index] > 0
        extend = head & seen
        if func == "count":
            base = np.asarray(col.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
            total = base + np.where(head, prior["count"][index], 0)
            return pa.array(total, type=col.type)
        if func == "avg":
            ksum = pc.fill_null(pc.cast(wt.column(helpers["sum"]), pa.float64()), 0.0)
            ksum = np.asarray(ksum.combine_chunks().to_numpy(zero_copy_only=False))
            kcnt = np.asarray(
                wt.column(helpers["cnt"]).combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            tsum = ksum + np.where(extend, prior["sum"][index].astype(np.float64), 0.0)
            tcnt = kcnt + np.where(extend, prior["count"][index], 0)
            value = tsum / np.maximum(tcnt, 1)
            fixed = pa.array(value, type=pa.float64(), mask=tcnt == 0)
            return pc.if_else(pa.array(head), fixed, pc.cast(col, pa.float64()))
        kernel = col.combine_chunks()
        valid = np.asarray(kernel.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
        kind = np.int64 if pa.types.is_integer(kernel.type) else np.float64
        base = np.asarray(pc.fill_null(kernel, 0).to_numpy(zero_copy_only=False), dtype=kind)
        extra = prior[func][index].astype(kind)
        if func == "sum":
            merged = np.where(valid, base + extra, extra)
        elif func == "min":
            merged = np.where(
                valid,
                np.fmin(base, extra) if kind is np.float64 else np.minimum(base, extra),
                extra,
            )
        else:
            merged = np.where(valid, np.maximum(base, extra), extra)
        value = np.where(extend, merged, base)
        return pa.array(value, type=kernel.type, mask=~(valid | extend))
