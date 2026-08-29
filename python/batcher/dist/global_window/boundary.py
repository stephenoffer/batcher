"""The one ordered-bucket correction that reads a *neighbouring* bucket's rows.

Every other correction in `offsets` is a running scalar: a bucket is handed what the prior
buckets accumulated and needs nothing else from them. `lag` is different. A row `k` positions
into a bucket has its source row `k` positions back, which for the bucket's first `k` rows is
in the bucket *before* it — so the window kernel, run on the bucket alone, returns NULL there
where the global answer is a real value.

That is recoverable with a **boundary exchange**, and a small one: only the last `k` values of
the running order ever cross a cut, whatever the bucket holds. This module is that exchange —
a rolling tail of `k` values carried from bucket to bucket, and the arithmetic that reads a
bucket's first `k` rows out of it.

The complication, and the reason this is not two lines inside `offsets`, is that the kernel
returns a bucket's rows in **arrival** order rather than sort order, so "the first `k` rows"
is not a slice. It is recovered from a `row_number` the kernel is asked for alongside the
`lag` — the bucket-local rank, deliberately *not* offset by the prior buckets' rows the way
the total-dividing rankings' helpers are, because what identifies a boundary row is its
position within its own bucket.

`lead` is the mirror image and is **not** here: it reads the *next* bucket, which the walk has
not seen, so it needs a held-back bucket and a second pass rather than a rolling tail. It stays
declined, and the declining is what keeps a `lead` from being computed per bucket and returning
NULLs at every cut.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["TrailingValues", "lag_across_buckets"]


class TrailingValues:
    """The last `k` input values seen, in the window's own order — what a `lag` reads back.

    Stateful and order-dependent by contract, exactly as `OrderedBucketOffsets` is: feed it the
    buckets in `bucket_order`, once each. It holds at most `k` values however many buckets have
    gone past, so the exchange is bounded by the lag distance and not by the data.

    Args:
        distance: The `lag` distance `k` (`WindowFuncSpec.offset`).
    """

    def __init__(self, distance: int) -> None:
        #: The lag distance. Public because `lag_across_buckets` is the other half of this
        #: one algorithm and needs it to size the head it builds; a property would be a
        #: ceremony around a number that is set once and never changes.
        self.distance = max(1, int(distance))
        self._values: list = []

    def head_replacements(self) -> list:
        """The global `lag` value for a row at bucket-local rank `1..k`, in that order.

        A row at local rank `r` has global rank `prior + r` and reads global position
        `prior + r - k`. The tail's last element *is* global position `prior`, so the value
        sits `k - r` places before its end — index `held - 1 - (k - r)`. Where fewer than
        `k - r + 1` values have gone past (the very start of the relation) the answer is NULL,
        which is what the single-node kernel returns there too.

        Returns:
            A list of `k` values (or `None`s), indexed by local rank minus one.
        """
        held = len(self._values)
        return [
            self._values[held - 1 - (self.distance - r)] if held >= self.distance - r + 1 else None
            for r in range(1, self.distance + 1)
        ]

    def absorb(self, values: list) -> None:
        """Take this bucket's input values, in the window's order, and keep the last `k`.

        Args:
            values: The bucket's values in order; only its last `k` can matter to any later
                bucket, so the rest are dropped immediately.
        """
        self._values.extend(values)
        if len(self._values) > self.distance:
            del self._values[: len(self._values) - self.distance]


def lag_across_buckets(wt: pa.Table, column, rank_alias: str, input_name: str, trailing):
    """One bucket's `lag` column, corrected to its global value.

    Args:
        wt: The bucket's windowed rows, in the kernel's arrival order.
        column: The kernel's within-bucket `lag` column, NULL for the first `k` rows.
        rank_alias: The helper column holding each row's bucket-local `row_number`.
        input_name: The `lag`'s input column, whose boundary values the next bucket reads.
        trailing: The `TrailingValues` carrying the prior buckets' tail. Advanced here.

    Returns:
        The `lag` column with its first `k` rows filled from the prior buckets.
    """
    import pyarrow.compute as pc

    ranks = wt.column(rank_alias)
    k = trailing.distance
    head = pa.array(trailing.head_replacements(), type=column.type)
    # Every row is given an index into `head`, clamped so the ones that will not be selected
    # still address a valid slot; `if_else` then keeps the kernel's own value for them.
    index = pc.min_element_wise(pc.subtract(ranks, 1), k - 1)
    corrected = pc.if_else(pc.less_equal(ranks, k), pc.take(head, index), column)
    trailing.absorb(_tail_in_order(wt, ranks, rank_alias, input_name, k))
    return corrected


def _tail_in_order(wt: pa.Table, ranks, rank_alias: str, input_name: str, k: int) -> list:
    """This bucket's last `k` input values, in the window's order.

    The filter is an Arrow op over the whole bucket and the sort is over the `k` rows that
    survive it, so the per-row work stays in Arrow and only a bounded tail is ever ordered in
    Python — the control plane does not touch a row it does not have to.
    """
    import pyarrow.compute as pc

    if not wt.num_rows:
        return []
    cutoff = wt.num_rows - k
    keep = wt.select([rank_alias, input_name])
    if cutoff > 0:
        keep = keep.filter(pc.greater(ranks, cutoff))
    return keep.sort_by([(rank_alias, "ascending")]).column(input_name).to_pylist()
