"""Tolerance-aware row correspondence, for when two sorted rowsets disagree only on floats.

`compare.to_rowset` sorts both sides on their *exact* columns first and their float columns
after, so two results with the same multiset of exact keys line up key-for-key. Inside one
run of equal exact keys, though, rows are ordered by rounded floats, and two rows whose
floats agree within tolerance can round to different grid points and swap. The positional
comparison then pairs the wrong rows and reports a mismatch on an answer that matches.

That is a matching problem, not a sorting one: the two sides agree when there is a
one-to-one pairing of each group's rows in which every pair agrees on every float column.
This module decides that exactly, with a bipartite matching per disagreeing group, and only
runs when the vectorized positional check has already failed, so the common path pays
nothing for it.

A group larger than `MAX_GROUP_ROWS` is not searched: the pairwise agreement matrix is
quadratic in the group size. Such a group is reported as a mismatch that *says* it was not
searched, rather than silently passed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["MAX_GROUP_ROWS", "group_ids", "unmatched_group"]

#: The largest run of equal exact keys the matcher searches. 512 rows is a 262,144-entry
#: agreement matrix per float column, evaluated in Arrow kernels.
MAX_GROUP_ROWS = 512

Agree = Callable[[str, pa.Array, pa.Array], pa.Array]


def group_ids(table: pa.Table, exact: Sequence[str]) -> np.ndarray:
    """The run id of every row: consecutive rows share one while their exact keys are equal.

    Null equals null here, the same rule the comparison itself applies.
    """
    n = table.num_rows
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.zeros(n, dtype=bool)
    starts[0] = True
    for name in exact:
        col = table.column(name).combine_chunks()
        prev, cur = col.slice(0, n - 1), col.slice(1)
        same = pc.fill_null(pc.equal(prev, cur), False)
        same = pc.or_(same, pc.and_(prev.is_null(), cur.is_null()))
        starts[1:] |= ~same.to_numpy(zero_copy_only=False)
    return np.cumsum(starts) - 1


def _agreement_matrix(
    ref: pa.Table, oth: pa.Table, floats: Sequence[tuple[str, str]], agree: Agree
) -> np.ndarray:
    """``m[i, j]``: does reference row ``i`` agree with candidate row ``j`` on every float?"""
    n = ref.num_rows
    left = np.repeat(np.arange(n), n)
    right = np.tile(np.arange(n), n)
    ok = np.ones(n * n, dtype=bool)
    for name, cls in floats:
        a = ref.column(name).combine_chunks().take(pa.array(left))
        b = oth.column(name).combine_chunks().take(pa.array(right))
        ok &= agree(cls, a, b).to_numpy(zero_copy_only=False)
    return ok.reshape(n, n)


def _perfect_matching(adj: np.ndarray) -> bool:
    """Kuhn's augmenting-path algorithm: does `adj` admit a perfect matching?"""
    n = adj.shape[0]
    neighbours = [np.flatnonzero(adj[i]).tolist() for i in range(n)]
    if any(not ns for ns in neighbours):
        return False
    match_of_right = [-1] * n

    def augment(i: int, seen: list[bool]) -> bool:
        # Recursion depth is bounded by the group size, which `MAX_GROUP_ROWS` keeps well
        # under the interpreter's limit.
        for j in neighbours[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_of_right[j] == -1 or augment(match_of_right[j], seen):
                match_of_right[j] = i
                return True
        return False

    return all(augment(i, [False] * n) for i in range(n))


def unmatched_group(
    ref: pa.Table,
    oth: pa.Table,
    exact: Sequence[str],
    floats: Sequence[tuple[str, str]],
    failing_rows: np.ndarray,
    agree: Agree,
) -> str | None:
    """Why no tolerance-respecting row pairing exists, or ``None`` when one does.

    Both tables must already be sorted exact-keys-first and agree exactly on every `exact`
    column, so a run of equal keys occupies the same row range on both sides.

    Args:
        ref: The reference rowset's table.
        oth: The candidate rowset's table, aligned with `ref` on the exact columns.
        exact: The exactly-compared columns, in sort order.
        floats: ``(name, class)`` of every tolerance-compared column.
        failing_rows: Row indices where the positional comparison disagreed.
        agree: The comparison's per-row agreement kernel.

    Returns:
        ``None`` when every disagreeing group has a perfect pairing, else a message naming
        the first group that has none.
    """
    groups = group_ids(ref, exact)
    for gid in np.unique(groups[failing_rows]):
        rows = np.flatnonzero(groups == gid)
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        if hi - lo > MAX_GROUP_ROWS:
            return (
                f"rows {lo}..{hi - 1}: {hi - lo} rows share one exact key, over the "
                f"{MAX_GROUP_ROWS}-row limit for tolerance-aware matching; not searched"
            )
        adj = _agreement_matrix(ref.slice(lo, hi - lo), oth.slice(lo, hi - lo), floats, agree)
        if not _perfect_matching(adj):
            return f"rows {lo}..{hi - 1}: no pairing agrees within tolerance on every column"
    return None
