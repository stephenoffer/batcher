"""Sorting and ranking, including the edge cases that hide bugs.

Sort order is where nulls, ties, and descending flags interact badly. Decide explicitly
where nulls go and how ties break, because the default is rarely what a report wants and
the difference is invisible until someone checks a boundary row.

    python examples/expressions/sorting_and_ranking.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    scores = bt.from_pydict(
        {
            "name": ["a", "b", "c", "d", "e"],
            "score": [10, 30, 20, 30, None],
        }
    )

    # Ascending by default.
    asc = scores.sort("score").to_pydict()
    print("ascending:", asc["name"], asc["score"])
    assert asc["score"][:4] == [10, 20, 30, 30]

    # Descending.
    desc = scores.sort("score", descending=True).to_pydict()
    assert desc["score"][0] == 30

    # Where the null goes is a choice, not an accident.
    nulls_last = scores.sort("score", nulls_first=False).to_pydict()
    assert nulls_last["score"][-1] is None
    nulls_first = scores.sort("score", nulls_first=True).to_pydict()
    assert nulls_first["score"][0] is None

    # A tie-break column makes the order deterministic. Without one, rows `b` and `d`
    # could come back either way, and a "stable" result today is not a guarantee.
    stable = scores.sort("score", "name", descending=[True, False]).to_pydict()
    print("tie-broken:", stable["name"], stable["score"])
    assert stable["name"][:2] == ["b", "d"]

    # Ranking. `min` gives tied rows the same rank and skips the next; `dense` does not.
    ranked = (
        scores.with_columns(
            min_rank=col("score").rank(method="min", descending=True),
            dense_rank=col("score").rank(method="dense", descending=True),
        )
        .sort("name")
        .to_pydict()
    )
    print(ranked)
    # b and d tie at 30, so both are rank 1 and the next distinct value is rank 3.
    by_name = dict(zip(ranked["name"], ranked["min_rank"], strict=True))
    assert by_name["b"] == by_name["d"] == 1
    assert by_name["c"] == 3
    dense_by_name = dict(zip(ranked["name"], ranked["dense_rank"], strict=True))
    assert dense_by_name["c"] == 2

    # Top-N without ordering the whole table: this is a heap, not a sort.
    top2 = scores.top_k(2, by="score").to_pydict()
    print("top 2:", top2["name"])
    assert sorted(top2["score"]) == [30, 30]

    # Sort knowledge is tracked, so a redundant re-sort can be skipped.
    ordered = scores.sort("score")
    assert ordered.meta.is_known_sorted_by("score")

    # Sorting is not order-independent -- assert on the sequence, not on a set, or the
    # test cannot fail when the sort breaks.
    seq = scores.drop_nulls(["score"]).sort("score").to_pydict()["score"]
    assert seq == sorted(seq)


if __name__ == "__main__":
    main()
