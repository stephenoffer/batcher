"""Getting results out: batches, rows, slices, and the single-value cases.

Prefer ``iter_batches``: it streams and stays columnar. ``iter_rows`` exists for the cases
where you genuinely need one row at a time, and it is the slowest way to leave the engine,
so treat reaching for it as a signal that the work belonged in an expression.

    python examples/dataset/iteration.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    ds = bt.from_pydict({"id": list(range(10)), "v": [i * i for i in range(10)]})

    # The streaming boundary: Arrow batches, bounded memory.
    total_rows = 0
    for batch in ds.iter_batches(batch_size=4):
        total_rows += batch.num_rows
    assert total_rows == 10

    # Larger chunks, still Arrow.
    slices = list(ds.iter_slices(5))
    print("slices:", len(slices))
    assert len(slices) >= 2

    # Row-at-a-time, when you really need it. Rows come back as plain tuples in column
    # order, not dicts.
    rows = list(ds.iter_rows())
    print("first row:", rows[0])
    assert len(rows) == 10
    assert rows[0] == (0, 0)

    # Head/tail/limit for a peek.
    assert ds.head(3).to_pydict()["id"] == [0, 1, 2]
    assert ds.tail(2).to_pydict()["id"] == [8, 9]
    assert ds.limit(4).count() == 4
    assert ds.slice(2, 3).to_pydict()["id"] == [2, 3, 4]

    # First and last row, returned as plain tuples rather than a Dataset.
    print("first row tuple:", ds.first(), "| last:", ds.last())
    assert ds.first()[0] == 0
    assert ds.last()[0] == 9

    # A single scalar out of a one-row, one-column result.
    assert ds.select(t=col("v").sum()).item() == sum(i * i for i in range(10))

    # Top and bottom by a column, which beats sorting the whole table.
    assert ds.top_k(2, by="v").to_pydict()["id"] == [9, 8]
    assert ds.bottom_k(2, by="v").to_pydict()["id"] == [0, 1]
    assert ds.nlargest(2, "v").to_pydict()["id"] == [9, 8]
    assert ds.nsmallest(2, "v").to_pydict()["id"] == [0, 1]

    # Every nth row, and reversal.
    assert ds.gather_every(3).to_pydict()["id"] == [0, 3, 6, 9]
    assert ds.reverse().to_pydict()["id"][0] == 9

    # Python-native output, for small results only.
    assert ds.limit(2).to_pylist() == [{"id": 0, "v": 0}, {"id": 1, "v": 1}]
    assert ds.limit(2).to_dicts()[1]["v"] == 1

    # The point of all of the above: the aggregate below never leaves Rust, and is the
    # right answer whenever you were about to write a Python loop.
    looped = sum(int(v) for _id, v in ds.iter_rows())
    native = ds.select(t=col("v").sum()).item()
    assert looped == native


if __name__ == "__main__":
    main()
