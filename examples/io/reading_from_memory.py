"""Constructing a Dataset from data already in the process.

The `from_*` constructors are the boundary for data you did not read from storage. All of
them go through Arrow, so the cost is a reinterpretation where the layouts agree and a copy
where they do not.

    python examples/io/reading_from_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

import batcher as bt
from batcher import col


def main() -> None:
    columns = {"id": [1, 2, 3], "name": ["a", "b", "c"], "score": [1.5, 2.5, 3.5]}

    from_dict = bt.from_pydict(columns)
    from_rows = bt.from_pylist(
        [
            {"id": 1, "name": "a", "score": 1.5},
            {"id": 2, "name": "b", "score": 2.5},
            {"id": 3, "name": "c", "score": 3.5},
        ]
    )
    from_arrow = bt.from_arrow(pa.table(columns))
    # `from_items` builds a *single* column from a list of scalars.
    from_items = bt.from_items([10, 20, 30])

    for name, dataset in (
        ("from_pydict", from_dict),
        ("from_pylist", from_rows),
        ("from_arrow", from_arrow),
    ):
        print(f"{name:<12} {dataset.count()} rows, columns {dataset.columns}")
        assert dataset.count() == 3
        assert set(dataset.columns) == {"id", "name", "score"}

    # Column-major and row-major constructions agree.
    assert from_dict.sort("id").to_pydict() == from_rows.sort("id").to_pydict()
    assert from_dict.sort("id").to_pydict() == from_arrow.sort("id").to_pydict()

    assert from_items.count() == 3
    assert from_items.width == 1

    # `bt.range` generates without any input data at all, which is how you build a
    # calendar, a shard list, or a synthetic probe.
    generated = bt.range(0, 1_000)
    assert generated.count() == 1_000
    column = generated.columns[0]
    assert generated.agg(m=col(column).max()).to_pydict()["m"][0] == 999

    # Round-tripping back out.
    assert from_dict.to_arrow().num_rows == 3
    assert len(from_dict.to_pylist()) == 3


if __name__ == "__main__":
    main()
