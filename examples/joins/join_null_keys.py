"""Null join keys, and why they match nothing.

A null key equals nothing, including another null, so null-keyed rows never join. An inner
join drops them, a left join keeps them with a null right side, and the difference between
those two counts is exactly how many there were.

    python examples/joins/join_null_keys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    left = bt.from_pydict(
        {"id": [1, 2, 3, 4], "key": ["a", "b", None, "d"], "value": [10, 20, 30, 40]}
    )
    right = bt.from_pydict({"key": ["a", "b", None], "label": ["A", "B", "NULL-LABEL"]})

    inner = left.join(right, on="key")
    print("inner:", inner.to_pydict())
    # Two matches: the null on each side does not pair with the other.
    assert inner.count() == 2
    assert set(inner.to_pydict()["id"]) == {1, 2}

    outer_left = left.join(right, on="key", how="left")
    assert outer_left.count() == left.count()

    # The null-keyed row survives with a null label, exactly as an unmatched key does.
    kept = outer_left.filter(col("label").is_null()).to_pydict()
    print("unmatched:", kept["id"])
    assert set(kept["id"]) == {3, 4}

    # The difference between the two counts is the unmatched population.
    assert outer_left.count() - inner.count() == 2

    # An anti join finds them, and does not distinguish "null key" from "no match".
    orphans = left.join(right, on="key", how="anti")
    assert set(orphans.to_pydict()["id"]) == {3, 4}

    # If a null key should match, make it a real value first. That is a decision, and it
    # has to be made explicitly because the engine cannot make it for you.
    sentinel = bt.lit("<none>")
    filled_left = left.with_columns(key=bt.coalesce(col("key"), sentinel))
    filled_right = right.with_columns(key=bt.coalesce(col("key"), sentinel))
    matched = filled_left.join(filled_right, on="key")
    print("after filling:", matched.to_pydict()["id"])
    assert set(matched.to_pydict()["id"]) == {1, 2, 3}


if __name__ == "__main__":
    main()
