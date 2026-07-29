"""Transforming inside a list column, without exploding it first.

``explode`` then ``group_by`` re-collects is the expensive way to map over list elements.
``.list.transform`` and ``.list.filter`` do it in place, which keeps the row count fixed
and avoids the shuffle a regroup would cost. Both take an *expression* over
``bt.element()`` -- the current element -- not a Python lambda, so the body runs in Rust.

    python examples/expressions/lists_transforms.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    baskets = bt.from_pydict(
        {
            "user": ["a", "b"],
            "prices": [[10.0, 20.0, 30.0], [5.0, None, 15.0]],
        }
    )

    shaped = baskets.with_columns(
        # Map over every element.
        with_tax=col("prices").list.transform(bt.element() * 1.2),
        # Keep only the elements matching a predicate.
        expensive=col("prices").list.filter(bt.element() > 10.0),
        # Drop the nulls that the source left behind.
        clean=col("prices").list.drop_nulls(),
        # Reduce, after cleaning.
        total=col("prices").list.drop_nulls().list.sum(),
        n_valid=col("prices").list.drop_nulls().list.len(),
    ).to_pydict()

    print(shaped)

    assert shaped["with_tax"][0] == [12.0, 24.0, 36.0]
    assert shaped["expensive"][0] == [20.0, 30.0]
    assert shaped["clean"][1] == [5.0, 15.0]
    assert shaped["total"] == [60.0, 20.0]
    assert shaped["n_valid"] == [3, 2]
    # The row count is unchanged: no explode, no regroup.
    assert len(shaped["user"]) == 2

    # Vector shaping, for embeddings held as list columns.
    vectors = bt.from_pydict({"v": [[3.0, 4.0], [1.0, 0.0]]})
    normed = vectors.with_columns(
        unit=col("v").list.normalize(),
        soft=col("v").list.softmax(),
        pooled_mean=col("v").list.mean_pool(),
        pooled_max=col("v").list.max_pool(),
    ).to_pydict()
    print(normed)

    # [3, 4] has length 5, so the unit vector is [0.6, 0.8].
    assert [round(x, 6) for x in normed["unit"][0]] == [0.6, 0.8]
    # Softmax sums to one.
    assert abs(sum(normed["soft"][0]) - 1.0) < 1e-9
    assert normed["pooled_max"][0] == 4.0

    # `flatten` collapses one level of nesting, so it wants a list *of lists*.
    nested = bt.from_pydict({"n": [[[1, 2], [3]], [[4]]]})
    flat = nested.select(f=col("n").list.flatten()).to_pydict()
    print("flattened:", flat)
    assert flat["f"][0] == [1, 2, 3]

    # Sorting and de-duplicating inside a row.
    tags = bt.from_pydict({"t": [["b", "a", "b", "c"]]})
    tidy = tags.with_columns(
        sorted_tags=col("t").list.sort(),
        unique_tags=col("t").list.unique(),
        joined=col("t").list.unique().list.sort().list.join("|"),
    ).to_pydict()
    print(tidy)
    assert tidy["sorted_tags"][0] == ["a", "b", "b", "c"]
    assert sorted(tidy["unique_tags"][0]) == ["a", "b", "c"]
    assert tidy["joined"][0] == "a|b|c"


if __name__ == "__main__":
    main()
