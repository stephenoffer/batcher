"""Vector shape: magnitudes, unit norm, and pooling many vectors into one.

`magnitude`, `dim` and `is_unit_norm` describe a vector without decoding it into Python.
`mean_pool` and `max_pool` reduce one vector to one number — they pool *within* a vector,
not across several, which is the reading that trips people up. Combining many vectors into
one is a group-by over rows.

    python examples/expr_vectors/normalization_and_pooling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    vectors = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "vector": [[3.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        }
    )

    described = vectors.select(
        "id",
        magnitude=col("vector").list.magnitude(),
        dimensions=col("vector").list.dim(),
        unit=col("vector").list.is_unit_norm(),
        zero=col("vector").list.is_zero_vector(),
        sum_squares=col("vector").list.sum_squares(),
    )
    result = described.to_pydict()
    print(result)

    # A 3-4-5 triangle: the first vector has magnitude 5.
    assert abs(result["magnitude"][0] - 5.0) < 1e-9
    assert result["dimensions"] == [3, 3, 3]
    assert result["unit"] == [False, True, False]
    assert not any(result["zero"])

    # Magnitude squared is the sum of squares.
    assert all(
        abs(magnitude**2 - squares) < 1e-6
        for magnitude, squares in zip(result["magnitude"], result["sum_squares"], strict=True)
    )

    # Element-wise arithmetic between two list columns.
    doubled = vectors.select("id", squared=col("vector").list.multiply(col("vector"))).to_pydict()
    print(doubled)
    assert doubled["squared"][0] == [9.0, 16.0, 0.0]

    # Pooling reduces one vector to one number: the mean and the largest component.
    # These are the sequence-pooling primitives, so they collapse a vector rather than
    # combining several — a document embedding built from chunk embeddings is a
    # `group_by` over chunk rows, not a call to these.
    pooled = vectors.select(
        "id",
        mean=col("vector").list.mean_pool(),
        peak=col("vector").list.max_pool(),
    ).to_pydict()
    print(pooled)
    assert [round(value, 6) for value in pooled["mean"]] == [2.333333, 0.333333, 0.666667]
    assert pooled["peak"] == [4.0, 1.0, 2.0]

    # The mean component times the dimension is the sum of the components.
    summed = vectors.select("id", total=col("vector").list.sum()).to_pydict()
    assert all(
        abs(mean * 3 - total) < 1e-6
        for mean, total in zip(pooled["mean"], summed["total"], strict=True)
    )


if __name__ == "__main__":
    main()
