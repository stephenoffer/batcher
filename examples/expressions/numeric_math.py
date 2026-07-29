"""Arithmetic and math functions on numeric columns.

All of these are columnar and fuse into a single pass, so a chain of ten of them is not
ten scans. Watch the division operators in particular: ``/`` is true division and
``floordiv`` truncates, and mixing them up is a quiet source of off-by-one bugs.

    python examples/expressions/numeric_math.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    nums = bt.from_pydict({"x": [-4.0, -0.5, 0.0, 2.0, 9.0], "n": [7, 7, 7, 7, 7]})

    out = nums.with_columns(
        magnitude=col("x").abs(),
        sign=col("x").sign(),
        rounded=col("x").round(0),
        floored=col("x").floor(),
        ceiled=col("x").ceil(),
        # `sqrt`/`log` are undefined outside their domain and yield null or NaN there.
        root=col("x").abs().sqrt(),
        exponent=col("x").exp(),
        squared=col("x").pow(2),
        # Bound a column without a branch.
        clipped=col("x").clip(-1.0, 5.0),
    ).to_pydict()

    print(out)

    assert out["magnitude"] == [4.0, 0.5, 0.0, 2.0, 9.0]
    assert out["sign"] == [-1.0, -1.0, 0.0, 1.0, 1.0]
    assert out["floored"] == [-4.0, -1.0, 0.0, 2.0, 9.0]
    assert out["ceiled"] == [-4.0, 0.0, 0.0, 2.0, 9.0]
    assert out["root"][-1] == 3.0
    assert out["squared"] == [16.0, 0.25, 0.0, 4.0, 81.0]
    assert out["clipped"] == [-1.0, -0.5, 0.0, 2.0, 5.0]

    # Division and modulo.
    div = nums.select(
        true_div=col("n") / 2,
        floor_div=col("n").floordiv(2),
        remainder=col("n").mod(2),
    ).to_pydict()
    print(div)
    assert div["true_div"][0] == 3.5
    assert div["floor_div"][0] == 3
    assert div["remainder"][0] == 1

    # Range and membership predicates.
    checks = nums.select(
        in_range=col("x").between(0.0, 9.0),
        open_range=col("x").between(0.0, 9.0, closed="none"),
        listed=col("x").is_in([0.0, 9.0]),
    ).to_pydict()
    print(checks)
    assert checks["in_range"] == [False, False, True, True, True]
    # `closed="none"` excludes both endpoints, so 0.0 and 9.0 drop out.
    assert checks["open_range"] == [False, False, False, True, False]
    assert checks["listed"] == [False, False, True, False, True]

    # Bitwise operations on integers.
    bits = bt.from_pydict({"a": [0b1100], "b": [0b1010]})
    bw = bits.select(
        conj=col("a").bitwise_and(col("b")),
        disj=col("a").bitwise_or(col("b")),
    ).to_pydict()
    print(bw)
    assert bw["conj"] == [0b1000]
    assert bw["disj"] == [0b1110]


if __name__ == "__main__":
    main()
