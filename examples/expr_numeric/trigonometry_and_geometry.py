"""Trigonometric and geometric helpers.

`hypot` and `atan2` exist because the obvious formulas overflow or lose the quadrant.
`hypot(x, y)` is stable where `sqrt(x*x + y*y)` overflows, and `atan2(y, x)` knows which
quadrant it is in where `atan(y/x)` does not.

    python examples/expr_numeric/trigonometry_and_geometry.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    points = bt.from_pydict(
        {
            "x": [3.0, -3.0, 0.0, 1.0],
            "y": [4.0, 4.0, 5.0, 0.0],
        }
    )

    derived = points.select(
        "x",
        "y",
        distance=bt.hypot(col("x"), col("y")),
        angle=bt.atan2(col("y"), col("x")),
        naive=(col("x") ** 2 + col("y") ** 2).sqrt(),
    )
    result = derived.to_pydict()
    print({name: [round(v, 4) for v in column] for name, column in result.items()})

    # 3-4-5 triangles.
    assert abs(result["distance"][0] - 5.0) < 1e-9
    assert abs(result["distance"][2] - 5.0) < 1e-9

    # `hypot` agrees with the naive formula where the naive one is safe.
    assert all(
        abs(stable - naive) < 1e-9
        for stable, naive in zip(result["distance"], result["naive"], strict=True)
    )

    # `atan2` keeps the quadrant: the two 3-4-5 points differ in sign of x, and their
    # angles differ accordingly. `atan(y/x)` would give the same answer for both.
    assert result["angle"][0] != result["angle"][1]
    assert 0 < result["angle"][0] < math.pi / 2
    assert math.pi / 2 < result["angle"][1] < math.pi

    # Straight up the y axis is a right angle; straight along x is zero.
    assert abs(result["angle"][2] - math.pi / 2) < 1e-9
    assert abs(result["angle"][3]) < 1e-9


if __name__ == "__main__":
    main()
