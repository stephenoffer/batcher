"""Multiplicative folds, and the overflow you get for free.

`product` over 200,000 positive values overflows to infinity — not an error, an IEEE
infinity. The standard fix is to add logarithms instead and exponentiate at the end,
which is also what makes the fold numerically stable.

    python examples/aggregations/products_and_overflow.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    # Over the whole table this overflows, silently and by design.
    overflowing = lineitem.agg(p=bt.product(col("l_quantity"))).to_pydict()["p"][0]
    print("product over 200k rows:", overflowing)
    assert math.isinf(overflowing)

    # Over a handful of rows it is an ordinary number.
    small = lineitem.head(10)
    exact = small.agg(p=bt.product(col("l_quantity"))).to_pydict()["p"][0]
    by_hand = math.prod(small.to_pydict()["l_quantity"])
    print("product over 10 rows:", exact)
    assert exact == by_hand

    # The log-sum form survives the full table and stays comparable across group sizes.
    log_space = lineitem.agg(
        log_total=col("l_quantity").log().sum(),
        rows=bt.count(),
    ).to_pydict()
    geometric = math.exp(log_space["log_total"][0] / log_space["rows"][0])
    print(f"geometric mean via logs: {geometric:.4f}")

    # Which is exactly what the geometric-mean aggregate computes.
    built_in = lineitem.agg(g=bt.geometric_mean(col("l_quantity"))).to_pydict()["g"][0]
    assert abs(geometric - built_in) / built_in < 1e-9


if __name__ == "__main__":
    main()
