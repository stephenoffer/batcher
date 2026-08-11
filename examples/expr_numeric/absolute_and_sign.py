"""Magnitude and sign, and the clipping that bounds a column.

Clipping is the honest alternative to dropping outliers: the row survives with a bounded
value rather than disappearing, so the row count stays comparable across runs.

    python examples/expr_numeric/absolute_and_sign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount")

    centred = lineitem.with_columns(
        deviation=col("l_quantity") - lineitem.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
    )

    described = centred.select(
        "deviation",
        magnitude=col("deviation").abs(),
        clipped=col("deviation").clip(-10.0, 10.0),
    )
    result = described.to_pydict()
    print({name: [round(v, 2) for v in column[:4]] for name, column in result.items()})

    # Magnitude is never negative and never smaller than the value it came from.
    assert all(value >= 0 for value in result["magnitude"])
    assert all(
        magnitude >= abs(value)
        for value, magnitude in zip(result["deviation"], result["magnitude"], strict=True)
    )

    # Clipping bounds without dropping: same row count, bounded range.
    assert len(result["clipped"]) == lineitem.count()
    assert all(-10.0 <= value <= 10.0 for value in result["clipped"])

    # And it only moves the values that were outside the bounds.
    assert all(
        clipped == value
        for value, clipped in zip(result["deviation"], result["clipped"], strict=True)
        if -10.0 <= value <= 10.0
    )

    # The frame-level form clips every numeric column at once.
    bounded = lineitem.clip(0.0, 100.0)
    assert bounded.count() == lineitem.count()
    assert bt is not None


if __name__ == "__main__":
    main()
