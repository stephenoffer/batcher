"""The numeric helpers a storage-usage report reaches for, and why each is the exact one.

A report over measured data runs into the same four problems every time: a float sum that
drifts, a percentile that has to be a value someone actually observed, a size nobody can
read, and flags packed into an integer. Each has a matching expression here, and each is a
different answer from the obvious one.

    python examples/expr_numeric/reporting_pass_helpers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    usage = bt.from_pydict(
        {
            "tenant": ["a", "a", "a", "b", "b", "b"],
            # A large value beside small ones: the shape that loses the small ones in a
            # naive left-to-right float sum.
            "bytes": [1e16, 1.0, 1.0, 2.0**30, 512.0, 1536.0],
            "budget": [1e16, 2.0, 2.0, 2.0**30, 1024.0, 1024.0],
            # Permission bits packed into one integer, as an ACL column arrives.
            "flags": [0b0001, 0b0110, 0b0100, 0b1000, 0b0010, 0b0011],
        }
    )

    # `kahan_sum` carries the rounding error the running total drops. Over this column a
    # plain `sum` loses the two 1.0s entirely against the 1e16.
    totals = (
        usage.group_by("tenant")
        .agg(plain=col("bytes").sum(), exact=col("bytes").kahan_sum())
        .sort("tenant")
        .to_pydict()
    )
    assert totals["exact"][0] == 1e16 + 2.0, totals["exact"][0]
    assert totals["plain"][0] != totals["exact"][0], "the fixture stopped exercising drift"

    # `quantile_disc` returns a value that is *in* the column. The interpolating
    # `quantile` may return one that is not, which is wrong for a report that has to name
    # a real observation ("the p50 tenant used this much").
    observed = usage.select("bytes").to_pydict()["bytes"]
    disc = usage.agg(p50=col("bytes").quantile_disc(0.5)).to_pydict()["p50"][0]
    assert disc in observed, f"{disc} is not an observed value"

    # `abs_diff` is the signless distance to the budget — shorter than `(a - b).abs()` and,
    # more usefully, it says what it means at the call site.
    over = usage.select(gap=col("bytes").abs_diff(col("budget"))).to_pydict()["gap"]
    assert over[1] == 1.0 and over[4] == 512.0, over

    # `format_bytes` renders the ladder a human reads, so the report does not carry a
    # column of raw integers.
    sized = usage.select(human=col("bytes").cast("int64").format_bytes()).to_pydict()["human"]
    assert sized[3] == "1.0 GiB", sized[3]
    assert sized[4] == "512 bytes", sized[4]

    # Bit math on the packed flags: shift a mask into place rather than hard-coding the
    # constant, so the bit positions stay readable.
    write_bit = 1
    perms = usage.select(
        can_write=(col("flags").bitwise_right_shift(write_bit) % 2) == 1,
        doubled=col("flags").bitwise_left_shift(1),
    ).to_pydict()
    assert perms["can_write"] == [False, True, False, False, True, True], perms["can_write"]
    assert perms["doubled"][0] == 0b0010

    # `rolling_count` counts the non-null rows in the window rather than its width, which
    # is what makes it the denominator for a rate over sparse data.
    sparse = bt.from_pydict({"t": [1, 2, 3, 4], "v": [1.0, None, 3.0, None]})
    windowed = sparse.select(t=col("t"), seen=col("v").rolling_count(2, order_by="t")).to_pydict()[
        "seen"
    ]
    assert windowed[-1] == 1, windowed  # rows 3 and 4 hold one non-null between them

    # `lgamma` is log-factorial without the overflow: `gamma(171)` is already infinite in
    # float64, and a log-likelihood needs the log anyway.
    big = bt.from_pydict({"n": [170.0, 171.0, 5.0]})
    lg = big.select(lg=col("n").lgamma()).to_pydict()["lg"]
    assert all(v == v and v != float("inf") for v in lg), lg
    assert abs(lg[2] - 3.178053830347946) < 1e-9  # lgamma(5) = ln(4!) = ln(24)

    # `AggExpr.map_operands` rewrites what an aggregate reads without rebuilding it: here,
    # pointing the same `sum` at a different column. `operands` is the read side of it.
    total = col("bytes").sum()
    assert [str(o) for o in total.operands()] == [str(col("bytes"))]
    retargeted = total.map_operands(lambda _e: col("budget"))
    assert [str(o) for o in retargeted.operands()] == [str(col("budget"))]
    checked = usage.agg(b=retargeted).to_pydict()["b"][0]
    assert checked == usage.agg(b=col("budget").sum()).to_pydict()["b"][0]

    print("tenant a exact total:", totals["exact"][0])
    print("p50 (an observed value):", disc)
    print("largest row rendered  :", sized[3])


if __name__ == "__main__":
    main()
