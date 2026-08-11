"""Dividing without producing an infinity or a null you did not plan for.

A zero denominator gives an infinity for floats and an error for integers, and both propagate
into every aggregate downstream. Guarding the denominator explicitly is one `when`, and it is
the difference between a rate column and a column of infinities.

    python examples/expr_numeric/safe_division.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    counts = bt.from_pydict(
        {
            "name": ["full", "half", "empty", "none"],
            "hits": [10.0, 5.0, 0.0, 3.0],
            "total": [10.0, 10.0, 10.0, 0.0],
        }
    )

    naive = counts.select("name", rate=col("hits") / col("total")).to_pydict()
    print("naive:", naive["rate"])

    # The zero denominator produces an infinity, silently.
    assert math.isinf(naive["rate"][3])

    guarded = counts.select(
        "name",
        rate=bt.when(col("total") > 0).then(col("hits") / col("total")).otherwise(0.0),
    ).to_pydict()
    print("guarded:", guarded["rate"])
    assert all(math.isfinite(value) for value in guarded["rate"])
    assert guarded["rate"] == [1.0, 0.5, 0.0, 0.0]

    # `nullif` is the other spelling: turn the zero into a null and let the null
    # propagate deliberately rather than an infinity propagate accidentally.
    nulled = counts.select(
        "name", rate=col("hits") / bt.nullif(col("total"), bt.lit(0.0))
    ).to_pydict()
    print("nulled:", nulled["rate"])
    assert nulled["rate"][3] is None
    assert all(value is not None for value in nulled["rate"][:3])

    # The choice matters downstream: an infinity poisons a mean, a null is skipped by it.
    poisoned = (
        counts.select(rate=col("hits") / col("total")).agg(m=col("rate").mean()).to_pydict()["m"][0]
    )
    clean = (
        counts.select(rate=col("hits") / bt.nullif(col("total"), bt.lit(0.0)))
        .agg(m=col("rate").mean())
        .to_pydict()["m"][0]
    )
    print(f"mean with an infinity: {poisoned}, with a null: {clean:.4f}")
    assert math.isinf(poisoned)
    assert math.isfinite(clean)


if __name__ == "__main__":
    main()
