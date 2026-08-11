"""Flagging rows that do not look like the rest.

Three detectors, in increasing order of assumption: a fixed business rule, a percentile cut,
and a z-score. The business rule is the only one that cannot be wrong about the distribution,
which is why it goes first.

    python examples/quality/anomaly_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice")
    total = orders.count()

    # 1. A business rule: no assumption about the distribution at all.
    impossible = orders.filter(col("o_totalprice") <= 0).count()
    print("non-positive prices:", impossible)
    assert impossible == 0

    # 2. A percentile cut: distribution-free, and it flags a fixed share by construction.
    cutoff = orders.agg(t=bt.quantile(col("o_totalprice"), 0.995)).to_pydict()["t"][0]
    extreme = orders.filter(col("o_totalprice") > cutoff)
    print(f"above the 99.5th percentile: {extreme.count()} ({extreme.count() / total:.4%})")
    assert 0.002 < extreme.count() / total < 0.01

    # 3. A z-score: assumes a shape this column does not have, and flags a different set.
    moments = orders.agg(
        mean=col("o_totalprice").mean(), sd=bt.std(col("o_totalprice"))
    ).to_pydict()
    mean, sd = moments["mean"][0], moments["sd"][0]
    scored = orders.with_columns(z=(col("o_totalprice") - mean) / sd)
    sigma_flagged = scored.filter(col("z").abs() > 3)
    print(f"beyond three sigma: {sigma_flagged.count()} ({sigma_flagged.count() / total:.4%})")

    # On a normal distribution that would be about 0.27%. It is not, because the column is
    # right-skewed — which is the reason to prefer the percentile cut here.
    assert sigma_flagged.count() != extreme.count()

    # The two agree on the most extreme rows even where they disagree on the boundary.
    top_by_z = set(scored.sort("z", descending=True).head(10).to_pydict()["o_orderkey"])
    top_by_value = set(
        orders.sort("o_totalprice", descending=True).head(10).to_pydict()["o_orderkey"]
    )
    assert top_by_z == top_by_value


if __name__ == "__main__":
    main()
