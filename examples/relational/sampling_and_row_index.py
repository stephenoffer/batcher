"""Sampling a large table, and attaching a row number.

A seeded sample is reproducible, which is what makes it usable in a test or a report.
`with_row_index` numbers rows in the order they arrive, so it is only meaningful after a
sort — otherwise the numbers describe the scan, not the data.

    python examples/relational/sampling_and_row_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_extendedprice")
    total = lineitem.count()

    # A fixed number of rows.
    sample = lineitem.sample(n=1000, seed=7)
    assert sample.count() == 1000

    # The same seed gives the same rows.
    again = lineitem.sample(n=1000, seed=7)
    assert sample.to_pydict() == again.to_pydict()

    # A fraction of the table.
    tenth = lineitem.sample(frac=0.1, seed=11)
    print("sampled", tenth.count(), "of", total)
    assert 0.05 * total < tenth.count() < 0.15 * total

    # Row numbers over a defined order.
    numbered = (
        lineitem.sort("l_extendedprice", descending=True).with_row_index(name="rank").head(5)
    ).to_pydict()
    print(numbered["rank"], [round(value) for value in numbered["l_extendedprice"]])
    assert numbered["rank"] == [0, 1, 2, 3, 4]
    assert numbered["l_extendedprice"] == sorted(numbered["l_extendedprice"], reverse=True)

    # A sample is a fair estimate of the mean, not a guess at it.
    population_mean = lineitem.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
    sample_mean = tenth.agg(m=col("l_quantity").mean()).to_pydict()["m"][0]
    print(f"population {population_mean:.3f} vs sample {sample_mean:.3f}")
    assert abs(population_mean - sample_mean) < 1.0


if __name__ == "__main__":
    main()
