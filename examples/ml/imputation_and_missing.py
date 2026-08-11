"""Filling missing values, and recording that they were missing.

An imputed value is a guess, and a model that cannot tell a guess from a measurement will
trust it equally. `MissingIndicator` keeps that information: impute the column *and* add a
flag, so the model can learn that absence itself is a signal.

    python examples/ml/imputation_and_missing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_custkey", "o_totalprice")
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity")

    # A left join gives a column with genuine gaps.
    gapped = orders.join(lineitem, left_on="o_orderkey", right_on="l_orderkey", how="left")
    missing = gapped.filter(col("l_quantity").is_null()).count()
    print(f"{missing} of {gapped.count()} rows have no quantity")
    assert missing > 0

    # Record the gap before filling it.
    flagged = ml.MissingIndicator("l_quantity").fit(gapped).transform(gapped)
    indicator = next(name for name in flagged.columns if name not in gapped.columns)
    print("indicator column:", indicator)

    filled = ml.SimpleImputer("l_quantity", strategy="median").fit(flagged).transform(flagged)
    assert filled.filter(col("l_quantity").is_null()).count() == 0

    # The flag still marks the rows that were imputed.
    # The indicator is a boolean column, so it is the predicate — no comparison.
    marked = filled.filter(col(indicator)).count()
    assert marked == missing

    # The imputed value is the training median, not zero.
    median = gapped.agg(m=col("l_quantity").median()).to_pydict()["m"][0]
    imputed_values = set(
        filled.filter(col(indicator)).select("l_quantity").distinct().to_pydict()["l_quantity"]
    )
    print("imputed with:", imputed_values, "median was:", median)
    assert len(imputed_values) == 1
    assert abs(next(iter(imputed_values)) - median) < 1e-6


if __name__ == "__main__":
    main()
