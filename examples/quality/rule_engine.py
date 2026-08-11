"""A rule engine as a projection: one boolean column per rule.

Every rule is a boolean column and the verdict is a fold over them. That shape means adding a
rule is adding a column, the per-rule violation counts come free, and the whole thing is one
pass rather than one pass per rule.

    python examples/quality/rule_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem")

    rules = {
        "quantity_in_range": (col("l_quantity") >= 1) & (col("l_quantity") <= 50),
        "discount_in_range": (col("l_discount") >= 0.0) & (col("l_discount") <= 0.1),
        "price_positive": col("l_extendedprice") > 0,
        "ship_before_receipt": col("l_shipdate") <= col("l_receiptdate"),
        "flag_known": col("l_returnflag").is_in(["A", "N", "R"]),
    }

    checked = lineitem.with_columns(**rules)

    # Per-rule violation counts, in one pass.
    counts = checked.agg(
        **{name: bt.count_if(~col(name)) for name in rules}, rows=bt.count()
    ).to_pydict()
    for name in rules:
        print(f"  {name:<22} {counts[name][0]:>6} violations")
    assert counts["rows"][0] == lineitem.count()

    # This data satisfies every rule.
    assert all(counts[name][0] == 0 for name in rules)

    # The verdict: a row passes when every rule holds.
    verdict = checked.with_columns(passes=bt.all_horizontal(*[col(name) for name in rules]))
    assert verdict.agg(n=bt.count_if(col("passes"))).to_pydict()["n"][0] == lineitem.count()

    # A rule that does fail, to show the machinery works.
    strict = checked.with_columns(discount_small=col("l_discount") <= 0.02)
    failures = strict.agg(n=bt.count_if(~col("discount_small"))).to_pydict()["n"][0]
    print("rows failing the strict discount rule:", failures)
    assert 0 < failures < lineitem.count()

    # And the failing rows are exactly the ones the rule names.
    quarantined = strict.filter(~col("discount_small"))
    assert quarantined.count() == failures
    assert quarantined.agg(m=col("l_discount").min()).to_pydict()["m"][0] > 0.02


if __name__ == "__main__":
    main()
