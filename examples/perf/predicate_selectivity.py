"""Selectivity: how much a predicate actually removes, and why it matters.

Selectivity is the input to every join-order decision, and it is measurable in one count.
Measuring the two or three predicates in a slow query is usually enough to see which one the
optimizer is mis-estimating.

    python examples/perf/predicate_selectivity.py
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
    total = lineitem.count()

    predicates = {
        "quantity > 45": col("l_quantity") > 45,
        "discount = 0": col("l_discount") == 0.0,
        "shipmode = AIR": col("l_shipmode") == "AIR",
        "returnflag = R": col("l_returnflag") == "R",
        "comment contains final": col("l_comment").str.contains("final"),
    }

    selectivities: dict[str, float] = {}
    for name, predicate in predicates.items():
        kept = lineitem.filter(predicate).count()
        selectivities[name] = kept / total
        print(f"{name:<24} keeps {kept:>7} of {total} ({kept / total:.4%})")

    # Every selectivity is a proportion.
    assert all(0.0 <= value <= 1.0 for value in selectivities.values())

    # The most selective predicate is the one to apply first.
    most_selective = min(selectivities, key=lambda key: selectivities[key])
    print("most selective:", most_selective)
    assert selectivities[most_selective] < 0.5

    # Conjunctions are at most as selective as their least selective term, and usually
    # far more — which is the estimate a cost model has to guess at.
    combined = lineitem.filter(predicates["quantity > 45"] & predicates["shipmode = AIR"]).count()
    print(f"both: {combined} ({combined / total:.4%})")
    assert combined <= min(
        lineitem.filter(predicates["quantity > 45"]).count(),
        lineitem.filter(predicates["shipmode = AIR"]).count(),
    )

    # And the independence assumption, which is what a naive estimator uses.
    predicted = selectivities["quantity > 45"] * selectivities["shipmode = AIR"] * total
    print(f"independence estimate: {predicted:.0f} vs actual {combined}")
    assert predicted > 0
    assert bt is not None


if __name__ == "__main__":
    main()
