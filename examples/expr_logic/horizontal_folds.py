"""Reducing across columns in a row, not down a column.

The `*_horizontal` family is the row-wise counterpart to an aggregate. It is the right tool
whenever a value is spread across columns rather than rows — which usually means the table
is in wide form, and this saves you unpivoting it first.

    python examples/expr_logic/horizontal_folds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount", "l_tax")

    folded = lineitem.select(
        total=bt.sum_horizontal(col("l_discount"), col("l_tax")),
        biggest=bt.max_horizontal(col("l_discount"), col("l_tax")),
        smallest=bt.min_horizontal(col("l_discount"), col("l_tax")),
        average=bt.mean_horizontal(col("l_discount"), col("l_tax")),
        any_charge=bt.any_horizontal(col("l_discount") > 0, col("l_tax") > 0),
        all_charges=bt.all_horizontal(col("l_discount") > 0, col("l_tax") > 0),
        # `count_horizontal` counts *non-null* arguments, not true ones — it is the
        # row-wise counterpart of `col(x).count()`. To count satisfied predicates,
        # cast them to integers and sum.
        non_null=bt.count_horizontal(col("l_discount"), col("l_tax")),
        set_count=bt.sum_horizontal(
            (col("l_discount") > 0).cast("int64"), (col("l_tax") > 0).cast("int64")
        ),
    )

    result = folded.head(5).to_pydict()
    print({name: column[:3] for name, column in result.items()})

    full = folded.to_pydict()

    # The fold identities hold row by row.
    assert all(low <= high for low, high in zip(full["smallest"], full["biggest"], strict=True))
    assert all(
        abs(total / 2 - mean) < 1e-9
        for total, mean in zip(full["total"], full["average"], strict=True)
    )

    # Both columns are always present, so the non-null count is 2 everywhere.
    assert set(full["non_null"]) == {2}

    # `any` and `all` bracket the count of satisfied predicates.
    assert all(
        (count > 0) == any_set
        for count, any_set in zip(full["set_count"], full["any_charge"], strict=True)
    )
    assert all(
        (count == 2) == all_set
        for count, all_set in zip(full["set_count"], full["all_charges"], strict=True)
    )
    # Not every line is discounted, so the two genuinely differ somewhere.
    assert set(full["set_count"]) != {2}


if __name__ == "__main__":
    main()
