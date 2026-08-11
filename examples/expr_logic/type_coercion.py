"""What happens when an expression mixes types.

Mixed arithmetic widens to the type that can hold both, which is usually what you want and
occasionally not: an integer division that silently becomes a float changes the meaning of a
count. Asserting the resulting type is how you find out which case you are in.

    python examples/expr_logic/type_coercion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice", "l_discount")

    mixed = lineitem.select(
        int_times_int=col("l_quantity") * col("l_quantity"),
        int_times_float=col("l_quantity") * col("l_discount"),
        int_divided=col("l_quantity") / col("l_quantity"),
        int_floor_divided=col("l_quantity") // col("l_quantity"),
        int_plus_literal=col("l_quantity") + 1,
        int_plus_float_literal=col("l_quantity") + 1.5,
    )
    types = dict(zip(mixed.columns, [str(dtype) for dtype in mixed.dtypes], strict=True))
    for name, dtype in types.items():
        print(f"  {name:<24} {dtype}")

    # Integer arithmetic stays integer.
    assert types["int_times_int"] == "int64"
    assert types["int_plus_literal"] == "int64"

    # Mixing with a float widens, whether the float is a column or a literal.
    assert types["int_times_float"] == "double"
    assert types["int_plus_float_literal"] == "double"

    # True division always widens; floor division does not. That difference is the one
    # that turns a count into a fraction without anyone noticing.
    assert types["int_divided"] == "double"
    assert types["int_floor_divided"] == "int64"

    values = mixed.head(3).to_pydict()
    assert all(value == 1.0 for value in values["int_divided"])
    assert all(value == 1 for value in values["int_floor_divided"])

    # An explicit cast is how you pin the type when it matters.
    pinned = lineitem.select(ratio=(col("l_quantity") / col("l_quantity")).cast("int64"))
    assert str(pinned.dtypes[0]) == "int64"


if __name__ == "__main__":
    main()
