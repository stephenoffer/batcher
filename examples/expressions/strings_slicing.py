"""String slicing: taking a fixed piece of every value.

``head``/``tail`` take from the ends, ``slice``/``substr`` take from an offset, and
``split_part`` takes the nth field of a delimited value. All of them are safe on values
shorter than the requested window: you get what is there rather than an error.

    python examples/expressions/strings_slicing.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    codes = bt.from_pydict(
        {
            "sku": ["US-1234-RED", "EU-99-BLUE", "AP-70001-GREEN"],
        }
    )

    parts = codes.with_columns(
        # Fixed-width pieces from each end.
        region=col("sku").str.head(2),
        last4=col("sku").str.tail(4),
        # From an offset. `slice` is 0-based; `substr` is 1-based like SQL.
        after_region=col("sku").str.slice(3),
        first_two_slice=col("sku").str.slice(0, 2),
        sql_style=col("sku").str.substr(1, 2),
        # Take a delimited field by position (1-based).
        middle=col("sku").str.split_part("-", 2),
        colour=col("sku").str.split_part("-", 3),
        # `left`/`right` are the SQL spellings of head/tail.
        left2=col("sku").str.left(2),
        right3=col("sku").str.right(3),
    )

    result = parts.to_pydict()
    print(result)

    assert result["region"] == ["US", "EU", "AP"]
    assert result["last4"] == ["-RED", "BLUE", "REEN"]
    assert result["after_region"] == ["1234-RED", "99-BLUE", "70001-GREEN"]
    assert result["first_two_slice"] == ["US", "EU", "AP"]
    assert result["sql_style"] == ["US", "EU", "AP"]
    assert result["middle"] == ["1234", "99", "70001"]
    assert result["colour"] == ["RED", "BLUE", "GREEN"]
    assert result["left2"] == result["region"]
    assert result["right3"] == ["RED", "LUE", "EEN"]


if __name__ == "__main__":
    main()
