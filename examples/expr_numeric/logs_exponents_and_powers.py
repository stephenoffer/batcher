"""Logarithms, exponentials and powers over a real numeric column.

The log family is the standard way to compress a right-skewed distribution before you
model it, and to keep a product from overflowing. Every one of these is undefined
somewhere, and the undefined case gives you a null or an infinity rather than an error.

    python examples/expr_numeric/logs_exponents_and_powers.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    lineitem = tpch("lineitem").select("l_quantity", "l_extendedprice").head(5_000)

    transformed = lineitem.select(
        "l_quantity",
        natural=col("l_quantity").log(),
        base10=col("l_quantity").log10(),
        base2=col("l_quantity").log2(),
        squared=col("l_quantity") ** 2,
        root=col("l_quantity").sqrt(),
        exponent=col("l_quantity").log().exp(),
    )

    sample = transformed.head(3).to_pydict()
    print({name: [round(value, 4) for value in column] for name, column in sample.items()})

    full = transformed.to_pydict()

    # Logs in different bases are the same number scaled by a constant.
    assert all(
        abs(natural / math.log(10) - ten) < 1e-9
        for natural, ten in zip(full["natural"], full["base10"], strict=True)
    )
    assert all(
        abs(natural / math.log(2) - two) < 1e-9
        for natural, two in zip(full["natural"], full["base2"], strict=True)
    )

    # exp and log invert each other.
    assert all(
        abs(original - restored) < 1e-9
        for original, restored in zip(full["l_quantity"], full["exponent"], strict=True)
    )

    # Square and square root invert each other too.
    assert all(
        abs(root**2 - original) < 1e-9
        for original, root in zip(full["l_quantity"], full["root"], strict=True)
    )

    # The log transform compresses the range, which is what it is for.
    spread = transformed.agg(
        raw=col("l_quantity").max() - col("l_quantity").min(),
        logged=col("natural").max() - col("natural").min(),
    ).to_pydict()
    print(spread)
    assert spread["logged"][0] < spread["raw"][0]


if __name__ == "__main__":
    main()
