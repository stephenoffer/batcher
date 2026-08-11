"""Case conversion and padding, for fixed-width output.

Padding is how a variable-length value becomes a fixed-width field, which is what a
mainframe extract or a report column needs. The width is in characters, so a multi-byte
string pads to the character count rather than the byte count.

    python examples/expr_text/case_and_padding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    nation = tpch("nation").select("n_name")

    shaped = nation.select(
        "n_name",
        upper=col("n_name").str.to_uppercase(),
        lower=col("n_name").str.to_lowercase(),
        title=col("n_name").str.to_titlecase(),
        padded=col("n_name").str.rpad(20, "."),
        left_padded=col("n_name").str.lpad(20, " "),
        zeroed=col("n_name").str.zfill(20),
    )

    result = shaped.head(3).to_pydict()
    for value in result["padded"]:
        print(repr(value))

    full = shaped.to_pydict()
    assert all(value == value.upper() for value in full["upper"])
    assert all(value == value.lower() for value in full["lower"])

    # Every padded value is exactly the requested width.
    assert all(len(value) == 20 for value in full["padded"])
    assert all(len(value) == 20 for value in full["left_padded"])
    assert all(value.endswith(".") for value in full["padded"])

    # Padding never truncates: a value already at the width is unchanged.
    long_enough = nation.select(p=col("n_name").str.rpad(3, ".")).to_pydict()
    assert all(len(value) >= 3 for value in long_enough["p"])


if __name__ == "__main__":
    main()
