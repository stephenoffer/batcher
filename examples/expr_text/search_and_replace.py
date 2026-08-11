"""Finding and rewriting substrings, literally and by pattern.

`replace` is literal and `regexp_replace` is a pattern, and mixing them up is how a `.` in a
"literal" string quietly matches everything. Reach for the literal form unless you actually
need the pattern.

    python examples/expr_text/search_and_replace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_clerk", "o_comment").head(1_000)

    rewritten = orders.select(
        "o_clerk",
        literal=col("o_clerk").str.replace("Clerk#", "C"),
        pattern=col("o_clerk").str.regexp_replace(r"Clerk#0*", "clerk-"),
        found=col("o_clerk").str.position("#"),
        prefix=col("o_clerk").str.substring_index("#", 1),
    )
    result = rewritten.head(3).to_pydict()
    print(result)

    full = rewritten.to_pydict()
    assert all(value.startswith("C0") for value in full["literal"])
    assert all(value.startswith("clerk-") for value in full["pattern"])
    # The leading zeros are gone from the pattern version but not the literal one.
    assert full["pattern"][0] != full["literal"][0]

    # `position` is 1-based, and every clerk id has its hash in the same place.
    assert len(set(full["found"])) == 1
    assert full["found"][0] == 6
    assert all(value == "Clerk" for value in full["prefix"])

    # Counting matches without materializing them.
    counts = orders.select(
        words=col("o_comment").str.regexp_count(r"\w+"),
        has_final=col("o_comment").str.contains("final"),
    ).to_pydict()
    assert all(value > 0 for value in counts["words"])
    assert any(counts["has_final"])


if __name__ == "__main__":
    main()
