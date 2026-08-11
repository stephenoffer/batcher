"""When both sides have a column of the same name.

The right-hand duplicate gets a suffix rather than overwriting the left. Choose the
suffix deliberately: `_right` on ten columns is how a downstream `select` starts guessing.
Projecting each side down before the join is usually better than renaming after it.

    python examples/joins/column_collisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    nation = tpch("nation").select("n_nationkey", "n_name", "n_regionkey", "n_comment")

    # Join nation to itself: every non-key column collides.
    other = nation.rename({"n_nationkey": "partner_key"})
    collided = nation.join(other, left_on="n_regionkey", right_on="partner_key", how="left")
    print(collided.columns)
    assert any(name.endswith("_right") for name in collided.columns)

    # A chosen suffix reads better than the default.
    suffixed = nation.join(
        other, left_on="n_regionkey", right_on="partner_key", how="left", suffix="_partner"
    )
    assert "n_name_partner" in suffixed.columns
    assert "n_name" in suffixed.columns

    # The cleanest fix is to name the columns before joining, not after.
    tidy_right = nation.select(
        col("n_nationkey").alias("partner_key"),
        col("n_name").alias("partner_name"),
    )
    tidy = nation.join(tidy_right, left_on="n_regionkey", right_on="partner_key", how="left")
    print(tidy.head(3).to_pydict())
    assert not any(name.endswith("_right") for name in tidy.columns)
    assert set(tidy.columns) == {
        "n_nationkey",
        "n_name",
        "n_regionkey",
        "n_comment",
        "partner_name",
    }


if __name__ == "__main__":
    main()
