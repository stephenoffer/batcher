"""Walking a hierarchy without recursion.

A fixed-depth hierarchy is a chain of self-joins, one per level, and that is usually enough:
a category tree is three deep, an org chart six. Unbounded recursion is a different problem;
this is the one people actually have.

    python examples/relational/self_referential_hierarchies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    nodes = bt.from_pydict(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["root", "eu", "us", "france", "germany", "paris"],
            "parent": [None, 1, 1, 2, 2, 4],
        }
    )

    def level(child: bt.Dataset, suffix: str) -> bt.Dataset:
        return nodes.select(
            col("id").alias(f"{suffix}_id"),
            col("name").alias(f"{suffix}_name"),
            col("parent").alias(f"{suffix}_parent"),
        )

    # One join per level up.
    with_parent = nodes.join(level(nodes, "p"), left_on="parent", right_on="p_id", how="left")
    with_grandparent = with_parent.join(
        level(nodes, "g"), left_on="p_parent", right_on="g_id", how="left"
    )

    result = (
        with_grandparent.select("name", parent=col("p_name"), grandparent=col("g_name"))
        .sort("name")
        .to_pydict()
    )
    for row in zip(result["name"], result["parent"], result["grandparent"], strict=True):
        print(f"  {row[0]:<10} parent={row[1]!s:<8} grandparent={row[2]}")

    lookup = dict(zip(result["name"], result["parent"], strict=True))
    grand = dict(zip(result["name"], result["grandparent"], strict=True))

    # The root has neither.
    assert lookup["root"] is None
    assert grand["root"] is None

    # A depth-1 node has a parent and no grandparent.
    assert lookup["eu"] == "root"
    assert grand["eu"] is None

    # A depth-3 node has both.
    assert lookup["paris"] == "france"
    assert grand["paris"] == "eu"

    # A left join at every level means no node is lost, whatever its depth.
    assert len(result["name"]) == nodes.count()


if __name__ == "__main__":
    main()
