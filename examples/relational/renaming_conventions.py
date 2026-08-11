"""Keeping column names sane through a multi-join pipeline.

Prefixing every table's columns at the source is the cheapest naming convention there is: no
collisions, no suffixes, and every column says where it came from. TPC-H already does it,
which is why its joins need no disambiguation.

    python examples/relational/renaming_conventions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # TPC-H prefixes every column with its table, so a five-way join collides on nothing.
    joined = (
        tpch("lineitem")
        .join(tpch("orders"), left_on="l_orderkey", right_on="o_orderkey")
        .join(tpch("customer"), left_on="o_custkey", right_on="c_custkey")
        .join(tpch("nation"), left_on="c_nationkey", right_on="n_nationkey")
        .join(tpch("region"), left_on="n_regionkey", right_on="r_regionkey")
    )
    print("columns after five joins:", joined.width)
    assert not any(name.endswith("_right") for name in joined.columns)
    assert len(set(joined.columns)) == len(joined.columns)

    # Without the convention, the same joins collide.
    plain_left = tpch("nation").select(col("n_nationkey").alias("id"), col("n_name").alias("name"))
    plain_right = tpch("region").select(col("r_regionkey").alias("id"), col("r_name").alias("name"))
    collided = plain_left.join(plain_right, on="id")
    print("collided columns:", collided.columns)
    assert any(name.endswith("_right") for name in collided.columns)

    # Applying the convention retrospectively is a rename, which is metadata only.
    prefixed = plain_right.rename({"id": "region_id", "name": "region_name"})
    clean = plain_left.join(prefixed, left_on="id", right_on="region_id")
    print("clean columns:", clean.columns)
    assert not any(name.endswith("_right") for name in clean.columns)

    # A programmatic prefix, for a table whose columns you do not know ahead of time.
    def prefix(dataset: bt.Dataset, tag: str) -> bt.Dataset:
        return dataset.rename({name: f"{tag}_{name}" for name in dataset.columns})

    tagged = prefix(tpch("supplier"), "sup")
    assert all(name.startswith("sup_") for name in tagged.columns)
    assert tagged.count() == tpch("supplier").count()


if __name__ == "__main__":
    main()
