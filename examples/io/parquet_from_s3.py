"""Reading Parquet straight from S3, with no download step.

A Parquet file on object storage is read the same way as one on disk: the path scheme
picks the filesystem and nothing else changes. What does change is that every byte costs a
round trip, which is why the projection and the predicate below matter more here.

    python examples/io/parquet_from_s3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_uri
from batcher import col


def main() -> None:
    # The public TPC-H mirror. Anonymous access, no credentials configured.
    uri = tpch_uri("region")
    print("reading", uri)

    region = bt.read.parquet(uri)
    print(region.schema)

    # This mirror stores columns positionally, so the names are column0..columnN. Real
    # tables carry their names; this one is worth renaming on the way in.
    named = region.select(
        regionkey=col("column0"),
        name=col("column1"),
    ).sort("regionkey")

    result = named.to_pydict()
    print(result)

    assert result["name"] == ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
    assert result["regionkey"] == [0, 1, 2, 3, 4]

    # A glob reads every matching object as one dataset.
    nations = bt.read.parquet(tpch_uri("nation"))
    print("nations:", nations.count())
    assert nations.count() == 25


if __name__ == "__main__":
    main()
