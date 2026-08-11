"""Cloud paths: the schemes, the globs, and what format inference can see.

Format inference reads the extension off the literal part of the path and stops at the first
`*`. So a globbed path has nothing to infer from and needs a typed reader — which is the one
rule that catches everybody once.

    python examples/io/cloud_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_uri
from batcher import col


def main() -> None:
    uri = tpch_uri("nation")
    print("reading", uri)

    # A typed reader works with a glob.
    nations = bt.read.parquet(uri)
    assert nations.count() == 25

    # The untyped `bt.read` has nothing to infer from once the path contains a star.
    try:
        bt.read(uri).count()
    except Exception as error:
        print("inference refused:", type(error).__name__, str(error)[:70])
    else:
        print("this build inferred the format from a globbed path")

    # A literal path carries its extension, so inference works.
    literal = "s3://ray-benchmark-data/tpch/parquet/sf1/nation/nation.parquet"
    inferred = bt.read(literal)
    assert inferred.count() == 25

    # Anonymous access needs no credentials configured, which is what makes this corpus
    # usable as a fixture.
    named = inferred.select(nationkey=col("column0"), name=col("column1")).sort("nationkey")
    result = named.head(3).to_pydict()
    print(result)
    assert result["name"][0] == "ALGERIA"

    # The scheme picks the filesystem and nothing else changes: the same query shape works
    # against a local path.
    assert named.count() == 25
    assert named.agg(n=col("nationkey").max()).to_pydict()["n"][0] == 24


if __name__ == "__main__":
    main()
