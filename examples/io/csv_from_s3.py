"""Reading delimited text from S3, including files that carry no header.

The TPC-H text mirror is `dbgen` output: pipe-delimited, no header, and one trailing
delimiter per line so every row ends with an empty field. Every one of those is a reader
option rather than a cleanup pass afterwards.

    python examples/io/csv_from_s3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch_csv_uri
from batcher import col


def main() -> None:
    uri = tpch_csv_uri("nation")
    print("reading", uri)

    raw = bt.read.csv(uri, delimiter="|", has_header=False)
    print(raw.schema)

    # No header means generated names, and the trailing delimiter leaves a final
    # all-null field. Name the columns you want and drop the rest.
    nation = raw.select(
        nationkey=col("f0"),
        name=col("f1"),
        regionkey=col("f2"),
    ).sort("nationkey")

    result = nation.to_pydict()
    print(result["name"][:5])

    assert nation.count() == 25
    assert result["name"][0] == "ALGERIA"
    assert result["nationkey"] == list(range(25))
    # Types are inferred from the text, not left as strings.
    assert all(isinstance(value, int) for value in result["regionkey"])


if __name__ == "__main__":
    main()
