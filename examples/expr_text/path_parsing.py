"""Pulling the pieces out of a file path or URI.

Parsing a path with string slicing breaks on the first Windows separator or the first
query string. These do it properly, which matters because a path column is usually the
partition key in disguise.

    python examples/expr_text/path_parsing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    paths = bt.from_pydict(
        {
            "uri": [
                "s3://lake/events/day=2024-01-01/part-0.parquet",
                "s3://lake/events/day=2024-01-02/part-1.parquet",
                "/mnt/data/raw/readings.csv",
            ]
        }
    )

    parsed = paths.select(
        "uri",
        name=col("uri").str.parse_filename(),
        # `parse_dirpath` is the whole directory; `parse_dirname` is the *root* of the
        # path, which is the scheme for a URI. To get the innermost directory, take the
        # filename of the directory path.
        full_dir=col("uri").str.parse_dirpath(),
        root=col("uri").str.parse_dirname(),
    ).with_columns(directory=col("full_dir").str.parse_filename())
    result = parsed.to_pydict()
    print(result)

    assert result["name"] == ["part-0.parquet", "part-1.parquet", "readings.csv"]
    assert result["directory"][0] == "day=2024-01-01"

    # The partition value, extracted from the directory that encodes it.
    partitioned = parsed.with_columns(
        day=col("directory").str.extract(r"day=(\d{4}-\d{2}-\d{2})")
    ).to_pydict()
    print(partitioned["day"])
    assert partitioned["day"][:2] == ["2024-01-01", "2024-01-02"]
    # The third path carries no partition, so the extract yields an empty match.
    assert partitioned["day"][2] in ("", None)

    # The extension, for routing by format.
    extensions = parsed.select(ext=col("name").str.extract(r"\.(\w+)$")).to_pydict()
    assert extensions["ext"] == ["parquet", "parquet", "csv"]


if __name__ == "__main__":
    main()
