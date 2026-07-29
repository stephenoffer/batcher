"""Parsing file paths held in a column.

Object-storage listings arrive as one long URI per row. Splitting them in the engine keeps
the partition key, the extension, and the directory available as ordinary columns you can
group and filter by.

    python examples/expressions/strings_paths.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    listing = bt.from_pydict(
        {
            "uri": [
                "s3://lake/events/day=2024-01-01/part-0.parquet",
                "s3://lake/events/day=2024-01-02/part-1.parquet",
                "/local/data/raw/notes.txt",
            ],
        }
    )

    parsed = listing.with_columns(
        filename=col("uri").str.parse_filename(),
        # `parse_dirpath` is the full parent path; `parse_dirname` is the *root* segment
        # ("s3:" or "/"), not the immediate parent, so take the last path segment when
        # that is what you want.
        dirpath=col("uri").str.parse_dirpath(),
        dirname=col("uri").str.parse_dirpath().str.parse_filename(),
        segments=col("uri").str.parse_path(),
        # The partition value, recovered from the directory name.
        day=col("uri").str.extract(r"day=([0-9-]+)"),
        extension=col("uri").str.extract(r"\.([a-z0-9]+)$"),
    ).to_pydict()

    print(parsed)

    assert parsed["filename"] == ["part-0.parquet", "part-1.parquet", "notes.txt"]
    assert parsed["dirname"][0] == "day=2024-01-01"
    assert parsed["dirpath"][2] == "/local/data/raw"
    assert parsed["segments"][0][-1] == "part-0.parquet"
    # A non-matching `extract` yields an empty string, not null, so screen for
    # non-matches with `!= ""` rather than `.is_null()`.
    assert parsed["day"] == ["2024-01-01", "2024-01-02", ""]
    assert parsed["extension"] == ["parquet", "parquet", "txt"]

    # The grouping this exists for: how many files per extension?
    by_ext = (
        listing.group_by(ext=col("uri").str.extract(r"\.([a-z0-9]+)$"))
        .agg(n=bt.count())
        .sort("ext")
        .to_pydict()
    )
    print(by_ext)
    assert by_ext["ext"] == ["parquet", "txt"]
    assert by_ext["n"] == [2, 1]

    # Recovering the partition column a Hive-style read does not reconstruct for you.
    with_partition = (
        listing.filter(col("uri").str.extract(r"day=([0-9-]+)") != "")
        .select(
            day=col("uri").str.extract(r"day=([0-9-]+)"),
            file=col("uri").str.parse_filename(),
        )
        .to_pydict()
    )
    print(with_partition)
    assert with_partition["day"] == ["2024-01-01", "2024-01-02"]


if __name__ == "__main__":
    main()
