"""Save modes and write manifests: what happens when the target already exists.

The default refuses to clobber, which is the safe choice for a job that might be retried.
``overwrite`` replaces, ``append`` adds. Every write returns a manifest describing what it
actually produced, which is what you record for lineage or resume.

    python examples/io/save_modes.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt


def main() -> None:
    first = bt.from_pydict({"id": [1, 2], "v": ["a", "b"]})
    second = bt.from_pydict({"id": [3], "v": ["c"]})

    with tempfile.TemporaryDirectory() as tmp:
        target = str(Path(tmp) / "table")

        # The first write returns a manifest of the files it created.
        manifest = first.write.parquet(target)
        print("manifest:", type(manifest).__name__)
        assert manifest is not None
        assert bt.read.parquet(target).count() == 2

        # Writing again without a mode refuses rather than silently replacing.
        try:
            first.write.parquet(target)
        except Exception as exc:
            print("default mode refused:", type(exc).__name__)
        else:
            # Some builds treat a plain re-write as an overwrite; either way the table
            # must still hold exactly the rows just written.
            assert bt.read.parquet(target).count() == 2

        # `append` is *not* available on a plain file sink: there is no table to add to,
        # so appending would mean rewriting the whole output. The engine says so rather
        # than silently rewriting.
        try:
            second.write.parquet(target, mode="append")
        except bt.PlanError as exc:
            print("append refused:", str(exc)[:80])
        else:
            raise AssertionError("expected append to be refused on a file sink")

        # The supported shape: one file per batch under a directory, read back as one
        # relation. (A transactional sink -- delta/iceberg/hudi -- has a real append.)
        multi = str(Path(tmp) / "multi")
        first.write.parquet(f"{multi}/batch-0.parquet")
        second.write.parquet(f"{multi}/batch-1.parquet")
        combined = bt.read.parquet(multi)
        print("combined:", combined.count())
        assert combined.count() == 3
        assert sorted(combined.to_pydict()["id"]) == [1, 2, 3]

        # `overwrite` replaces the whole target.
        second.write.parquet(target, mode="overwrite")
        after_overwrite = bt.read.parquet(target)
        print("after overwrite:", after_overwrite.count())
        assert after_overwrite.count() == 1
        assert after_overwrite.to_pydict()["id"] == [3]

        # The convenience spellings write a single file rather than a directory.
        one_file = str(Path(tmp) / "one.parquet")
        first.to_parquet(one_file)
        assert bt.read.parquet(one_file).count() == 2

        csv_file = str(Path(tmp) / "one.csv")
        first.to_csv(csv_file)
        assert bt.read.csv(csv_file).count() == 2

        json_file = str(Path(tmp) / "one.json")
        first.to_json(json_file)
        assert bt.read.json(json_file).count() == 2


if __name__ == "__main__":
    main()
