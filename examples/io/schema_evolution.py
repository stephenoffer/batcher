"""Files whose schemas disagree: the three `schema_mode`s, and what each one returns.

A directory written over months drifts: a column is added, a type widens, a column is
dropped. `bt.read.parquet(path, schema_mode=...)` decides what the read makes of that.

- ``"strict"`` (the default): the first file's schema is the contract. A later file may
  carry extra columns (they are dropped, with a warning) or a type that converts to
  the contract's without changing a value, but a missing column or a value that would change
  raises `bt.SchemaError` naming the file.
- ``"union"``: every column of every file, each at the narrowest type that holds all of its
  values. A file without a column reads it as null.
- ``"latest"``: the newest file's columns and types. Older files are cast toward it, and a
  value that would not survive the cast raises.

A distributed read (`collect(distributed=True)`) answers exactly as the single-node read
does in every mode: the same rows, the same types, or the same error.

    python examples/io/schema_evolution.py
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import batcher as bt


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # The old generation has two columns; upstream later added `channel` and started
        # sending `amount` as a float.
        bt.from_pydict({"id": [1, 2], "amount": [10, 20]}).write.parquet(str(root / "p0.parquet"))
        bt.from_pydict({"id": [3], "amount": [30.5], "channel": ["web"]}).write.parquet(
            str(root / "p1.parquet")
        )

        # strict: file 0 is the contract, and 30.5 cannot become an int64 unchanged.
        try:
            bt.read.parquet(str(root)).collect(distributed=False)
        except bt.SchemaError as error:
            print("strict refused:", str(error).split(",")[0])
            assert "p1.parquet" in str(error)
        else:
            raise AssertionError("strict mode must refuse a value it would have to truncate")

        # union: every column, `amount` widened to float64, `channel` null where absent.
        union = (
            bt.read.parquet(str(root), schema_mode="union").sort("id").collect(distributed=False)
        )
        print(union.schema)
        assert union.column_names == ["id", "amount", "channel"]
        assert union.schema.field("amount").type == "double"
        assert union.to_pydict() == {
            "id": [1, 2, 3],
            "amount": [10.0, 20.0, 30.5],
            "channel": [None, None, "web"],
        }

        # latest: the newest file's shape. Its column order is `id, amount, channel` too.
        latest = (
            bt.read.parquet(str(root), schema_mode="latest").sort("id").collect(distributed=False)
        )
        assert latest.to_pydict() == union.to_pydict()

        # strict with only an *added* column reads every row and drops the column, loudly.
        added = root / "added"
        added.mkdir()
        bt.from_pydict({"id": [1]}).write.parquet(str(added / "p0.parquet"))
        bt.from_pydict({"id": [2], "channel": ["web"]}).write.parquet(str(added / "p1.parquet"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            narrowed = bt.read.parquet(str(added)).collect(distributed=False)
        assert narrowed.column_names == ["id"] and narrowed.num_rows == 2
        dropped_warning = "columns the read will not return: ['channel']"
        assert any(dropped_warning in str(w.message) for w in caught)

        # A dropped column is the one drift strict mode cannot absorb: the contract
        # promised it, and a file that lacks it raises.
        dropped = root / "dropped"
        dropped.mkdir()
        bt.from_pydict({"id": [1], "channel": ["web"]}).write.parquet(str(dropped / "p0.parquet"))
        bt.from_pydict({"id": [2]}).write.parquet(str(dropped / "p1.parquet"))
        try:
            bt.read.parquet(str(dropped)).collect(distributed=False)
        except bt.SchemaError as error:
            assert "missing column 'channel'" in str(error)
        else:
            raise AssertionError("strict mode must refuse a file missing a declared column")
        filled = bt.read.parquet(str(dropped), schema_mode="union").sort("id")
        assert filled.to_pydict()["channel"] == ["web", None]


if __name__ == "__main__":
    main()
