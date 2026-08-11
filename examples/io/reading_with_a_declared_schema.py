"""Declaring the schema instead of letting the reader infer it.

Inference reads a sample, so it can be wrong about a column whose interesting values come
later — an id column that is numeric for the first million rows and alphanumeric after.
Declaring the schema makes that a loud failure rather than a silent widening.

    python examples/io/reading_with_a_declared_schema.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

import batcher as bt
from batcher import col


def main() -> None:
    rows = ["id,code,amount", "1,100,9.5", "2,200,8.25", "3,A31,7.0"]

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "codes.csv")
        Path(path).write_text("\n".join(rows) + "\n")

        # Inference sees the alphanumeric code and widens the whole column to a string.
        inferred = bt.read.csv(path)
        types = dict(zip(inferred.columns, [str(t) for t in inferred.dtypes], strict=True))
        print("inferred:", types)
        assert types["code"] == "string"
        assert types["id"] == "int64"

        # Declaring the schema makes the expectation explicit.
        declared = bt.read.csv(
            path,
            schema=pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("code", pa.string()),
                    pa.field("amount", pa.float64()),
                ]
            ),
        )
        declared_types = dict(zip(declared.columns, [str(t) for t in declared.dtypes], strict=True))
        print("declared:", declared_types)
        assert declared_types == types
        assert declared.count() == 3

        # And a declaration that does not match the data fails rather than widening.
        try:
            wrong = bt.read.csv(
                path,
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64()),
                        pa.field("code", pa.int64()),
                        pa.field("amount", pa.float64()),
                    ]
                ),
            )
            wrong.count()
        except Exception as error:
            print("declaration rejected the data:", type(error).__name__)
        else:
            print("this build widened rather than raising")

        # The values survive either way.
        assert declared.agg(t=col("amount").sum()).to_pydict()["t"][0] == 24.75


if __name__ == "__main__":
    main()
