"""Reading text that is not entirely well-formed.

Two behaviours worth knowing, and they pull in opposite directions. Type *inference* is
permissive: one non-numeric value in a column widens the whole column to a string, silently.
A *cast* is strict: handed that same value it raises rather than producing a null.

So the failure surfaces at the cast, not at the read — which means the way to tolerate bad
rows is to test them before converting, not to convert and hope.

    python examples/io/csv_error_tolerance.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    rows = [
        "id,name,score",
        "1,alice,90",
        "2,bob,85",
        "3,carol,not-a-number",
        "4,dave,70",
    ]

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "scores.csv")
        Path(path).write_text("\n".join(rows) + "\n")

        # Inference widens the column rather than failing.
        loaded = bt.read.csv(path)
        print(loaded.schema)
        assert loaded.count() == 4

        types = dict(zip(loaded.columns, [str(t) for t in loaded.dtypes], strict=True))
        print("score inferred as:", types["score"])
        assert types["score"] == "string"

        # The cast is strict: it raises rather than nulling the bad value.
        try:
            loaded.select(score=col("score").cast("int64")).count()
        except Exception as error:
            print("cast refused:", str(error)[:60])
        else:
            raise AssertionError("casting a non-numeric string must fail")

        # So test first, then convert what passes.
        numeric = loaded.filter(col("score").str.is_numeric())
        bad = loaded.filter(~col("score").str.is_numeric())
        print(f"{numeric.count()} parseable, {bad.count()} not")
        assert numeric.count() == 3
        assert bad.count() == 1

        parsed = numeric.select("id", "name", score=col("score").cast("int64"))
        values = parsed.to_pydict()["score"]
        print("parsed scores:", values)
        assert values == [90, 85, 70]

        # The rejected rows are a quarantine set, not a silent loss.
        assert bad.to_pydict()["name"] == ["carol"]


if __name__ == "__main__":
    main()
