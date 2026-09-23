"""Reading text that is not entirely well-formed.

Two behaviours worth knowing, and they pull in opposite directions. Type *inference* is
permissive: one non-numeric value in a column widens the whole column to a string, silently.
A *cast* is strict: handed that same value it raises rather than producing a null.

So the failure surfaces at the cast, not at the read — which means the way to tolerate bad
rows is to test them before converting, not to convert and hope.

A line with the wrong number of fields is the other kind of bad row, and there the read
itself refuses. ``on_bad_lines="skip"`` (or ``"warn"``) drops such a line and keeps the rest
of the file, which ``on_error="skip"`` would not: that drops the whole file.

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

        # A *structurally* bad line is a different failure: one field too many. The default
        # refuses the read and names the line, and `on_bad_lines` decides instead.
        ragged = str(Path(directory) / "ragged.csv")
        Path(ragged).write_text("id,name\n1,alice\n2,bob,EXTRA\n3,carol\n")
        try:
            bt.read.csv(ragged).count()
        except bt.FormatError as error:
            print("ragged line refused:", str(error)[:70])
            assert "on_bad_lines" in str(error)
        else:
            raise AssertionError("a line with an extra field must not parse silently")

        # "skip" drops the line and keeps every good row of the file.
        kept = bt.read.csv(ragged, on_bad_lines="skip").to_pydict()
        assert kept == {"id": [1, 3], "name": ["alice", "carol"]}

        # "warn" returns the same rows and logs each dropped line with its file.
        assert bt.read.csv(ragged, on_bad_lines="warn").to_pydict() == kept


if __name__ == "__main__":
    main()
