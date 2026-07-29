"""CSV, JSON, and Arrow IPC round trips.

Text formats carry no schema, so types are inferred on read. That inference is the usual
source of a surprise: a zip code column of "01234" becomes an integer and loses the
leading zero. Read it, check the schema, and cast at the edge.

    python examples/io/text_formats.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt


def main() -> None:
    rows = bt.from_pydict(
        {
            "id": [1, 2, 3],
            "name": ["ada", "grace", "alan"],
            "score": [9.5, 8.25, 7.0],
        }
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # CSV.
        csv_path = root / "people.csv"
        rows.write.csv(str(csv_path))
        from_csv = bt.read.csv(str(csv_path))
        print("csv schema:", from_csv.schema)
        back = from_csv.to_pydict()
        assert back["name"] == ["ada", "grace", "alan"]
        assert back["score"] == [9.5, 8.25, 7.0]

        # JSON (newline-delimited).
        json_path = root / "people.json"
        rows.write.json(str(json_path))
        from_json = bt.read.json(str(json_path)).to_pydict()
        assert from_json["id"] == [1, 2, 3]

        # Arrow IPC: the schema travels with the data, so nothing is inferred.
        arrow_path = root / "people.arrow"
        rows.write.arrow(str(arrow_path))
        from_arrow = bt.read.arrow(str(arrow_path))
        assert from_arrow.schema == rows.schema
        assert from_arrow.to_pydict()["name"] == back["name"]

        # The inference trap, made concrete: a zero-padded id round-trips as an integer.
        codes = bt.from_pydict({"zip": ["01234", "00999"]})
        zip_path = root / "zips.csv"
        codes.write.csv(str(zip_path))
        inferred = bt.read.csv(str(zip_path)).to_pydict()
        print("inferred zips:", inferred["zip"])
        assert inferred["zip"] == [1234, 999]  # leading zeros gone

        # Arrow IPC keeps them, because the type is recorded rather than guessed.
        codes.write.arrow(str(root / "zips.arrow"))
        kept = bt.read.arrow(str(root / "zips.arrow")).to_pydict()
        assert kept["zip"] == ["01234", "00999"]


if __name__ == "__main__":
    main()
