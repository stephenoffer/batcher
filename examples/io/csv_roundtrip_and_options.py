"""Writing and re-reading CSV, and the fidelity you lose on the way.

CSV has no types, so a round trip is only lossless if the reader's inference happens to
agree with what you wrote. Dates are where it usually does not. Assert the schema after
reading back rather than assuming it survived.

The second half covers the reader options a real export needs: a null token written with
`null_value=` and read back with `null_values=`, a preamble skipped with `skip_rows=`, a
headerless file named with `names=`, quoted fields, a non-default quote character, and a
non-UTF-8 encoding.

    python examples/io/csv_roundtrip_and_options.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    orders = (
        tpch("orders")
        .select("o_orderkey", "o_orderdate", "o_totalprice", "o_orderpriority")
        .limit(1_000)
    )

    with tempfile.TemporaryDirectory() as directory:
        target = str(Path(directory) / "orders.csv")
        orders.write.csv(target)

        back = bt.read.csv(target)
        print(back.schema)
        assert back.count() == orders.count()
        assert back.columns == orders.columns

        # Numbers and dates survive inference here.
        original = orders.to_pydict()
        restored = back.to_pydict()
        assert restored["o_orderkey"] == original["o_orderkey"]
        assert restored["o_orderpriority"] == original["o_orderpriority"]

        # Floating point round-trips through text within representation error.
        assert all(
            abs(left - right) < 1e-6
            for left, right in zip(original["o_totalprice"], restored["o_totalprice"], strict=True)
        )

        # A non-default delimiter, written and read back.
        tsv = str(Path(directory) / "orders.tsv")
        orders.write.csv(tsv, delimiter="\t")
        tabbed = bt.read.csv(tsv, delimiter="\t")
        assert tabbed.count() == orders.count()

        # Reading it with the wrong delimiter collapses every row into one column.
        wrong = bt.read.csv(tsv)
        assert wrong.width < tabbed.width

        # A null token: written bare, so it reads back as a null in a still-numeric column.
        # A real string equal to the token is quoted on the way out, so it survives as text.
        gaps = bt.from_pydict({"k": [1, None, 3], "note": ["ok", None, "NULL"]})
        nulls = str(Path(directory) / "nulls.csv")
        gaps.write.csv(nulls, null_value="NULL")
        print(Path(nulls).read_text())
        reread = bt.read.csv(nulls, null_values="NULL")
        assert reread.to_pydict() == {"k": [1, None, 3], "note": ["ok", None, "NULL"]}
        assert str(reread.dtypes[0]) == "int64"

        # A headerless export with a preamble: skip the preamble, then name the columns.
        raw = Path(directory) / "export.csv"
        raw.write_text("exported 2024-01-01\nsystem: billing\n7,paid\n8,open\n")
        named = bt.read.csv(str(raw), skip_rows=2, header=None, names=["invoice", "status"])
        assert named.to_pydict() == {"invoice": [7, 8], "status": ["paid", "open"]}

        # Quoting: a comma and a line break inside a quoted field stay inside the field.
        quoted = Path(directory) / "quoted.csv"
        quoted.write_text('id,comment\n1,"late, again"\n2,"two\nlines"\n')
        comments = bt.read.csv(str(quoted)).to_pydict()["comment"]
        assert comments == ["late, again", "two\nlines"]

        # A single-quoted file says so.
        single = Path(directory) / "single.csv"
        single.write_text("id,comment\n1,'late, again'\n")
        assert bt.read.csv(str(single), quote_char="'").to_pydict()["comment"] == ["late, again"]

        # Encoding: a Latin-1 export decodes when told what it is.
        latin = Path(directory) / "latin.csv"
        latin.write_bytes("city\nZ\u00fcrich\n".encode("latin-1"))
        assert bt.read.csv(str(latin), encoding="latin-1").to_pydict() == {"city": ["Z\u00fcrich"]}


if __name__ == "__main__":
    main()
