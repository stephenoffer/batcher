"""Writing and re-reading CSV, and the fidelity you lose on the way.

CSV has no types, so a round trip is only lossless if the reader's inference happens to
agree with what you wrote. Dates are where it usually does not. Assert the schema after
reading back rather than assuming it survived.

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
        .head(1_000)
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


if __name__ == "__main__":
    main()
