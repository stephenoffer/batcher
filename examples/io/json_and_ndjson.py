"""JSON on the way out and back, and why newline-delimited is the one to write.

A single JSON array has to be parsed as one document, so it cannot be split across
workers. Newline-delimited JSON can be split at any newline, which is what makes it the
format to write when something else will read it in parallel.

    python examples/io/json_and_ndjson.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_name", "c_acctbal").head(500)

    with tempfile.TemporaryDirectory() as directory:
        target = str(Path(directory) / "customers.json")
        customer.write.json(target)

        back = bt.read.json(target)
        assert back.count() == customer.count()
        assert set(back.columns) == set(customer.columns)

        original = customer.sort("c_custkey").to_pydict()
        restored = back.sort("c_custkey").to_pydict()
        assert restored["c_name"] == original["c_name"]

        # The file is newline-delimited: one object per line, so line count == row count.
        written = Path(target)
        if written.is_dir():
            written = next(written.rglob("*.json"))
        lines = [line for line in written.read_text().splitlines() if line.strip()]
        print("first line:", lines[0][:80])
        assert len(lines) == customer.count()
        assert lines[0].startswith("{")


if __name__ == "__main__":
    main()
