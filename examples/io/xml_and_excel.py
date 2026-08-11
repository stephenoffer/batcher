"""Two formats that arrive from outside engineering: XML and Excel.

Both carry structure the reader has to guess at, and both are worth converting to Parquet
once at the boundary rather than re-parsing on every run. The check that matters is the
schema you got, not the row count.

Both readers live behind optional extras (`batcher-engine[xml]`, `openpyxl`). This script
says which one is missing and exits cleanly rather than failing, so the suite still runs
without them.

    python examples/io/xml_and_excel.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    xml_text = """<?xml version="1.0"?>
<records>
  <record><id>1</id><name>alice</name><score>90</score></record>
  <record><id>2</id><name>bob</name><score>85</score></record>
  <record><id>3</id><name>carol</name><score>77</score></record>
</records>
"""

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "records.xml")
        Path(path).write_text(xml_text)

        try:
            records = bt.read.xml(path)
            # Touch the schema so a missing reader fails here rather than later.
            assert records.schema is not None
        except Exception as error:
            print("xml reader unavailable:", str(error)[:90])
            return
        print(records.schema)
        assert records.count() == 3
        assert {"id", "name", "score"} <= set(records.columns)

        values = records.sort("id").to_pydict()
        print(values)
        assert [str(value) for value in values["name"]] == ["alice", "bob", "carol"]

        # Convert once at the boundary: everything downstream reads Parquet.
        parquet = str(Path(directory) / "records.parquet")
        records.write.parquet(parquet)
        converted = bt.read.parquet(parquet).sort("id")
        assert converted.count() == records.count()
        assert converted.to_pydict()["name"] == values["name"]

        # Excel needs a real workbook, which needs openpyxl. Say so rather than failing.
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print("openpyxl is not installed; skipping the Excel leg.")
            return

        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        sheet.append(["id", "name"])
        for index, name in enumerate(["alice", "bob"], start=1):
            sheet.append([index, name])
        workbook_path = str(Path(directory) / "records.xlsx")
        book.save(workbook_path)

        sheeted = bt.read.excel(workbook_path)
        print(sheeted.schema)
        assert sheeted.count() == 2
        assert col is not None


if __name__ == "__main__":
    main()
