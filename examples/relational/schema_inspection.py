"""Reading a dataset's shape without reading its rows.

`schema`, `columns` and `dtypes` come from the plan, so they answer without executing.
`count`, `describe` and `glimpse` execute. Knowing which is which is what keeps an
exploratory session from accidentally scanning a terabyte to print a column list.

    python examples/relational/schema_inspection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem")

    # Metadata: no scan.
    print("columns:", lineitem.columns[:4], "...")
    print("width:", lineitem.width)
    assert lineitem.width == len(lineitem.columns) == 16
    assert lineitem.schema.names == lineitem.columns

    types = dict(zip(lineitem.columns, [str(dtype) for dtype in lineitem.dtypes], strict=True))
    print("l_shipdate is", types["l_shipdate"])
    assert types["l_orderkey"] == "int64"
    assert "date" in types["l_shipdate"]

    # `collect_schema` is the explicit form, returning a name-to-type mapping.
    resolved = lineitem.collect_schema()
    assert list(resolved) == lineitem.columns

    # These read data.
    print("rows:", lineitem.height)
    assert lineitem.height == lineitem.count()
    assert lineitem.shape == (lineitem.height, lineitem.width)
    assert not lineitem.is_empty()

    # A per-column statistical summary, one row per statistic.
    summary = lineitem.select("l_quantity", "l_discount").describe().to_pydict()
    print(summary["statistic"])
    assert "mean" in summary["statistic"]

    # A compact printout of types and first values.
    lineitem.select("l_orderkey", "l_shipmode").glimpse()


if __name__ == "__main__":
    main()
