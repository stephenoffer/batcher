"""What is running: engine version, build profile, and an isolated session.

`bt.Session` gives a query its own catalog, which is what you want when two parts of a
process register tables under the same names. The version and profile are what a bug report
needs and what a benchmark must record.

    python examples/operations/session_and_versions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    print("engine version:", bt.engine_version())
    print("package version:", bt.__version__)
    assert isinstance(bt.engine_version(), str)
    assert bt.engine_version()

    orders = tpch("orders").select("o_orderkey", "o_totalprice")

    # The default catalog: a CREATE registers a name every later query can see.
    bt.sql("CREATE OR REPLACE TABLE shared AS SELECT * FROM orders", orders=orders)
    shared = bt.sql("SELECT COUNT(*) AS n FROM shared").to_pydict()
    print("default catalog:", shared["n"][0])
    assert shared["n"][0] == orders.count()

    # A Session has its own catalog, so the same name can mean something else.
    session = bt.Session()
    session.sql(
        "CREATE OR REPLACE TABLE shared AS SELECT * FROM orders WHERE o_totalprice > 200000",
        orders=orders,
    )
    scoped = session.sql("SELECT COUNT(*) AS n FROM shared").to_pydict()
    print("session catalog:", scoped["n"][0])
    assert scoped["n"][0] < shared["n"][0]

    # And the default catalog is untouched by the session's definition.
    still = bt.sql("SELECT COUNT(*) AS n FROM shared").to_pydict()
    assert still["n"][0] == shared["n"][0]

    bt.sql("DROP TABLE shared")


if __name__ == "__main__":
    main()
