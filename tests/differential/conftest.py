"""Differential-testing fixtures.

The core correctness strategy (per the plan): run the same query through Batcher
and through a trusted oracle (DuckDB), and assert the results are equal. The
interpreter is deterministic and built on arrow's typed kernels, so any
divergence from DuckDB is a real bug — and once the JIT tiers land, each tier is
checked against this same oracle.

The comparison helpers themselves live in `tests/_harness.py` so that a test can
import them by an unambiguous module name; they are re-exported here because this
is where they are documented and where the `duck` fixture lives. Import them from
`_harness` in new tests — a bare ``from conftest import ...`` resolves to whichever
`conftest` pytest imported first, which breaks any run spanning two test directories.
"""

from __future__ import annotations

import pytest

from _harness import (
    assert_same,
    assert_same_ordered,
    assert_tables_equal,
    duck_materialize,
)

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")

__all__ = [
    "assert_same",
    "assert_same_ordered",
    "assert_tables_equal",
    "duck_materialize",
]


@pytest.fixture
def duck():
    """A fresh in-memory DuckDB connection."""
    con = duckdb.connect()
    yield con
    con.close()
