"""Differential-testing fixtures.

The core correctness strategy (per the plan): run the same query through Batcher
and through a trusted oracle (DuckDB), and assert the results are equal. The
interpreter is deterministic and built on arrow's typed kernels, so any
divergence from DuckDB is a real bug — and once the JIT tiers land, each tier is
checked against this same oracle.

The comparison helpers live in `tests/_harness.py`; import them from `_harness`. A bare
``from conftest import ...`` resolves to whichever `conftest` pytest imported first, which
breaks any run spanning two test directories.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture
def duck():
    """A fresh in-memory DuckDB connection."""
    con = duckdb.connect()
    yield con
    con.close()
