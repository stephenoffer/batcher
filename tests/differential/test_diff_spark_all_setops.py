"""Spark's ``exceptAll``/``intersectAll`` port to ``distinct=False``, and only that, vs DuckDB.

The migration hint and the registry used to send ``exceptAll`` to a bare ``except_`` and
``intersectAll`` to a bare ``intersect``. Both default to ``distinct=True``, which is SQL's
``EXCEPT``/``INTERSECT`` (Spark's ``subtract``/``intersect``), so a ported ``exceptAll``
silently dropped every duplicate. These cases pin the spelling the hint now gives against
DuckDB's ``EXCEPT ALL``/``INTERSECT ALL`` on multi-column fixtures with repeated rows, NULL
keys, a one-row side and an empty side, and show the bare default really is the DISTINCT
form, which is the reason the hint has to name the keyword.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.api.dataset.compat.guidance._dataset_table import DATASET_UNSUPPORTED

pytestmark = pytest.mark.differential

_A = pa.table(
    {
        "k": pa.array([1, 1, 1, 2, 2, None, None, 3], pa.int64()),
        "s": pa.array(["x", "x", "x", "y", "y", None, None, "z"], pa.string()),
    }
)
_B = pa.table(
    {
        "k": pa.array([1, 2, 2, 2, None, 4], pa.int64()),
        "s": pa.array(["x", "y", "y", "y", None, "w"], pa.string()),
    }
)

#: (left, right) fixture pairs: duplicates on both sides, a one-row right side, an empty
#: right side, and an empty left side.
_PAIRS = {
    "duplicates": (_A, _B),
    "one_row": (_A, _B.slice(0, 1)),
    "empty_right": (_A, _B.slice(0, 0)),
    "empty_left": (_A.slice(0, 0), _B),
}

_OPS = [("except_", "EXCEPT", "exceptAll"), ("intersect", "INTERSECT", "intersectAll")]


@pytest.mark.parametrize("pair", sorted(_PAIRS))
@pytest.mark.parametrize(("method", "sql", "spark"), _OPS)
def test_all_form_matches_duckdb(duck, pair: str, method: str, sql: str, spark: str) -> None:
    """``ds.<op>(other, distinct=False)`` keeps multiplicity exactly as ``<OP> ALL`` does."""
    a, b = _PAIRS[pair]
    duck.register("a", a)
    duck.register("b", b)
    got = getattr(bt.from_arrow(a), method)(bt.from_arrow(b), distinct=False).collect()
    assert_same(got, duck.sql(f"SELECT k, s FROM a {sql} ALL SELECT k, s FROM b"))
    assert f"ds.{method}(other, distinct=False)" in DATASET_UNSUPPORTED[spark]


@pytest.mark.parametrize(("method", "sql", "_spark"), _OPS)
def test_bare_default_is_the_distinct_form(duck, method: str, sql: str, _spark: str) -> None:
    """The default answers ``<OP>`` (DISTINCT), so it is not a port of the ALL form."""
    duck.register("a", _A)
    duck.register("b", _B)
    default = getattr(bt.from_arrow(_A), method)(bt.from_arrow(_B)).collect()
    assert_same(default, duck.sql(f"SELECT k, s FROM a {sql} SELECT k, s FROM b"))
    every = getattr(bt.from_arrow(_A), method)(bt.from_arrow(_B), distinct=False).collect()
    assert every.num_rows > default.num_rows, "the fixture must tell the two forms apart"
