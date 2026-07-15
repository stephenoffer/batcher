"""Parser-level contract for SQL DML and ordered-set aggregates (no engine run).

These assert the *control-plane* behavior: DML dispatches to a catalog rebind, bad
statements raise a clean typed error before anything executes, and the WITHIN GROUP
rewrite reshapes the AST so the ordered column is not dropped. Result correctness is
pinned by ``tests/differential/test_diff_sql4_dml.py``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _session() -> bt.Session:
    s = bt.Session()
    s.register("t", pa.table({"x": pa.array([1, 2, 3], pa.int64()), "y": [10, 20, 30]}))
    return s


@pytest.mark.unit
@pytest.mark.parametrize(
    "dml",
    [
        "INSERT INTO t VALUES (4, 40)",
        "DELETE FROM t WHERE x > 1",
        "UPDATE t SET y = 0",
    ],
)
def test_dml_rebinds_catalog_lazily(dml):
    # DML rebinds the target to a *new* lazy Dataset preserving the schema; nothing
    # executes at parse time (control-plane only).
    s = _session()
    before = s.table("t")
    s.sql(dml)
    after = s.table("t")
    assert after is not before  # the name was rebound
    assert after.columns == before.columns == ["x", "y"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dml", "exc"),
    [
        ("INSERT INTO t VALUES (1)", PlanError),
        ("INSERT INTO t (x) VALUES (1, 2)", PlanError),
        ("INSERT INTO t (nope) VALUES (1)", PlanError),
        ("UPDATE t SET z = 1", PlanError),
        ("INSERT INTO missing VALUES (1, 1)", PlanError),
        ("INSERT INTO t VALUES (1, 1) ON CONFLICT DO NOTHING", NotImplementedError),
        ("INSERT INTO t VALUES (1, 1) RETURNING x", NotImplementedError),
        ("DELETE FROM t WHERE x = 1 RETURNING x", NotImplementedError),
    ],
)
def test_bad_dml_raises_clean(dml, exc):
    with pytest.raises(exc):
        _session().sql(dml)


@pytest.mark.unit
def test_within_group_rewrite_keeps_ordered_column():
    # The rewrite turns WithinGroup(PercentileCont(frac), ORDER BY col) into the
    # two-arg PercentileCont(col, frac); without it the ORDER BY column vanishes.
    import sqlglot
    from sqlglot import expressions as exp

    from batcher._sql.parser.core_utils import _within_group_to_agg

    ast = sqlglot.parse_one(
        "SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY x) FROM t", read="duckdb"
    )
    rewritten = ast.transform(_within_group_to_agg)
    pc = rewritten.expressions[0]
    assert isinstance(pc, exp.PercentileCont)
    assert isinstance(pc.this, exp.Column) and pc.this.name == "x"  # column preserved
    assert pc.expression.name == "0.25"  # fraction preserved


@pytest.mark.unit
def test_mode_within_group_rewrite():
    import sqlglot
    from sqlglot import expressions as exp

    from batcher._sql.parser.core_utils import _within_group_to_agg

    ast = sqlglot.parse_one("SELECT mode() WITHIN GROUP (ORDER BY x) FROM t", read="duckdb")
    mode = ast.transform(_within_group_to_agg).expressions[0]
    assert isinstance(mode, exp.Mode)
    assert isinstance(mode.this, exp.Column) and mode.this.name == "x"


@pytest.mark.unit
def test_percentile_disc_within_group_rejected():
    import sqlglot

    from batcher._sql.parser.core_utils import _within_group_to_agg

    ast = sqlglot.parse_one(
        "SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY x) FROM t", read="duckdb"
    )
    with pytest.raises(NotImplementedError):
        ast.transform(_within_group_to_agg)
