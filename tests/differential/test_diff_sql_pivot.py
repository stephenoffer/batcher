"""SQL `PIVOT` / `UNPIVOT` vs DuckDB.

`PIVOT (agg(v) FOR k IN ('a','b'))` widens a relation: one output column per listed `k`
value, each holding `agg(v)` over the rows sharing the remaining columns. `UNPIVOT` is the
inverse. Both are exactly the relational `Dataset.pivot` / `Dataset.unpivot` the engine
already has, so the SQL modifier now maps onto them instead of raising "use the
Dataset.pivot(...) method".

The case worth pinning is a value present in the data but *absent* from the `IN` list: it
must be dropped, not silently folded into another column.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    # `g` is the surviving index column; ('b','y') is missing so a NULL cell appears.
    table = pa.table(
        {
            "k": ["a", "a", "b", "b"],
            "g": ["x", "y", "x", "x"],
            "v": [1, 2, 3, 4],
        }
    )
    duck.register("t", table)
    return table


def _norm(d):
    n = len(next(iter(d.values()))) if d else 0
    return sorted([tuple(str(col[i]) for col in d.values()) for i in range(n)], key=str)


@pytest.mark.differential
@pytest.mark.parametrize("agg", ["sum", "min", "max", "count"])
def test_pivot_matches_duckdb(duck, t, agg):
    """Each aggregate widens identically to DuckDB, NULL where a cell has no rows."""
    query = f"SELECT * FROM t PIVOT ({agg}(v) FOR k IN ('a','b'))"
    got = bt.sql(query, t=t).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_pivot_drops_values_not_listed(duck, t):
    """A `k` value absent from the IN list must be dropped, not merged elsewhere."""
    query = "SELECT * FROM t PIVOT (sum(v) FOR k IN ('a'))"
    got = bt.sql(query, t=t).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert "b" not in got
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_unpivot_matches_duckdb(duck, t):
    """UNPIVOT narrows back to (name, value) pairs."""
    query = "SELECT * FROM t UNPIVOT (val FOR name IN (v))"
    got = bt.sql(query, t=t).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_unpivot_several_columns(duck):
    """Several measure columns unpivot into one name/value pair per column."""
    table = pa.table({"id": [1, 2], "a": [10, 20], "b": [30, 40]})
    duck.register("wide", table)
    query = "SELECT * FROM wide UNPIVOT (val FOR name IN (a, b))"
    got = bt.sql(query, wide=table).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert _norm(got) == _norm(exp)


def test_pivot_with_a_non_aggregate_rejects(t):
    """PIVOT's expression must be an aggregate — a bare column cannot widen.

    sqlglot rejects this at parse time, before the translator sees it, so the assertion is
    simply that it fails loudly. The translator keeps its own check anyway: it runs on the
    parsed AST and must not assume the parser is the only caller.
    """
    import sqlglot.errors

    with pytest.raises((NotImplementedError, sqlglot.errors.ParseError), match=r"[Aa]ggregat"):
        bt.sql("SELECT * FROM t PIVOT (v FOR k IN ('a'))", t=t).collect()
