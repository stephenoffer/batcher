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
from batcher._internal.errors import PlanError


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

    sqlglot rejects this at parse time, before the translator sees it. The assertion is
    that it fails loudly *as a Batcher error*: a parse failure is wrapped in `PlanError`
    so a caller catches one exception type for every plan-time problem rather than
    importing sqlglot to name its own. This test previously accepted the bare
    `sqlglot.errors.ParseError`, which is the leak that wrapping closes.

    The translator keeps its own check anyway: it runs on the parsed AST and must not
    assume the parser is the only caller, hence `NotImplementedError` is still allowed.
    """
    with pytest.raises((NotImplementedError, PlanError), match=r"[Aa]ggregat"):
        bt.sql("SELECT * FROM t PIVOT (v FOR k IN ('a'))", t=t).collect()


@pytest.fixture
def sparse(duck):
    """A wide table with a NULL cell in each measure column, plus an all-NULL row."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "a": pa.array([1, None, None], pa.int64()),
            "b": pa.array([None, 4, None], pa.int64()),
        }
    )
    duck.register("sparse", table)
    return table


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        # Bare UNPIVOT is EXCLUDE NULLS in the standard and in DuckDB: Batcher kept the
        # NULL rows and returned 6 where DuckDB returns 2.
        "SELECT * FROM sparse UNPIVOT (value FOR variable IN (a, b))",
        "SELECT * FROM sparse UNPIVOT EXCLUDE NULLS (value FOR variable IN (a, b))",
        "SELECT * FROM sparse UNPIVOT INCLUDE NULLS (value FOR variable IN (a, b))",
    ],
)
def test_unpivot_null_handling_matches_duckdb(duck, sparse, query):
    """UNPIVOT drops melted NULLs unless the query says INCLUDE NULLS."""
    got = bt.sql(query, sparse=sparse).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert len(got["id"]) == len(exp["id"])
    assert _norm(got) == _norm(exp)


@pytest.mark.differential
def test_bare_unpivot_over_empty_and_all_null(duck):
    """No input rows, or only NULL cells, unpivot to nothing."""
    for rows in ([], [None]):
        table = pa.table(
            {"id": pa.array(range(len(rows)), pa.int64()), "a": pa.array(rows, pa.int64())}
        )
        duck.register("e", table)
        query = "SELECT * FROM e UNPIVOT (value FOR variable IN (a))"
        got = bt.sql(query, e=table).collect()
        assert got.num_rows == duck.sql(query).to_arrow_table().num_rows == 0


@pytest.mark.differential
@pytest.mark.parametrize(
    ("ktype", "in_list"),
    [(pa.int64(), "(1, 2)"), (pa.int64(), "(2, -1)"), (pa.float64(), "(1.5, 2.0)")],
)
def test_pivot_on_a_numeric_key_matches_duckdb(duck, ktype, in_list):
    """A numeric pivot key compares against typed IN values, not their text.

    The IN list used to be stringified, so `k = '1'` met an Int64 column and the engine
    refused it with "Int64 == Utf8". The output column is still named by the value's text.
    """
    keys = [1, 2, 1, -1, 2] if ktype == pa.int64() else [1.5, 2.0, 1.5, 3.0, 2.0]
    table = pa.table(
        {
            "i": pa.array([1, 1, 2, 2, 2], pa.int64()),
            "k": pa.array(keys, ktype),
            "v": pa.array([10, 20, None, 40, 50], pa.int64()),
        }
    )
    duck.register("nk", table)
    query = f"SELECT * FROM nk PIVOT (sum(v) FOR k IN {in_list})"
    got = bt.sql(query, nk=table).collect().to_pydict()
    exp = duck.sql(query).to_arrow_table().to_pydict()
    assert sorted(got) == sorted(exp)
    assert _norm({c: got[c] for c in sorted(got)}) == _norm({c: exp[c] for c in sorted(exp)})


def test_pivot_in_list_of_the_wrong_type_is_a_plan_error():
    """A string IN value against an integer key is refused at plan time, by name."""
    table = pa.table({"i": [1], "k": pa.array([1], pa.int64()), "v": [1]})
    with pytest.raises(PlanError, match="cannot match column 'k'"):
        bt.sql("SELECT * FROM nk PIVOT (sum(v) FOR k IN ('1'))", nk=table).collect()
