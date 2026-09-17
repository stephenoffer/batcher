"""The positional verbs `zip`, `split` and `transpose` against DuckDB, on every path.

All three take a row's position under an explicit `order_by`, because a relation has no row
order of its own. DuckDB is a usable oracle despite having none of the three methods: a
position under an order is ``row_number() OVER (ORDER BY ...)``, a part of a split is
``ORDER BY ... LIMIT ... OFFSET ...``, and a transposed column is a filtered ``max`` per input
column. Each verb's contract is positional, so every comparison here is order-sensitive
(`assert_same_ordered`); an order-independent helper would pass a result whose rows sit at the
wrong positions, which is the defect class these verbs actually have.

Every case runs ``collect()``, a spilled ``collect(num_partitions=4)`` and ``iter_batches()``,
over inputs with nulls, an empty relation, one row, duplicate payloads, a descending order,
and a ``multibatch`` shape past two morsels so a part boundary can fall inside a morsel.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered, assert_tables_equal

pytestmark = pytest.mark.differential

PATHS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True, num_partitions=4),
    "iter_batches": lambda ds: _stream(ds),
}


def _stream(ds: bt.Dataset) -> pa.Table:
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


def _rows(n: int, *, seed: int) -> pa.Table:
    """A unique shuffled order key `i`, a nullable payload with duplicates, a string column."""
    order = [(k * 7919 + seed) % n for k in range(n)] if n else []
    x = [None if k % 5 == 0 else (k * seed) % 4 for k in order]
    s = [None if k % 7 == 1 else f"s{k % 3}" for k in order]
    return pa.table(
        {"i": pa.array(order, pa.int64()), "x": pa.array(x, pa.int64()), "s": pa.array(s)}
    )


#: `n` for each shape; 7919 is prime and coprime with every `n`, so `i` is a permutation.
SIZES = {"base": 50, "empty": 0, "single": 1, "multibatch": 40_000}


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("shape", sorted(SIZES))
def test_zip_pairs_rows_by_position(duck, shape, descending, path):
    """Three inputs, each shuffled differently, paired by rank under the same order."""
    n = SIZES[shape]
    a, b, c = _rows(n, seed=3), _rows(n, seed=11), _rows(n, seed=17)
    for name, table in {"a": a, "b": b, "c": c}.items():
        duck.register(name, table)
    ds = bt.from_arrow(a).zip(
        bt.from_arrow(b), bt.from_arrow(c), order_by="i", descending=descending
    )
    direction = "DESC" if descending else "ASC"
    ranked = "SELECT *, row_number() OVER (ORDER BY i {d}) AS rn FROM {t}"
    expected = duck.sql(
        "SELECT a.i, a.x, a.s, b.i AS i_1, b.x AS x_1, b.s AS s_1, "
        "c.i AS i_2, c.x AS x_2, c.s AS s_2 "
        f"FROM ({ranked.format(d=direction, t='a')}) a "
        f"JOIN ({ranked.format(d=direction, t='b')}) b USING (rn) "
        f"JOIN ({ranked.format(d=direction, t='c')}) c USING (rn) ORDER BY rn"
    )
    out = PATHS[path](ds)
    assert out.column_names == ["i", "x", "s", "i_1", "x_1", "s_1", "i_2", "x_2", "s_2"]
    assert_same_ordered(out, expected)


def test_zip_refuses_different_row_counts():
    with pytest.raises(bt.PlanError, match="same number of rows"):
        bt.from_arrow(_rows(5, seed=3)).zip(bt.from_arrow(_rows(4, seed=3)), order_by="i")


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("equal", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("shape", sorted(SIZES))
def test_split_parts_hold_their_position_ranges(duck, shape, descending, equal, path):
    """Each part is the DuckDB rows at its positions; together they cover the input once."""
    n = SIZES[shape]
    table = _rows(n, seed=5)
    duck.register("t", table)
    parts = bt.from_arrow(table).split(3, order_by="i", descending=descending, equal=equal)
    assert len(parts) == 3
    base, extra = divmod(n, 3)
    sizes = [base] * 3 if equal else [base + (1 if k < extra else 0) for k in range(3)]
    direction = "DESC" if descending else "ASC"
    start = 0
    for part, size in zip(parts, sizes, strict=True):
        expected = duck.sql(f"SELECT * FROM t ORDER BY i {direction} LIMIT {size} OFFSET {start}")
        assert_same_ordered(PATHS[path](part), expected)
        start += size


def _wide(names: list[str | None]) -> pa.Table:
    """One row per name, with an int, a nullable int and a float column to transpose."""
    n = len(names)
    return pa.table(
        {
            "name": pa.array(names, pa.string()),
            "a": pa.array([k * 3 for k in range(n)], pa.int64()),
            "b": pa.array([None if k % 2 else k for k in range(n)], pa.int64()),
            "c": pa.array([k / 4 for k in range(n)], pa.float64()),
        }
    )


TRANSPOSE_SHAPES = {
    "base": _wide(["q", "p", "s", "r"]),
    "single": _wide(["only"]),
    "descending": _wide(["z", "y", "x"]),
}


def _transpose_sql(names: list[str], key: str) -> str:
    """Each value column as one row, holding its value per name as a filtered max."""
    selects = []
    for position, column in enumerate(["a", "b", "c"]):
        cells = ", ".join(
            f"max(CAST({column} AS DOUBLE)) FILTER (WHERE {key} = '{v}') AS \"{v}\"" for v in names
        )
        selects.append(f"SELECT {position} AS pos, '{column}' AS \"column\", {cells} FROM t")
    return " UNION ALL ".join(selects) + " ORDER BY pos"


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(TRANSPOSE_SHAPES))
def test_transpose_by_a_naming_column(duck, shape, path):
    """Names from a column's values, output columns in ascending name order."""
    table = TRANSPOSE_SHAPES[shape]
    duck.register("t", table)
    names = sorted(table.column("name").to_pylist())
    ds = bt.from_arrow(table).transpose(column_names="name", include_header=True)
    out = PATHS[path](ds)
    assert out.column_names == ["column", *names]
    assert_same_ordered(out, _quoted(duck, names))


def _quoted(duck, names: list[str]):
    """The DuckDB relation for `_transpose_sql`, projected to the header and the name columns."""
    cols = ", ".join(f'"{n}"' for n in names)
    return duck.sql(f'SELECT "column", {cols} FROM ({_transpose_sql(names, "name")})')


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("shape", sorted(TRANSPOSE_SHAPES))
def test_transpose_by_position(duck, shape, descending, path):
    """Positional names `column_0..` follow `order_by`, which is what ties a name to a row."""
    table = TRANSPOSE_SHAPES[shape]
    ranked = sorted(table.column("name").to_pylist(), reverse=descending)
    duck.register("t", table)
    ds = (
        bt.from_arrow(table)
        .drop("name")
        .transpose(order_by=bt.col("a"), descending=descending, include_header=True)
    )
    out = PATHS[path](ds)
    positional = [f"column_{k}" for k in range(len(ranked))]
    assert out.column_names == ["column", *positional]
    # `a` rises with the name's row position, so ranking by `a` ranks the rows in table order.
    by_row = table.column("name").to_pylist()
    in_order = sorted(by_row, key=by_row.index, reverse=descending)
    relation = _quoted(duck, in_order)
    renamed = relation.to_arrow_table().rename_columns(["column", *positional])
    assert_tables_equal(out, renamed, ordered=True)


def test_transpose_refuses_positional_names_without_an_order():
    with pytest.raises(bt.PlanError, match="explicit order"):
        bt.from_arrow(_wide(["p"])).drop("name").transpose()


def test_transpose_refuses_a_repeated_name():
    with pytest.raises(bt.PlanError, match="repeats a value"):
        bt.from_arrow(_wide(["p", "p"])).transpose(column_names="name")


def test_transpose_refuses_an_empty_input():
    with pytest.raises(bt.PlanError, match="no rows"):
        bt.from_arrow(_wide([])).transpose(column_names="name")
