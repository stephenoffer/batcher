"""`lookup_join` must produce exactly what the equivalent SQL join produces.

A lookup join reaches the dimension by point lookup instead of by scan, but that is a
*strategy*, not a semantics: the rows it returns have to be the rows a LEFT or INNER JOIN
returns. This file holds it to DuckDB on the shapes where a join implementation goes wrong
— an unmatched key, a null key, a duplicated key on the probe side, a stored null, and an
empty input.

The store is `InMemoryLookup`, so what is under test is the join, not a driver. The key is
rendered as a string on the way into the store (that is what a key-value store keys on),
so the dimension here is keyed by strings and DuckDB is given the same.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

bt = pytest.importorskip("batcher")
duckdb = pytest.importorskip("duckdb")

from _harness import assert_same  # noqa: E402

_DIM = {
    "k": ["c1", "c2", "c3"],
    "name": ["Ann", "Bob", None],
    "tier": [1, 2, 3],
}


@pytest.fixture(autouse=True)
def _in_memory_store(monkeypatch):
    """Point `lookup_join` at an in-process store holding `_DIM`."""
    from batcher.io.lookup import backends, spec

    dim = pa.table(_DIM)
    monkeypatch.setattr(
        spec, "build_lookup", lambda uri, schema, options: backends.InMemoryLookup(dim, "k")
    )


@pytest.fixture
def duck():
    con = duckdb.connect()
    con.execute("create table dim (k varchar, name varchar, tier bigint)")
    con.executemany(
        "insert into dim values (?, ?, ?)",
        list(zip(_DIM["k"], _DIM["name"], _DIM["tier"], strict=True)),
    )
    yield con
    con.close()


def _load(con, name: str, rows: dict) -> None:
    con.execute(f"create table {name} (k varchar, v bigint)")
    params = list(zip(rows["k"], rows["v"], strict=True))
    if params:  # `executemany` refuses an empty parameter list; an empty table is the point
        con.executemany(f"insert into {name} values (?, ?)", params)


_SHAPES = {
    "all-matched": {"k": ["c1", "c2"], "v": [10, 20]},
    "some-unmatched": {"k": ["c1", "zz", "c2"], "v": [10, 20, 30]},
    "none-matched": {"k": ["zz", "yy"], "v": [10, 20]},
    "duplicate-probe-keys": {"k": ["c1", "c1", "c1", "c2"], "v": [1, 2, 3, 4]},
    "null-key": {"k": ["c1", None, "c2"], "v": [10, 20, 30]},
    "matched-with-stored-null": {"k": ["c3", "c1"], "v": [10, 20]},
    "empty": {"k": [], "v": []},
    "single-row": {"k": ["c1"], "v": [7]},
}


@pytest.mark.differential
@pytest.mark.parametrize("shape", sorted(_SHAPES))
@pytest.mark.parametrize("how", ["left", "inner"])
def test_lookup_join_matches_duckdb(duck, shape, how):
    rows = _SHAPES[shape]
    _load(duck, "probe", rows)
    got = (
        bt.from_pydict({"k": rows["k"], "v": rows["v"]})
        .lookup_join("memory://", on="k", schema={"name": "string", "tier": "int64"}, how=how)
        .collect()
    )
    join = "left join" if how == "left" else "join"
    assert_same(
        got,
        duck.sql(f"select p.k, p.v, d.name, d.tier from probe p {join} dim d on p.k = d.k"),
    )


@pytest.mark.differential
def test_a_prefixed_lookup_matches_the_aliased_sql(duck):
    rows = _SHAPES["some-unmatched"]
    _load(duck, "probe", rows)
    got = (
        bt.from_pydict({"k": rows["k"], "v": rows["v"]})
        .lookup_join("memory://", on="k", schema={"name": "string", "tier": "int64"}, prefix="dim_")
        .collect()
    )
    assert_same(
        got,
        duck.sql(
            "select p.k, p.v, d.name as dim_name, d.tier as dim_tier "
            "from probe p left join dim d on p.k = d.k"
        ),
    )


@pytest.mark.differential
def test_a_lookup_join_feeding_an_aggregate_matches_duckdb(duck):
    rows = {"k": ["c1", "c1", "c2", "zz"], "v": [10, 20, 30, 40]}
    _load(duck, "probe", rows)
    got = (
        bt.from_pydict(rows)
        .lookup_join("memory://", on="k", schema={"name": "string", "tier": "int64"})
        .group_by("tier")
        .agg(total=bt.col("v").sum(), n=bt.count())
        .collect()
    )
    assert_same(
        got,
        duck.sql(
            "select d.tier, sum(p.v) as total, count(*) as n "
            "from probe p left join dim d on p.k = d.k group by d.tier"
        ),
    )


@pytest.mark.differential
def test_a_filter_after_a_lookup_join_matches_duckdb(duck):
    rows = _SHAPES["some-unmatched"]
    _load(duck, "probe", rows)
    got = (
        bt.from_pydict(rows)
        .lookup_join("memory://", on="k", schema={"name": "string", "tier": "int64"})
        .filter(bt.col("tier") >= 2)
        .collect()
    )
    assert_same(
        got,
        duck.sql(
            "select p.k, p.v, d.name, d.tier from probe p left join dim d on p.k = d.k "
            "where d.tier >= 2"
        ),
    )


@pytest.mark.differential
@pytest.mark.parametrize("batch_size", [1, 2, 1000])
def test_the_result_does_not_depend_on_the_batch_size(duck, batch_size):
    # The lookup is per batch and the cache spans batches, so a bug in either shows up as a
    # result that changes with the batching — which is exactly the kind of bug a single
    # fixed batch size hides.
    rows = {"k": ["c1", "zz", "c1", "c2", "zz"], "v": [1, 2, 3, 4, 5]}
    _load(duck, "probe", rows)
    got = (
        bt.from_pydict(rows)
        .lookup_join(
            "memory://",
            on="k",
            schema={"name": "string", "tier": "int64"},
            batch_size=batch_size,
        )
        .collect()
    )
    assert_same(
        got,
        duck.sql("select p.k, p.v, d.name, d.tier from probe p left join dim d on p.k = d.k"),
    )
