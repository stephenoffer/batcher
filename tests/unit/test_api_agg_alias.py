"""A positional aggregate can name its own output with ``.alias(...)``.

``col("x").sum().alias("total")`` is how Polars and PySpark name an aggregate, and it
is the only *positional* spelling that can name a `count()` — which has no input
column to be named after — or two aggregates over one column. Before it, both cases
forced the keyword form and a ported script raised ``AttributeError: 'AggExpr' object
has no attribute 'alias'``.

The alias is consumed at the API surface: `group_by().agg()` reads it to key the
aggregate map, and it never reaches `to_ir`, where `AggregateSpec.alias` carries the
name instead. These pin both halves of that, so a Kyber rule rebuilding an `AggExpr`
(which drops the field) stays correct.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds():
    return bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})


@pytest.mark.unit
def test_alias_names_a_positional_aggregate(ds):
    got = ds.group_by("g").agg(bt.col("x").sum().alias("total")).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "total": [3, 3]}


@pytest.mark.unit
def test_alias_is_the_only_way_to_name_a_positional_count(ds):
    got = ds.group_by("g").agg(bt.count().alias("n")).sort("g").to_pydict()
    assert got == {"g": ["a", "b"], "n": [2, 1]}


@pytest.mark.unit
def test_alias_allows_two_aggregates_over_one_column(ds):
    got = (
        ds.group_by("g")
        .agg(bt.col("x").sum().alias("s"), bt.col("x").mean().alias("m"))
        .sort("g")
        .to_pydict()
    )
    assert got == {"g": ["a", "b"], "s": [3, 3], "m": [1.5, 3.0]}


@pytest.mark.unit
def test_alias_works_on_the_ungrouped_agg(ds):
    assert ds.agg(bt.col("x").sum().alias("t")).to_pydict() == {"t": [6]}


@pytest.mark.unit
def test_alias_returns_a_new_aggregate_and_leaves_the_original_unnamed():
    base = bt.col("x").sum()
    named = base.alias("t")
    assert base.name is None
    assert named.name == "t"
    assert named is not base


@pytest.mark.unit
def test_alias_preserves_the_parametric_and_binary_fields():
    # `.alias` must copy `param`/`input2`, or a quantile silently becomes a median.
    q = bt.col("x").quantile(0.9).alias("p90")
    assert q.func == bt.col("x").quantile(0.9).func
    assert q.param == 0.9
    assert q.name == "p90"


@pytest.mark.unit
def test_alias_does_not_reach_the_wire_contract():
    # The name is carried by `AggregateSpec.alias`; `to_ir` still takes it as an
    # argument, so a Kyber rule that rebuilds an `AggExpr` may drop `.name` safely.
    assert bt.col("x").sum().alias("t").to_ir("bound") == {
        "func": "sum",
        "alias": "bound",
        "input": {"e": "col", "name": "x"},
    }


@pytest.mark.unit
def test_two_aliases_naming_the_same_output_are_rejected(ds):
    with pytest.raises(PlanError, match="aliased 's'"):
        ds.group_by("g").agg(bt.col("x").sum().alias("s"), bt.col("x").mean().alias("s"))


@pytest.mark.unit
def test_the_bare_positional_collision_keeps_its_own_message(ds):
    # Two *unaliased* aggregates over one column are a different mistake, and the
    # message names the column rather than an alias the caller never wrote.
    with pytest.raises(PlanError, match="positional aggregates over column"):
        ds.group_by("g").agg(bt.col("x").sum(), bt.col("x").mean())


@pytest.mark.unit
def test_alias_shows_in_the_repr():
    assert repr(bt.col("x").sum().alias("t")) == "col('x').sum().alias('t')"


@pytest.mark.unit
def test_over_is_unaffected_by_the_new_field(ds):
    got = ds.with_columns(t=bt.col("x").sum().over(partition_by=["g"])).sort("x").to_pydict()
    assert got["t"] == [3, 3, 3]
