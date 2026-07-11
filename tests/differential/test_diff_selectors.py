"""Differential + equivalence tests for column selectors.

Selectors are projection sugar: they expand at plan time to an ordinary column list,
so the contract is that a selector-based projection equals the explicit one it stands
for — and, where a selector maps onto SQL, that the result matches DuckDB.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def wide():
    return bt.from_pydict(
        {
            "id": [1, 2, 3],
            "price": [1.5, 2.5, 3.5],
            "qty": [10, 20, 30],
            "name": ["a", "b", "c"],
            "flag": [True, False, True],
            "day": [datetime.date(2024, 1, i + 1) for i in range(3)],
        }
    )


@pytest.mark.differential
def test_dtype_selectors_match_explicit_lists(wide):
    """Each dtype selector picks exactly the columns of that kind, in input order."""
    assert wide.select(bt.numeric()).columns == ["id", "price", "qty"]
    assert wide.select(bt.integer()).columns == ["id", "qty"]
    assert wide.select(bt.floating()).columns == ["price"]
    assert wide.select(bt.string()).columns == ["name"]
    assert wide.select(bt.boolean()).columns == ["flag"]
    assert wide.select(bt.temporal()).columns == ["day"]
    assert wide.select(bt.all()).columns == wide.columns


@pytest.mark.differential
def test_name_selectors(wide):
    """starts_with / ends_with / contains / by_dtype match by name and exact type."""
    assert wide.select(bt.starts_with("p")).columns == ["price"]
    assert wide.select(bt.ends_with("e")).columns == ["price", "name"]
    assert wide.select(bt.contains("i")).columns == ["id", "price"]
    assert wide.select(bt.starts_with("i", "q")).columns == ["id", "qty"]
    assert wide.select(bt.by_dtype(pa.int64())).columns == ["id", "qty"]
    assert wide.select(bt.by_dtype(pa.float64(), pa.large_string())).columns == ["price"]


@pytest.mark.differential
def test_set_algebra_and_exclude(wide):
    """Union / intersection / difference / complement compose the matched sets."""
    assert wide.select(bt.numeric() - bt.floating()).columns == ["id", "qty"]
    assert wide.select(bt.integer() | bt.string()).columns == ["id", "qty", "name"]
    assert wide.select(bt.numeric() & bt.matches("id")).columns == ["id"]
    assert wide.select(~bt.numeric()).columns == ["name", "flag", "day"]
    assert wide.select(bt.exclude("id", "day")).columns == ["price", "qty", "name", "flag"]


@pytest.mark.differential
def test_computed_selector_equals_explicit(duck, wide):
    """A scalar op over a selector equals the same op written out column by column."""
    got = wide.with_columns(bt.numeric() * 2).to_pydict()
    explicit = wide.with_columns(
        id=bt.col("id") * 2, price=bt.col("price") * 2, qty=bt.col("qty") * 2
    ).to_pydict()
    assert got == explicit

    # And it matches DuckDB computing the same doubled columns.
    duck.register("wide", wide.to_arrow())
    duck_out = duck.sql(
        "SELECT id * 2 AS id, price * 2 AS price, qty * 2 AS qty, name, flag, day FROM wide"
    ).to_arrow_table()
    assert (
        pa.table(got).select(["id", "price", "qty"]).equals(duck_out.select(["id", "price", "qty"]))
    )


@pytest.mark.differential
def test_name_accessor_renames_expanded_columns(wide):
    """The .name accessor derives each output name from its matched input name."""
    assert wide.select(bt.numeric().name.prefix("n_")).columns == ["n_id", "n_price", "n_qty"]
    assert wide.select(bt.numeric().name.suffix("_x")).columns == ["id_x", "price_x", "qty_x"]
    assert wide.select(bt.string().name.to_uppercase()).columns == ["NAME"]
    renamed = wide.select(bt.all().name.map(lambda c: c[:2]))
    assert renamed.columns == ["id", "pr", "qt", "na", "fl", "da"]


@pytest.mark.differential
def test_drop_accepts_selectors(wide):
    """drop() removes the columns a selector matches, keeping the rest in order."""
    assert wide.drop(bt.temporal(), bt.boolean()).columns == ["id", "price", "qty", "name"]
    assert wide.drop(bt.numeric()).columns == ["name", "flag", "day"]


@pytest.mark.differential
def test_selector_misuse_raises_plan_error(wide):
    """A selector outside a projection, or an over-broad alias, is a typed error."""
    with pytest.raises(PlanError, match="cannot be used here"):
        wide.filter(bt.numeric() > 0).collect()
    with pytest.raises(PlanError, match="at most one column selector"):
        wide.select(bt.numeric() + bt.integer()).collect()
    with pytest.raises(PlanError, match="names a single column"):
        wide.select(bt.numeric().alias("z")).collect()
    with pytest.raises(PlanError, match="matched no columns"):
        wide.select(bt.matches("^nope$")).collect()
