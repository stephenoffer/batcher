"""`available_schema()` — pre-execution type inference must match the engine.

The plan's type-carrying `available_schema()` answers `Dataset.schema` without
scanning rows. Its one hard requirement is faithfulness: an inferred output type
MUST equal what the engine actually produces. The oracle here is a *non-empty*
execution (`collect()` over real rows) — note that a zero-row execution degenerates
derived columns to ``null`` type, which is exactly the degenerate path this
inference replaces, so it cannot be the oracle.

For every case the inference claims to know (returns a schema), every field type
must match the executed result; cases it declines (``None``) are allowed and simply
fall back. Types are compared by name, ignoring nullability/metadata.
"""

from __future__ import annotations

import pytest

import batcher as bt

_BASE = {
    "a": [1, 2, 3, 4],
    "b": [10, 20, 30, 40],
    "f": [1.5, 2.5, 3.5, 4.5],
    "s": ["foo", "bar", "baz", "qux"],
    "g": ["x", "x", "y", "y"],
}


def _ds():
    return bt.from_pydict(_BASE)


# Each case maps a base Dataset to a derived one whose output types we can check.
_CASES = {
    "select_arith_add": lambda d: d.select((bt.col("a") + bt.col("f")).alias("c")),
    "select_arith_sub_int": lambda d: d.select((bt.col("a") - bt.col("b")).alias("c")),
    "select_arith_mul": lambda d: d.select((bt.col("a") * bt.col("b")).alias("c")),
    "select_compare": lambda d: d.select((bt.col("a") > bt.col("b")).alias("c")),
    "select_and": lambda d: d.select(((bt.col("a") > 1) & (bt.col("b") < 30)).alias("c")),
    "select_cast_float": lambda d: d.select(bt.col("a").cast("float64").alias("c")),
    "select_cast_float32": lambda d: d.select(bt.col("a").cast("float32").alias("c")),
    "select_cast_string": lambda d: d.select(bt.col("a").cast("string").alias("c")),
    "select_str_len": lambda d: d.select(bt.col("s").str.len().alias("c")),
    "select_str_upper": lambda d: d.select(bt.col("s").str.upper().alias("c")),
    "select_str_contains": lambda d: d.select(bt.col("s").str.contains("a").alias("c")),
    "select_math_sqrt": lambda d: d.select(bt.col("f").sqrt().alias("c")),
    "select_math_abs": lambda d: d.select(bt.col("a").abs().alias("c")),
    "select_passthrough": lambda d: d.select("a", "f", "s"),
    "filter": lambda d: d.filter(bt.col("a") > 1),
    "with_columns": lambda d: d.with_columns(c=bt.col("a") + bt.col("b")),
    "with_row_index": lambda d: d.with_row_index("idx"),
    "distinct": lambda d: d.select("g").distinct(),
    "sort": lambda d: d.sort("a"),
    "limit": lambda d: d.limit(2),
    "agg_sum_mean_count": lambda d: d.group_by("g").agg(
        total=bt.col("a").sum(), avg=bt.col("f").mean(), n=bt.count()
    ),
    "agg_min_max": lambda d: d.group_by("g").agg(lo=bt.col("a").min(), hi=bt.col("f").max()),
    "union": lambda d: d.select("a", "f").union(d.select("a", "f")),
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_CASES))
def test_available_schema_matches_execution(name):
    plan_ds = _CASES[name](_ds())
    inferred = plan_ds._plan.available_schema()
    actual = plan_ds.collect().schema
    if inferred is None:
        pytest.skip(f"{name}: inference declined (fallback path) — allowed")
    inferred = inferred.arrow
    inferred_types = dict(zip(inferred.names, inferred.types, strict=True))
    actual_types = dict(zip(actual.names, actual.types, strict=True))
    assert set(inferred_types) == set(actual_types), f"{name}: column-name mismatch"
    for n, t in inferred_types.items():
        assert t.equals(actual_types[n]), (
            f"{name}: column {n!r} inferred {t} but engine produced {actual_types[n]}"
        )


@pytest.mark.unit
def test_common_cases_take_the_fast_path():
    # A guard that the inference actually fires for the bread-and-butter shapes
    # (not silently always-None, which would pass the oracle vacuously).
    fired = [name for name in _CASES if _CASES[name](_ds())._plan.available_schema() is not None]
    assert "select_arith_add" in fired
    assert "agg_sum_mean_count" in fired
    assert "with_columns" in fired


# The list-accessor numeric reductions, with the exact Arrow type the engine returns.
# These MUST be inferred (not declined): an uninferable projection routes `Dataset.schema`
# through the zero-row execution fallback, which collapses *every* output column — the
# list column and its plain passthrough neighbours included — to `null`. So a single
# uninferred `list.sum` makes `Dataset.schema` lie about the whole relation, not just the
# reduced column.
_LIST_REDUCTIONS = {
    "sum": "double",
    "mean": "double",
    "median": "double",
    "product": "double",
    "std": "double",
    "var": "double",
    "l2_norm": "double",
    "min": "int64",  # preserves the (int) element type
    "max": "int64",
}


@pytest.mark.unit
@pytest.mark.parametrize("fn", sorted(_LIST_REDUCTIONS))
def test_list_reduction_schema_is_inferred_not_null_collapsed(fn):
    ds = bt.from_pydict({"arr": [[1, 2, 3], [4, 5, 6]], "k": [7, 8]})
    plan_ds = ds.with_columns(r=getattr(bt.col("arr").list, fn)())

    inferred = plan_ds._plan.available_schema()
    assert inferred is not None, (
        f"list.{fn} inference declined → Dataset.schema falls back to the zero-row "
        "execution, which collapses the whole schema to null"
    )
    inferred_types = {f.name: str(f.type) for f in inferred.arrow}
    # The reduced column has the engine's real type ...
    assert inferred_types["r"] == _LIST_REDUCTIONS[fn]
    # ... and the passthrough neighbours are NOT null-collapsed.
    assert inferred_types["arr"] == "list<item: int64>"
    assert inferred_types["k"] == "int64"

    # And the inference is faithful to a real (non-empty) execution.
    actual = {f.name: str(f.type) for f in plan_ds.collect().schema}
    assert inferred_types == actual, f"list.{fn}: inferred {inferred_types} vs engine {actual}"
