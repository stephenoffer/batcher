"""Python-friendly dunder/builtin behavior on the public objects.

`repr(col("x") + 1)` reads like the code that built it; `len(ds)`, `col in ds`,
`ds[cols]`, a dict-like `Session`, a truthy `ValidationReport`, and a
context-manager `StreamingQuery` all behave the way a Python dev expects.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (bt.col("x"), "col('x')"),
        (bt.lit(5), "lit(5)"),
        (bt.col("x") + 1, "(col('x') + lit(1))"),
        (bt.col("x") * bt.col("y"), "(col('x') * col('y'))"),
        (bt.col("x").sum(), "col('x').sum()"),
        (bt.count(), "count_star()"),
        (bt.col("x").cast("float64"), "col('x').cast('float64')"),
        (bt.col("x").is_null(), "col('x').is_null()"),
        (~bt.col("b"), "~col('b')"),
        (bt.col("s").str.upper(), "col('s').str.upper()"),
        (bt.col("d").dt.year(), "col('d').dt.year()"),
        (bt.col("x").alias("total"), "col('x').alias('total')"),
        (bt.coalesce(bt.col("a"), bt.lit(0)), "coalesce(col('a'), lit(0))"),
    ],
)
def test_expr_repr_reads_like_source(expr, expected):
    assert repr(expr) == expected


def test_expr_repr_never_leaks_object_address():
    # A deeply-composed expression still renders cleanly, no ``<... at 0x...>``.
    e = bt.when(bt.col("x") > 0).then(bt.col("y").sum()).otherwise(bt.lit(0))
    assert "0x" not in repr(e)
    assert "object at" not in repr(e)


def test_dataset_builtins():
    ds = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(ds) == 3
    assert "a" in ds and "z" not in ds
    assert isinstance(ds["a"], type(bt.col("a")))  # a column Expr
    assert ds[["a"]].columns == ["a"]
    assert sum(batch.num_rows for batch in ds) == 3  # iterates batches
    assert "Dataset(columns=" in repr(ds)


def test_dataset_bool_is_ambiguous():
    from batcher._internal.errors import PlanError

    # Like pandas: bool(frame) is ambiguous — use has_rows / is_empty.
    with pytest.raises(PlanError, match="ambiguous"):
        bool(bt.from_pydict({"x": [1]}))


def test_groupby_repr():
    ds = bt.from_pydict({"g": ["a"], "h": ["b"], "x": [1]})
    assert repr(ds.group_by("g")) == "GroupBy(keys=['g'])"
    assert repr(ds.group_by("g", "h")) == "GroupBy(keys=['g', 'h'])"


def test_session_is_dict_like():
    ds = bt.from_pydict({"x": [1]})
    s = bt.Session()
    s.register("emp", ds)
    s.register("dept", ds)
    assert len(s) == 2
    assert "emp" in s and "missing" not in s
    assert s["emp"].columns == ["x"]
    assert repr(s) == "Session(tables=['emp', 'dept'])"


def test_validation_report_is_truthy_when_clean():
    clean = bt.from_pydict({"x": [1, 2, 3]}).dq.in_range("x", 0, 10).validate()
    dirty = bt.from_pydict({"x": [1, 2, -3]}).dq.in_range("x", 0, 10).validate()
    assert bool(clean) is True
    assert bool(dirty) is False
    assert ("ok" if clean else "bad") == "ok"
