"""Unit coverage for the Python builtin/dunder protocols on Dataset and Expr.

These are control-plane behaviors (no engine execution needed for most): the
`Dataset` container protocols (`__iter__`/`__contains__`/set operators) and the
`Expr` guards (`__bool__`) and indexing/desugaring shapes.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def test_dataset_contains_is_column_membership():
    ds = bt.from_pydict({"x": [1, 2], "y": [3, 4]})
    assert "x" in ds
    assert "z" not in ds
    assert 1 not in ds  # non-str keys are never columns


def test_dataset_iter_yields_arrow_batches():
    import pyarrow as pa

    ds = bt.from_pydict({"x": [1, 2, 3]})
    batches = list(ds)
    assert batches and all(isinstance(b, pa.RecordBatch) for b in batches)
    assert [v for b in batches for v in b.column("x").to_pylist()] == [1, 2, 3]


def test_dataset_len_counts_rows():
    assert len(bt.from_pydict({"x": [1, 2, 3, 4]})) == 4


def test_dataset_set_operators():
    d1 = bt.from_pydict({"x": [1, 2]})
    d2 = bt.from_pydict({"x": [2, 3]})
    assert sorted((d1 + d2).collect().to_pydict()["x"]) == [1, 2, 2, 3]  # UNION ALL
    assert sorted((d1 | d2).collect().to_pydict()["x"]) == [1, 2, 3]  # UNION
    assert sorted((d1 & d2).collect().to_pydict()["x"]) == [2]  # INTERSECT
    assert sorted((d1 - d2).collect().to_pydict()["x"]) == [1]  # EXCEPT


def test_dataset_set_operator_with_nondataset_is_typeerror():
    d1 = bt.from_pydict({"x": [1, 2]})
    with pytest.raises(TypeError):
        _ = d1 + 5


def test_expr_bool_is_guarded():
    # Using an expression in a boolean context is a silent-bug footgun; guard it.
    with pytest.raises(PlanError):
        bool(col("x") > 0)
    with pytest.raises(PlanError):
        _ = col("x") in [1, 2, 3]  # `in` calls __eq__ then bool()


def test_expr_getitem_dispatch():
    assert col("a")[2].to_ir()["e"] == "list_get"
    assert col("a")[1:3].to_ir()["e"] == "list_slice"
    assert col("s")["field"].to_ir()["e"] == "struct_field"
    with pytest.raises(PlanError):
        _ = col("a")[1:10:2]  # step != 1
    with pytest.raises(PlanError):
        _ = col("a")[True]  # bool is not a valid index
