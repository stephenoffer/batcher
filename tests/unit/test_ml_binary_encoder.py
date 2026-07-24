"""Binary categorical encoding.

Binary encoding has no external oracle, so it is pinned to its structural contract: each
category maps to a distinct base-2 code, the bits reconstruct that integer code, the column count
is logarithmic in the cardinality, and an unseen category encodes as all-zero bits distinct from
every learned one.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import BinaryEncoder

pytestmark = pytest.mark.unit


def _codes(out: dict, categories: list, values: list, n_bits: int) -> dict:
    codes: dict = {}
    for i, value in enumerate(values):
        code = sum(out[f"c_{b}"][i] << b for b in range(n_bits))
        codes.setdefault(value, set()).add(code)
    return codes


def test_each_category_gets_a_distinct_code() -> None:
    categories = [f"cat{i}" for i in range(20)]
    values = categories * 5
    ds = bt.from_pydict({"c": values})
    pre = BinaryEncoder("c", drop_original=False).fit(ds)
    out = pre.transform(ds).to_pydict()
    codes = _codes(out, categories, values, pre.n_bits_)
    assert all(len(v) == 1 for v in codes.values())  # one code per category
    assert len({next(iter(v)) for v in codes.values()}) == len(categories)  # all distinct


def test_column_count_is_logarithmic() -> None:
    categories = [str(i) for i in range(100)]
    ds = bt.from_pydict({"c": categories})
    pre = BinaryEncoder("c").fit(ds)
    assert pre.n_bits_ == 7  # 100 categories -> codes 1..100 -> 7 bits


def test_output_columns_and_dropping() -> None:
    ds = bt.from_pydict({"c": ["a", "b"]})
    dropped = BinaryEncoder("c").fit_transform(ds)
    assert all(name.startswith("c_") for name in dropped.columns)
    kept = BinaryEncoder("c", drop_original=False).fit_transform(ds)
    assert "c" in kept.columns


def test_unseen_category_encodes_as_all_zero() -> None:
    train = bt.from_pydict({"c": ["a", "b", "c"]})
    pre = BinaryEncoder("c", drop_original=False).fit(train)
    serve = pre.transform(bt.from_pydict({"c": ["zzz"]})).to_pydict()
    assert all(serve[f"c_{b}"][0] == 0 for b in range(pre.n_bits_))


def test_rejects_multiple_columns() -> None:
    with pytest.raises(PlanError, match="one column"):
        BinaryEncoder(["a", "b"])
