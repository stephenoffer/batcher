"""Bug-hunt (wave 2) regressions for the categorical encoders (ml/preprocessors/encoders).

The defect: `OrdinalEncoder` / `LabelEncoder` on a column with **no learned categories**
(an all-null column, or an empty fit set) fell back to ``col(column) * 0 + unknown_value``
to build a constant column. Multiplying a null/string column by an integer is invalid in
the engine (``Null * Int64`` / ``Utf8 * Int64``), so the transform raised a `RuntimeError`
instead of producing the documented all-``unknown_value`` column. The fix uses a broadcast
literal.
"""

from __future__ import annotations

import batcher as bt
from batcher.ml.preprocessors import LabelEncoder, OrdinalEncoder


def test_ordinal_encoder_all_null_column_maps_to_unknown() -> None:
    ds = bt.from_pydict({"c": [None, None, None]})
    # Without the fix this raises RuntimeError("Invalid arithmetic operation: Null * Int64").
    out = OrdinalEncoder(["c"]).fit_transform(ds).to_pydict()
    assert out == {"c": [-1, -1, -1]}


def test_ordinal_encoder_all_null_honours_unknown_value() -> None:
    ds = bt.from_pydict({"c": [None, None]})
    out = OrdinalEncoder(["c"], unknown_value=-9).fit_transform(ds).to_pydict()
    assert out == {"c": [-9, -9]}


def test_label_encoder_all_null_column_maps_to_unknown() -> None:
    ds = bt.from_pydict({"y": [None, None]})
    out = LabelEncoder("y").fit_transform(ds).to_pydict()
    assert out == {"y": [-1, -1]}


def test_ordinal_encoder_normal_path_unchanged() -> None:
    # The non-empty-category path must still encode by sorted index, unseen -> unknown.
    ds = bt.from_pydict({"c": ["b", "a", "c", "a", None]})
    out = OrdinalEncoder(["c"]).fit_transform(ds).to_pydict()
    assert out == {"c": [1, 0, 2, 0, -1]}
