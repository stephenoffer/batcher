"""Fitting a preprocessor on data that cannot support it says so, in the preprocessor's terms.

The estimator side of this already exists (`test_estimator_degenerate_fits.py`); these are the
same failures on the `Preprocessor` family, and they arrived the same two ways.

`PCA` and `TruncatedSVD` read an aggregate straight into ``float()``, so an empty or all-null
fit surfaced ``TypeError: float() argument must be a string or a real number, not 'NoneType'``
from inside the fit, naming neither the preprocessor nor the column.

The *expanding* encoders — the ones that emit one column per learned category — failed worse.
`LabelBinarizer`, `MultiLabelBinarizer` and `MultiHotEncoder` reached ``with_columns()`` with no
projections at all and raised "with_columns() requires at least one column", an error from two
layers away. `OneHotEncoder` did not raise: it dropped the source column, appended nothing, and
returned a dataset silently missing a column — so a fit against an empty split quietly deleted
that column from every subsequent `transform`.

Encoders that emit a *fixed* number of columns are deliberately exempt and are pinned here too:
an empty category set is exactly what `OrdinalEncoder` and `LabelEncoder`'s `unknown_value` is
for, so those must keep fitting.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    PCA,
    LabelBinarizer,
    LabelEncoder,
    MultiHotEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
    TruncatedSVD,
)


@pytest.fixture
def empty() -> bt.Dataset:
    """A *typed* empty frame; an untyped one fails for the unrelated reason of null types."""
    return bt.from_arrow(
        pa.table(
            {
                "a": pa.array([], pa.float64()),
                "b": pa.array([], pa.float64()),
                "c": pa.array([], pa.string()),
                "tags": pa.array([], pa.list_(pa.string())),
            }
        )
    )


@pytest.fixture
def usable() -> bt.Dataset:
    return bt.from_pydict(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
            "c": ["x", "y", "x", "y"],
            "tags": [["p", "q"], ["q"], ["p"], ["p", "q"]],
        }
    )


@pytest.mark.parametrize("cls", [PCA, TruncatedSVD])
def test_decomposition_on_an_empty_fit_names_the_column(empty: bt.Dataset, cls) -> None:
    with pytest.raises(PlanError, match=r"'a'.*no non-null values"):
        cls(["a", "b"], n_components=1).fit(empty)


def test_svd_names_both_columns_when_their_non_nulls_never_overlap() -> None:
    """Neither column is empty, but no row has both — so the gram entry has no value."""
    disjoint = bt.from_arrow(
        pa.table(
            {
                "a": pa.array([1.0, None], pa.float64()),
                "b": pa.array([None, 2.0], pa.float64()),
            }
        )
    )
    with pytest.raises(PlanError, match=r"'a'.*'b'.*share no row"):
        TruncatedSVD(["a", "b"], n_components=1).fit(disjoint)


@pytest.mark.parametrize(
    ("build", "column"),
    [
        (lambda: LabelBinarizer("c"), "c"),
        (lambda: MultiLabelBinarizer("tags"), "tags"),
        (lambda: MultiHotEncoder("tags"), "tags"),
        (lambda: OneHotEncoder(["c"]), "c"),
    ],
)
def test_expanding_encoder_on_an_empty_fit_names_itself_and_the_column(
    empty: bt.Dataset, build, column: str
) -> None:
    with pytest.raises(PlanError, match=rf"{column!r}.*no categories to expand"):
        build().fit(empty)


def test_one_hot_does_not_silently_drop_the_column_it_could_not_encode(empty: bt.Dataset) -> None:
    """The regression proper: this used to return a dataset with 'c' deleted and nothing added."""
    with pytest.raises(PlanError):
        OneHotEncoder(["c"]).fit_transform(empty).to_pydict()


@pytest.mark.parametrize("build", [lambda: OrdinalEncoder(["c"]), lambda: LabelEncoder("c")])
def test_fixed_width_encoders_still_fit_an_empty_category_set(empty: bt.Dataset, build) -> None:
    """These map an unseen value to `unknown_value`, so an empty fit is well defined."""
    out = build().fit_transform(empty).to_pydict()
    assert out["c"] == []


def test_usable_data_still_fits_everywhere(usable: bt.Dataset) -> None:
    assert len(PCA(["a", "b"], n_components=1).fit(usable).components_) == 1
    assert len(TruncatedSVD(["a", "b"], n_components=1).fit(usable).components_) == 1
    assert LabelBinarizer("c").fit(usable).classes_ == ["x", "y"]
    assert MultiLabelBinarizer("tags").fit(usable).labels_ == ["p", "q"]
    assert MultiHotEncoder("tags").fit(usable).categories_ == ["p", "q"]
    assert OneHotEncoder(["c"]).fit(usable).categories_["c"] == ["x", "y"]
