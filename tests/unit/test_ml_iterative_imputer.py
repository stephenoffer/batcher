"""`IterativeImputer` — modelling each incomplete column from the others.

The property worth testing is that the imputation *uses the other columns*: a fill that
ignored them would equal `SimpleImputer`'s, and every shape and null-count assertion would
still pass. So the central tests compare against the exact value the relationship implies,
and against what a mean fill would have produced.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import IterativeImputer, Preprocessor, SimpleImputer

pytestmark = pytest.mark.unit


def _linear(missing_at: int = 2) -> bt.Dataset:
    """``b = 2a`` exactly, with one b missing — so the right answer is known."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [2.0 * v for v in a]
    b[missing_at] = None
    return bt.from_pydict({"a": a, "b": b})


def test_the_imputed_value_follows_the_relationship_not_the_mean() -> None:
    out = IterativeImputer(["a", "b"], max_iter=5).fit_transform(_linear()).to_pydict()
    assert out["b"][2] == pytest.approx(6.0, abs=1e-4)


def test_it_beats_a_mean_fill_when_the_columns_are_related() -> None:
    ds = _linear()
    iterative = IterativeImputer(["a", "b"], max_iter=5).fit_transform(ds).to_pydict()["b"][2]
    simple = SimpleImputer(["b"]).fit_transform(ds).to_pydict()["b"][2]
    assert abs(iterative - 6.0) < abs(simple - 6.0)


def test_observed_values_are_never_overwritten() -> None:
    """Only the entries that were null may move; everything else must survive untouched."""
    ds = _linear()
    before = ds.to_pydict()
    after = IterativeImputer(["a", "b"], max_iter=5).fit_transform(ds).to_pydict()
    for index, value in enumerate(before["b"]):
        if value is not None:
            assert after["b"][index] == pytest.approx(value)
    assert after["a"] == before["a"]


def test_the_output_has_no_nulls_and_no_helper_columns() -> None:
    out = IterativeImputer(["a", "b"], max_iter=3).fit_transform(_linear())
    assert out.columns == ["a", "b"]
    assert all(v is not None for v in out.to_pydict()["b"])


def test_several_incomplete_columns_are_all_imputed() -> None:
    ds = bt.from_pydict(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [2.0, None, 6.0, 8.0, 10.0],
            "c": [3.0, 6.0, 9.0, None, 15.0],
        }
    )
    out = IterativeImputer(["a", "b", "c"], max_iter=5).fit_transform(ds).to_pydict()
    assert out["a"][2] == pytest.approx(3.0, abs=1e-3)
    assert out["b"][1] == pytest.approx(4.0, abs=1e-3)
    assert out["c"][3] == pytest.approx(12.0, abs=1e-3)


def test_the_schedule_is_fitted_state_applied_to_a_new_split() -> None:
    train = _linear()
    fitted = IterativeImputer(["a", "b"], max_iter=5).fit(train)
    out = fitted.transform(bt.from_pydict({"a": [10.0], "b": [None]})).to_pydict()
    assert out["b"][0] == pytest.approx(20.0, abs=1e-3)


def test_a_frame_with_nothing_missing_fits_and_changes_nothing() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0], "b": [2.0, 4.0]})
    fitted = IterativeImputer(["a", "b"]).fit(ds)
    assert fitted.imputations_ == []
    assert fitted.transform(ds).to_pydict() == ds.to_pydict()


def test_convergence_stops_early_and_is_recorded() -> None:
    """An exact linear relationship converges in one round, so the rest is wasted work."""
    fitted = IterativeImputer(["a", "b"], max_iter=20, tol=1e-6).fit(_linear())
    assert fitted.n_iter_ < 20
    assert len(fitted.imputations_) == fitted.n_iter_


@pytest.mark.parametrize("max_iter", [1, 2, 3])
def test_max_iter_bounds_the_schedule(max_iter: int) -> None:
    """`tol` may stop it sooner, but nothing may run more rounds than asked for."""
    fitted = IterativeImputer(["a", "b"], max_iter=max_iter, tol=0.0).fit(_linear())
    assert 1 <= fitted.n_iter_ <= max_iter
    assert len(fitted.imputations_) == fitted.n_iter_


def test_a_tolerance_of_zero_still_stops_once_nothing_moves() -> None:
    """Exact convergence satisfies any non-negative tolerance, including zero."""
    fitted = IterativeImputer(["a", "b"], max_iter=50, tol=0.0).fit(_linear())
    assert fitted.n_iter_ < 50


def test_median_initialization_is_accepted() -> None:
    fitted = IterativeImputer(["a", "b"], initial_strategy="median", max_iter=3).fit(_linear())
    assert set(fitted.initial_) == {"a", "b"}


def test_fit_is_independent_of_partitioning() -> None:
    one = IterativeImputer(["a", "b"], max_iter=3).fit(_linear())
    many = IterativeImputer(["a", "b"], max_iter=3).fit(_linear().repartition(3))
    assert one.initial_ == pytest.approx(many.initial_)
    assert len(one.imputations_) == len(many.imputations_)


def test_transform_before_fit_names_the_class() -> None:
    with pytest.raises(PlanError, match="must be fitted"):
        IterativeImputer(["a", "b"]).transform(_linear())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"columns": ["a"]}, "at least two columns"),
        ({"columns": ["a", "b"], "max_iter": 0}, "max_iter must be at least 1"),
        ({"columns": ["a", "b"], "initial_strategy": "mode"}, "initial_strategy must be"),
        ({"columns": ["a", "b"], "tol": -1.0}, "tol must be non-negative"),
    ],
)
def test_configuration_is_validated(kwargs: dict, message: str) -> None:
    columns = kwargs.pop("columns")
    with pytest.raises(PlanError, match=message):
        IterativeImputer(columns, **kwargs)


def test_an_all_null_column_says_so() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0], "b": [None, None]})
    with pytest.raises(PlanError, match="no non-null values"):
        IterativeImputer(["a", "b"]).fit(ds)


def test_a_fitted_imputer_round_trips_through_save(tmp_path) -> None:
    ds = _linear()
    fitted = IterativeImputer(["a", "b"], max_iter=4).fit(ds)
    target = str(tmp_path / "imputer.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    assert restored.transform(ds).to_pydict() == fitted.transform(ds).to_pydict()


def test_it_composes_in_a_chain() -> None:
    from batcher.ml.preprocessors import Chain, StandardScaler

    out = Chain(IterativeImputer(["a", "b"], max_iter=3), StandardScaler(["a", "b"])).fit_transform(
        _linear()
    )
    assert out.count() == 6
    assert all(v is not None for v in out.to_pydict()["b"])
