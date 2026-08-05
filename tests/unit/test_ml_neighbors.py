"""k-NN estimators and `KNNImputer`, against scikit-learn.

scikit-learn is the oracle because k-NN has more convention in it than it looks: whether the
distance is weighted, how a tie at the k-th boundary is treated, and what happens when `k`
exceeds the training set are all choices, and matching a widely-used implementation is the
only claim worth making about them.

The one place Batcher deliberately differs is documented and tested here rather than hidden:
ties *at the k-th distance* all count as neighbours, because the alternative is to break the
tie by reference-set order, which makes a prediction depend on the order rows arrived in.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import KNeighborsClassifier, KNeighborsRegressor, KNNImputer
from batcher.ml.preprocessors import Preprocessor

pytestmark = pytest.mark.unit

sklearn_neighbors = pytest.importorskip("sklearn.neighbors")


def _blobs(n: int = 60, seed: int = 0) -> tuple[bt.Dataset, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    target = features[:, 0] * 2.0 - features[:, 1] + rng.normal(scale=0.1, size=n)
    ds = bt.from_pydict(
        {
            "f0": features[:, 0].tolist(),
            "f1": features[:, 1].tolist(),
            "f2": features[:, 2].tolist(),
            "y": target.tolist(),
            "label": (target > 0).astype(int).tolist(),
        }
    )
    return ds, features, target


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_matches_sklearn(k: int, weights: str) -> None:
    ds, features, target = _blobs()
    model = KNeighborsRegressor(["f0", "f1", "f2"], "y", k=k, weights=weights).fit(ds)
    got = model.predict(ds).to_pydict()["prediction"]

    theirs = sklearn_neighbors.KNeighborsRegressor(n_neighbors=k, weights=weights)
    theirs.fit(features, target)
    np.testing.assert_allclose(got, theirs.predict(features), rtol=1e-6)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_classifier_matches_sklearn(k: int) -> None:
    ds, features, target = _blobs()
    labels = (target > 0).astype(int)
    model = KNeighborsClassifier(["f0", "f1", "f2"], "label", k=k).fit(ds)
    got = model.predict(ds).to_pydict()["prediction"]

    theirs = sklearn_neighbors.KNeighborsClassifier(n_neighbors=k)
    theirs.fit(features, labels)
    assert got == theirs.predict(features).tolist()


def test_the_prediction_is_one_expression_not_a_join() -> None:
    """The reference set is folded in as literals, so scoring adds no relational op."""
    ds, _, _ = _blobs(20)
    model = KNeighborsRegressor(["f0"], "y", k=3).fit(ds)
    plan = model.predict(ds).explain()
    assert "join" not in plan.lower()


def test_k_larger_than_the_training_set_uses_everything() -> None:
    ds = bt.from_pydict({"x": [0.0, 1.0, 2.0], "y": [3.0, 6.0, 9.0]})
    model = KNeighborsRegressor(["x"], "y", k=99).fit(ds)
    got = model.predict(bt.from_pydict({"x": [1.0]})).to_pydict()["prediction"]
    assert got == [pytest.approx(6.0)]  # the mean of all three


def test_ties_at_the_boundary_all_count() -> None:
    """Two reference rows equidistant from the query both vote, rather than one winning by order."""
    train = bt.from_pydict({"x": [0.0, 2.0], "y": [10.0, 20.0]})
    model = KNeighborsRegressor(["x"], "y", k=1).fit(train)
    got = model.predict(bt.from_pydict({"x": [1.0]})).to_pydict()["prediction"]
    assert got == [pytest.approx(15.0)]


def test_a_coincident_row_dominates_a_distance_weighted_average() -> None:
    train = bt.from_pydict({"x": [5.0, 100.0], "y": [1.0, 2.0]})
    model = KNeighborsRegressor(["x"], "y", k=2, weights="distance").fit(train)
    got = model.predict(bt.from_pydict({"x": [5.0]})).to_pydict()["prediction"]
    assert got[0] == pytest.approx(1.0, abs=1e-6)


def test_the_classifier_reports_its_classes() -> None:
    train = bt.from_pydict({"x": [0.0, 1.0, 2.0], "label": ["b", "a", "b"]})
    assert KNeighborsClassifier(["x"], "label", k=1).fit(train).classes_ == ["a", "b"]


def test_rows_with_a_null_feature_or_target_are_not_references() -> None:
    train = bt.from_pydict({"x": [0.0, None, 2.0, 3.0], "y": [1.0, 5.0, None, 4.0]})
    model = KNeighborsRegressor(["x"], "y", k=1).fit(train)
    assert len(model.points_) == 2


def test_predict_before_fit_names_the_class() -> None:
    ds, _, _ = _blobs(5)
    for model in (
        KNeighborsRegressor(["f0"], "y"),
        KNeighborsClassifier(["f0"], "label"),
    ):
        with pytest.raises(PlanError, match="must be fitted"):
            model.predict(ds)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: KNeighborsRegressor([], "y"), "at least one feature"),
        (lambda: KNeighborsRegressor(["x"], "y", k=0), "k must be at least 1"),
        (lambda: KNeighborsRegressor(["x"], "y", weights="close"), "weights must be"),
        (lambda: KNNImputer(["a"]), "at least two columns"),
        (lambda: KNNImputer(["a", "b"], k=0), "k must be at least 1"),
        (lambda: KNNImputer(["a", "b"], weights="close"), "weights must be"),
    ],
)
def test_configuration_is_validated(factory, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        factory()


def test_a_reference_set_too_large_to_fold_in_is_refused() -> None:
    ds, _, _ = _blobs(60)
    with pytest.raises(PlanError, match="more than 10 usable rows"):
        KNeighborsRegressor(["f0"], "y", max_reference=10).fit(ds)


def test_a_missing_column_is_named() -> None:
    ds, _, _ = _blobs(5)
    with pytest.raises(ColumnNotFoundError):
        KNeighborsRegressor(["nope"], "y").fit(ds)


def test_an_empty_reference_set_says_so() -> None:
    ds = bt.from_pydict({"x": [None, None], "y": [1.0, 2.0]})
    with pytest.raises(PlanError, match="no rows with every feature present"):
        KNeighborsRegressor(["x"], "y").fit(ds)


@pytest.mark.parametrize("k", [1, 2])
def test_knn_imputer_agrees_with_sklearn_where_the_neighbour_is_unambiguous(k: int) -> None:
    """On data with a clear neighbourhood the two must pick the same donors.

    They can differ on unstructured noise, and that is a stated difference rather than a
    defect: scikit-learn lets a row with its *own* gaps donate, measuring distance over the
    coordinates two rows share and rescaling; Batcher donates only from rows complete across
    the imputed columns. With tied neighbours those pools choose differently, so the
    comparison is made where the answer is not a coin flip.
    """
    from sklearn.impute import KNNImputer as SkKNNImputer

    matrix = np.array(
        [
            [1.0, 1.0, 7.0],
            [1.2, 1.1, 7.4],
            [50.0, 50.0, 500.0],
            [51.0, 50.5, 505.0],
            [1.1, 1.05, np.nan],
        ]
    )
    ds = bt.from_pydict(
        {f"c{i}": [None if np.isnan(v) else float(v) for v in matrix[:, i]] for i in range(3)}
    )
    ours = KNNImputer(["c0", "c1", "c2"], k=k).fit_transform(ds).to_pydict()
    theirs = SkKNNImputer(n_neighbors=k).fit_transform(matrix)
    assert ours["c2"][4] == pytest.approx(theirs[4, 2], rel=1e-9)


def test_the_filled_value_is_always_one_the_neighbours_actually_had() -> None:
    """A weighted mean of donor values cannot fall outside their range."""
    rng = np.random.default_rng(2)
    rows = 40
    matrix = rng.normal(size=(rows, 3))
    holed = matrix.copy()
    holed[5, 1] = np.nan
    ds = bt.from_pydict(
        {f"c{i}": [None if np.isnan(v) else float(v) for v in holed[:, i]] for i in range(3)}
    )
    out = KNNImputer(["c0", "c1", "c2"], k=3).fit_transform(ds).to_pydict()
    donors = [v for v in holed[:, 1] if not np.isnan(v)]
    assert min(donors) <= out["c1"][5] <= max(donors)


def test_present_values_survive_imputation_untouched() -> None:
    rng = np.random.default_rng(2)
    rows = 30
    matrix = rng.normal(size=(rows, 3))
    holed = matrix.copy()
    holed[7, 2] = np.nan
    ds = bt.from_pydict(
        {f"c{i}": [None if np.isnan(v) else float(v) for v in holed[:, i]] for i in range(3)}
    )
    out = KNNImputer(["c0", "c1", "c2"], k=2).fit_transform(ds).to_pydict()
    got = np.array([out[f"c{i}"] for i in range(3)], dtype=float).T
    present = ~np.isnan(holed)
    np.testing.assert_allclose(got[present], holed[present])


def test_the_imputer_leaves_present_values_exactly_alone() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0, None], "b": [1.0, 2.0, 3.0, 4.0]})
    out = KNNImputer(["a", "b"], k=1).fit_transform(ds).to_pydict()
    assert out["a"][:3] == [1.0, 2.0, 3.0]
    assert out["b"] == [1.0, 2.0, 3.0, 4.0]


def test_the_imputer_fills_from_the_nearest_row_not_the_mean() -> None:
    ds = bt.from_pydict(
        {"size": [10.0, 11.0, 50.0, 51.0, 10.5], "price": [1.0, 1.2, 9.0, 9.4, None]}
    )
    out = KNNImputer(["size", "price"], k=2).fit_transform(ds).to_pydict()
    assert out["price"][4] == pytest.approx(1.1, abs=1e-6)  # not the column mean of ~5.15


def test_the_imputer_is_fitted_state_applied_to_a_new_split() -> None:
    train = bt.from_pydict({"a": [1.0, 9.0], "b": [1.0, 9.0]})
    fitted = KNNImputer(["a", "b"], k=1).fit(train)
    out = fitted.transform(bt.from_pydict({"a": [8.5], "b": [None]})).to_pydict()
    assert out["b"] == [pytest.approx(9.0)]


def test_a_fitted_neighbour_model_round_trips_through_save(tmp_path) -> None:
    from batcher.ml.persistence import load_model, save_model

    ds, _, _ = _blobs(20)
    fitted = KNeighborsRegressor(["f0", "f1"], "y", k=3).fit(ds)
    target = str(tmp_path / "knn.json")
    save_model(fitted, target)
    assert load_model(target).predict(ds).to_pydict() == fitted.predict(ds).to_pydict()


def test_the_imputer_round_trips_through_save(tmp_path) -> None:
    ds = bt.from_pydict({"a": [1.0, 9.0, None], "b": [1.0, 9.0, 8.5]})
    fitted = KNNImputer(["a", "b"], k=1).fit(ds)
    target = str(tmp_path / "imputer.json")
    fitted.save(target)
    assert Preprocessor.load(target).transform(ds).to_pydict() == fitted.transform(ds).to_pydict()


def test_the_imputer_composes_in_a_chain() -> None:
    from batcher.ml.preprocessors import Chain, StandardScaler

    ds = bt.from_pydict({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
    out = Chain(KNNImputer(["a", "b"], k=1), StandardScaler(["a"])).fit_transform(ds)
    assert all(v is not None for v in out.to_pydict()["a"])


def _imbalanced(rare: int = 4, common: int = 20) -> bt.Dataset:
    rng = np.random.default_rng(1)
    minority = rng.normal(0.0, 1.0, size=(rare, 2))
    majority = rng.normal(10.0, 1.0, size=(common, 2))
    points = np.vstack([minority, majority])
    return bt.from_pydict(
        {
            "a": points[:, 0].tolist(),
            "b": points[:, 1].tolist(),
            "label": ["rare"] * rare + ["common"] * common,
        }
    )


def test_smote_balances_the_classes_by_default() -> None:
    from batcher.ml import smote

    grown = smote(_imbalanced(), "label", minority="rare", features=["a", "b"])
    counts: dict[str, int] = {}
    for value in grown.to_pydict()["label"]:
        counts[value] = counts.get(value, 0) + 1
    assert counts["rare"] == counts["common"] == 20


def test_smote_makes_new_points_rather_than_copies() -> None:
    """The whole difference from `oversample`: the synthetic rows are not duplicates."""
    from batcher.ml import smote

    ds = _imbalanced()
    original = {
        (round(a, 9), round(b, 9)) for a, b in zip(*(ds.to_pydict()[c] for c in "ab"), strict=True)
    }
    grown = smote(ds, "label", minority="rare", features=["a", "b"]).to_pydict()
    synthetic = [
        (round(a, 9), round(b, 9))
        for a, b, lab in zip(grown["a"], grown["b"], grown["label"], strict=True)
    ][len(original) :]
    assert synthetic
    assert not set(synthetic) & original


def test_a_synthetic_row_lies_inside_the_minority_hull() -> None:
    """Interpolation only, never extrapolation: every new point is between two real ones."""
    from batcher.ml import smote

    ds = _imbalanced()
    frame = ds.to_pydict()
    rare = [
        (a, b)
        for a, b, lab in zip(frame["a"], frame["b"], frame["label"], strict=True)
        if lab == "rare"
    ]
    low_a, high_a = min(p[0] for p in rare), max(p[0] for p in rare)
    low_b, high_b = min(p[1] for p in rare), max(p[1] for p in rare)
    grown = smote(ds, "label", minority="rare", features=["a", "b"]).to_pydict()
    for a, b, lab in list(zip(grown["a"], grown["b"], grown["label"], strict=True))[
        len(frame["a"]) :
    ]:
        assert lab == "rare"
        assert low_a - 1e-9 <= a <= high_a + 1e-9
        assert low_b - 1e-9 <= b <= high_b + 1e-9


def test_smote_is_reproducible_and_partition_independent() -> None:
    from batcher.ml import smote

    ds = _imbalanced()
    first = smote(ds, "label", minority="rare", features=["a", "b"]).sort("a").to_pydict()
    second = smote(ds, "label", minority="rare", features=["a", "b"]).sort("a").to_pydict()
    split = (
        smote(ds.repartition(3), "label", minority="rare", features=["a", "b"])
        .sort("a")
        .to_pydict()
    )
    assert first == second == split


def test_a_different_seed_makes_different_points() -> None:
    from batcher.ml import smote

    ds = _imbalanced()
    one = smote(ds, "label", minority="rare", features=["a", "b"], seed=1).to_pydict()["a"]
    two = smote(ds, "label", minority="rare", features=["a", "b"], seed=2).to_pydict()["a"]
    assert one != two


def test_n_samples_controls_how_many_rows_are_made() -> None:
    from batcher.ml import smote

    ds = _imbalanced()
    grown = smote(ds, "label", minority="rare", features=["a", "b"], n_samples=7)
    assert grown.count() == ds.count() + 7


def test_more_synthetic_rows_than_minority_rows_still_differ() -> None:
    """Each repeat re-seeds, so asking for more than the class size is not duplication."""
    from batcher.ml import smote

    ds = _imbalanced(rare=3)
    grown = smote(ds, "label", minority="rare", features=["a", "b"], n_samples=9).to_pydict()
    made = list(zip(grown["a"], grown["b"], strict=True))[-9:]
    assert len({(round(a, 9), round(b, 9)) for a, b in made}) > 3


def test_columns_smote_cannot_interpolate_are_left_null() -> None:
    """Inventing an id or a free-text value would put fabricated data in an unchecked column."""
    from batcher.ml import smote

    ds = bt.from_pydict(
        {
            "a": [0.0, 1.0, 2.0, 9.0, 9.5],
            "note": ["x", "y", "z", "p", "q"],
            "label": ["rare", "rare", "rare", "common", "common"],
        }
    )
    grown = smote(ds, "label", minority="rare", features=["a"], n_samples=2).to_pydict()
    assert grown["note"][-2:] == [None, None]
    assert all(v is not None for v in grown["a"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"features": []}, "at least one feature column"),
        ({"features": ["a"], "k": 0}, "k must be at least 1"),
    ],
)
def test_smote_configuration_is_validated(kwargs: dict, message: str) -> None:
    from batcher.ml import smote

    with pytest.raises(PlanError, match=message):
        smote(_imbalanced(), "label", minority="rare", **kwargs)


def test_a_minority_class_of_one_says_to_use_oversample() -> None:
    from batcher.ml import smote

    ds = bt.from_pydict({"a": [0.0, 5.0, 6.0], "label": ["rare", "common", "common"]})
    with pytest.raises(PlanError, match="Use oversample"):
        smote(ds, "label", minority="rare", features=["a"])


def test_an_already_balanced_class_is_returned_unchanged() -> None:
    from batcher.ml import smote

    ds = bt.from_pydict({"a": [0.0, 1.0, 5.0, 6.0], "label": ["rare", "rare", "common", "common"]})
    assert smote(ds, "label", minority="rare", features=["a"]).count() == 4
