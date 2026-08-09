"""Hyperparameter search: the grid, the draw, and the direction of "better".

The failure this file is mostly guarding is `greater_is_better`. A search handed a loss and
left on its default returns the *worst* combination, confidently, with no error — so the
direction is pinned from both sides here rather than assumed.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import Ridge
from batcher.ml.metrics import evaluate
from batcher.ml.model_selection import (
    SearchResult,
    grid_search,
    param_grid,
    param_samples,
    random_search,
)

pytestmark = pytest.mark.unit


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]})


def _fit(train: bt.Dataset, params: dict) -> Ridge:
    return Ridge(["x"], "y", alpha=params["alpha"]).fit(train)


def _predict(model: Ridge, ds: bt.Dataset) -> bt.Dataset:
    return model.predict(ds)


def _r2(ds: bt.Dataset, y_true: str, y_pred: str) -> float:
    return evaluate(ds, y_true, y_pred=y_pred, task="regression", metrics=["r2"])["r2"]


def _rmse(ds: bt.Dataset, y_true: str, y_pred: str) -> float:
    return evaluate(ds, y_true, y_pred=y_pred, task="regression", metrics=["rmse"])["rmse"]


def test_param_grid_is_the_cartesian_product_in_a_stable_order() -> None:
    assert param_grid(a=[1, 2], b=["x", "y"]) == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]


def test_param_grid_with_no_parameters_is_one_empty_combination() -> None:
    """An empty product is one combination, so a search with nothing to vary still runs once."""
    assert param_grid() == [{}]


def test_param_grid_rejects_a_parameter_with_no_values() -> None:
    with pytest.raises(PlanError, match="no candidate values"):
        param_grid(a=[1], b=[])


def test_param_samples_is_reproducible_and_the_right_size() -> None:
    first = param_samples(5, seed=7, alpha=[0.1, 1.0, 10.0])
    second = param_samples(5, seed=7, alpha=[0.1, 1.0, 10.0])
    assert first == second
    assert len(first) == 5
    assert {d["alpha"] for d in first} <= {0.1, 1.0, 10.0}


def test_param_samples_accepts_a_callable_for_a_continuous_range() -> None:
    draws = param_samples(20, seed=1, depth=lambda rng: rng.randint(1, 4))
    assert all(1 <= d["depth"] <= 4 for d in draws)
    assert len({d["depth"] for d in draws}) > 1


def test_a_different_seed_draws_differently() -> None:
    assert param_samples(10, seed=0, a=list(range(50))) != param_samples(
        10, seed=1, a=list(range(50))
    )


@pytest.mark.parametrize(("n", "message"), [(0, "n must be at least 1"), (-3, "n must be")])
def test_param_samples_rejects_a_non_positive_count(n: int, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        param_samples(n, a=[1])


def test_grid_search_picks_the_best_scoring_combination() -> None:
    found = grid_search(
        _ds(),
        _fit,
        _predict,
        y_true="y",
        metric=_r2,
        grid=param_grid(alpha=[0.01, 100.0]),
        k=4,
        key="x",
    )
    assert found.best_params == {"alpha": 0.01}
    assert found.best_score > 0.99
    assert len(found.trials) == 2


def test_greater_is_better_false_minimizes_a_loss() -> None:
    """The same search on RMSE must pick the same model, not the opposite one."""
    found = grid_search(
        _ds(),
        _fit,
        _predict,
        y_true="y",
        metric=_rmse,
        grid=param_grid(alpha=[0.01, 100.0]),
        greater_is_better=False,
        k=4,
        key="x",
    )
    assert found.best_params == {"alpha": 0.01}
    assert found.best_score == min(t["mean"] for t in found.trials)


def test_trials_are_ordered_best_first() -> None:
    found = grid_search(
        _ds(),
        _fit,
        _predict,
        y_true="y",
        metric=_r2,
        grid=param_grid(alpha=[100.0, 0.01, 10.0]),
        k=4,
        key="x",
    )
    means = [t["mean"] for t in found.trials]
    assert means == sorted(means, reverse=True)
    assert found.trials[0]["params"] == found.best_params


def test_every_trial_reports_its_spread_and_per_fold_scores() -> None:
    found = grid_search(
        _ds(), _fit, _predict, y_true="y", metric=_r2, grid=param_grid(alpha=[1.0]), k=4, key="x"
    )
    trial = found.trials[0]
    assert len(trial["scores"]) == 4
    assert trial["mean"] == pytest.approx(sum(trial["scores"]) / 4)
    assert trial["std"] >= 0.0


def test_every_combination_is_scored_on_the_same_folds() -> None:
    """The comparison must be paired, or fold luck shows up as a parameter difference."""
    seen: list[int] = []

    def counting_fit(train: bt.Dataset, params: dict) -> Ridge:
        seen.append(train.count())
        return _fit(train, params)

    grid_search(
        _ds(),
        counting_fit,
        _predict,
        y_true="y",
        metric=_r2,
        grid=param_grid(alpha=[0.01, 1.0, 100.0]),
        k=4,
        key="x",
    )
    assert seen[0:4] == seen[4:8] == seen[8:12]


def test_random_search_runs_the_requested_number_of_trials() -> None:
    found = random_search(
        _ds(),
        _fit,
        _predict,
        y_true="y",
        metric=_r2,
        distributions={"alpha": [0.01, 0.1, 1.0]},
        n_iter=3,
        k=4,
        key="x",
    )
    assert len(found.trials) == 3
    assert found.best_score > 0.99


def test_random_search_is_reproducible() -> None:
    def run() -> dict:
        return random_search(
            _ds(),
            _fit,
            _predict,
            y_true="y",
            metric=_r2,
            distributions={"alpha": [0.01, 0.1, 1.0, 10.0]},
            n_iter=4,
            seed=3,
            k=4,
            key="x",
        ).best_params

    assert run() == run()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid": []}, "at least one parameter combination"),
        ({"grid": [{"alpha": 1.0}], "k": 1}, "at least 2 folds"),
    ],
)
def test_grid_search_configuration_is_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        grid_search(_ds(), _fit, _predict, y_true="y", metric=_r2, key="x", **kwargs)


def test_search_result_renders_as_a_dataset() -> None:
    found = grid_search(
        _ds(),
        _fit,
        _predict,
        y_true="y",
        metric=_r2,
        grid=param_grid(alpha=[0.01, 100.0]),
        k=4,
        key="x",
    )
    table = found.to_dataset().to_pydict()
    assert table["alpha"] == [0.01, 100.0]
    assert set(table) == {"alpha", "mean_score", "std_score"}


def test_search_result_is_immutable() -> None:
    found = SearchResult(best_params={"a": 1}, best_score=1.0, trials=[])
    with pytest.raises((AttributeError, TypeError)):
        found.best_score = 2.0
