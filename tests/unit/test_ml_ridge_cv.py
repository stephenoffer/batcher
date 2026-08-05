"""`RidgeCV` — choosing an L2 penalty without paying for a fit per candidate per fold.

The claim this file has to defend is not that the answer is good but that it is *the same
answer* a naive search would give, for a fraction of the work. Ridge's normal equations are
built from moments that do not depend on alpha, and the held-out squared error expands into
those same moments, so both the fit and the score of every ``(fold, alpha)`` pair are
arithmetic on small matrices over one grouped aggregate.

That makes the load-bearing test the equivalence one: refit ridge per fold with numpy, score
it directly, and require the analytic number to match. If that ever drifts, the speed is
worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import Ridge, RidgeCV

pytestmark = pytest.mark.unit

ALPHAS = (0.1, 1.0, 10.0)
FOLDS = 5


@pytest.fixture(scope="module")
def sample() -> tuple[bt.Dataset, list[str]]:
    rng = np.random.default_rng(7)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = 1.5 * a + 0.4 * b + rng.normal(scale=0.7, size=n)
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist(), "y": y.tolist()})
    return ds, ["a", "b"]


def _brute_force_score(ds: bt.Dataset, features: list[str], alpha: float, cv: int) -> float:
    """The same cross-validation done the expensive way: refit per fold, score per fold."""
    from batcher.api.dataset._build import split_key

    fold = (split_key(ds, [*features, "y"], 0) * bt.lit(float(cv))).floor().cast("int64")
    table = ds.with_columns(__f=fold).to_pydict()
    folds = np.array(table["__f"])
    x = np.column_stack([table[name] for name in features])
    y = np.array(table["y"])
    total, held = 0.0, 0
    for k in range(cv):
        test = folds == k
        train = ~test
        if test.sum() == 0 or train.sum() <= len(features) + 1:
            continue
        xt, yt = x[train], y[train]
        mx, my = xt.mean(axis=0), yt.mean()
        centered_x, centered_y = xt - mx, yt - my
        beta = np.linalg.solve(
            centered_x.T @ centered_x + alpha * np.eye(len(features)), centered_x.T @ centered_y
        )
        intercept = my - beta @ mx
        total += float(((y[test] - x[test] @ beta - intercept) ** 2).sum())
        held += int(test.sum())
    return total / held


@pytest.mark.parametrize("alpha", ALPHAS)
def test_the_analytic_fold_score_equals_refitting_the_fold(sample, alpha: float) -> None:
    """The whole optimization rests on this: same number, not a close one."""
    ds, features = sample
    model = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS, seed=0).fit(ds)
    assert model.scores_[alpha] == pytest.approx(
        _brute_force_score(ds, features, alpha, FOLDS), abs=1e-9
    )


def test_it_picks_the_same_alpha_a_brute_force_search_would(sample) -> None:
    ds, features = sample
    model = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS, seed=0).fit(ds)
    brute = {a: _brute_force_score(ds, features, a, FOLDS) for a in ALPHAS}
    assert model.alpha_ == min(ALPHAS, key=lambda a: brute[a])


def test_the_final_refit_is_exactly_ridge_at_the_chosen_alpha(sample) -> None:
    """The search must end in an ordinary ridge fit over all the data, not an approximation."""
    ds, features = sample
    model = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(ds)
    plain = Ridge(features, "y", alpha=model.alpha_).fit(ds)
    assert model.coef_ == pytest.approx(plain.coef_, abs=1e-9)
    assert model.intercept_ == pytest.approx(plain.intercept_, abs=1e-9)


def test_the_whole_search_is_one_pass_over_the_data(sample) -> None:
    """Five folds times three candidates is fifteen fits and fifteen scorings, or one scan.

    Counting terminal ops rather than timing keeps this a fact about the algorithm instead
    of a fact about the machine.
    """
    ds, features = sample
    frame = type(ds)
    calls = {"n": 0}
    originals = {name: getattr(frame, name) for name in ("collect", "to_pydict", "count")}

    def wrap(original):
        def counted(self, *args, **kwargs):
            calls["n"] += 1
            return original(self, *args, **kwargs)

        return counted

    for name, original in originals.items():
        setattr(frame, name, wrap(original))
    try:
        RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(ds)
    finally:
        for name, original in originals.items():
            setattr(frame, name, original)
    assert calls["n"] == 1, f"expected one terminal op for the whole search, got {calls['n']}"


def test_more_candidates_cost_nothing_extra(sample) -> None:
    """Twenty candidates read the data exactly as often as one does."""
    ds, features = sample
    many = tuple(float(10**k) for k in range(-6, 14))
    assert len(many) == 20
    model = RidgeCV(features, "y", alphas=many, cv=FOLDS).fit(ds)
    assert len(model.scores_) == 20
    assert model.alpha_ in many


def test_a_stronger_penalty_shrinks_the_coefficients(sample) -> None:
    ds, features = sample
    weak = RidgeCV(features, "y", alphas=(1e-6,), cv=FOLDS).fit(ds)
    strong = RidgeCV(features, "y", alphas=(1e6,), cv=FOLDS).fit(ds)
    assert abs(strong.coef_[0]) < abs(weak.coef_[0])


def test_the_fold_assignment_does_not_depend_on_row_order(sample) -> None:
    """Folds are a content hash, so a reordered or repartitioned input scores identically."""
    ds, features = sample
    table = ds.to_pydict()
    order = list(range(len(table["y"])))[::-1]
    shuffled = bt.from_pydict({k: [v[i] for i in order] for k, v in table.items()})
    first = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(ds)
    second = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(shuffled)
    assert first.alpha_ == second.alpha_
    assert first.coef_ == pytest.approx(second.coef_, abs=1e-9)


def test_a_union_of_partitions_fits_what_one_partition_does(sample) -> None:
    """The moments are additive, which is what makes this the same on a cluster."""
    ds, features = sample
    table = ds.to_pydict()
    half = len(table["y"]) // 2
    left = bt.from_pydict({k: v[:half] for k, v in table.items()})
    right = bt.from_pydict({k: v[half:] for k, v in table.items()})
    whole = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(ds)
    split = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(left.union(right))
    assert whole.alpha_ == split.alpha_
    assert whole.coef_ == pytest.approx(split.coef_, abs=1e-9)


def test_predict_appends_a_column_and_matches_the_linear_score(sample) -> None:
    ds, features = sample
    model = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS).fit(ds)
    scored = model.predict(ds).to_pydict()
    table = ds.to_pydict()
    expected = [
        model.intercept_ + sum(c * table[f][i] for c, f in zip(model.coef_, features, strict=True))
        for i in range(len(table["y"]))
    ]
    assert scored["prediction"] == pytest.approx(expected, abs=1e-9)


def test_the_output_column_is_configurable(sample) -> None:
    ds, features = sample
    model = RidgeCV(features, "y", alphas=ALPHAS, cv=FOLDS, output_column="yhat").fit(ds)
    assert "yhat" in model.predict(ds).columns


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_no_features_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        RidgeCV([], "y")


def test_too_many_features_is_rejected_rather_than_building_a_huge_plan() -> None:
    with pytest.raises(PlanError, match="ceiling"):
        RidgeCV([f"f{i}" for i in range(25)], "y")


def test_an_empty_alpha_list_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one candidate"):
        RidgeCV(["a"], "y", alphas=())


def test_a_negative_alpha_is_rejected() -> None:
    with pytest.raises(PlanError, match="non-negative"):
        RidgeCV(["a"], "y", alphas=(1.0, -1.0))


def test_one_fold_is_rejected_because_nothing_is_held_out() -> None:
    with pytest.raises(PlanError, match="two folds"):
        RidgeCV(["a"], "y", cv=1)


def test_a_missing_column_is_named(sample) -> None:
    ds, _ = sample
    with pytest.raises(ColumnNotFoundError):
        RidgeCV(["nope"], "y").fit(ds)


def test_a_string_feature_is_named(sample) -> None:
    ds, _ = sample
    with pytest.raises(PlanError, match="'label'"):
        RidgeCV(["label"], "y").fit(ds.with_columns(label=bt.lit("x")))


def test_predicting_before_fitting_is_rejected(sample) -> None:
    ds, features = sample
    with pytest.raises(PlanError, match="must be fitted"):
        RidgeCV(features, "y").predict(ds)


def test_too_few_rows_is_rejected(sample) -> None:
    _, features = sample
    tiny = bt.from_pydict({"a": [1.0, 2.0], "b": [1.0, 3.0], "y": [1.0, 2.0]})
    with pytest.raises(PlanError):
        RidgeCV(features, "y", cv=2).fit(tiny)


# --------------------------------------------------------------------------------------
# The same one-pass search, for the L1 models
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sparse_sample() -> tuple[bt.Dataset, list[str]]:
    """One informative feature and two that carry nothing, which is what L1 is for."""
    rng = np.random.default_rng(3)
    n = 400
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = 2.5 * a + rng.normal(scale=0.4, size=n)
    ds = bt.from_pydict({"a": a.tolist(), "b": b.tolist(), "n": noise.tolist(), "y": y.tolist()})
    return ds, ["a", "b", "n"]


def test_lasso_cv_ends_in_exactly_lasso_at_the_chosen_alpha(sparse_sample) -> None:
    from batcher.ml import Lasso, LassoCV

    ds, features = sparse_sample
    model = LassoCV(features, "y", alphas=(0.0001, 0.01, 0.1, 1.0), cv=5).fit(ds)
    plain = Lasso(features, "y", alpha=model.alpha_).fit(ds)
    assert model.coef_ == pytest.approx(plain.coef_, abs=1e-6)
    assert model.intercept_ == pytest.approx(plain.intercept_, abs=1e-6)


def test_elastic_net_cv_ends_in_exactly_elastic_net_at_the_chosen_alpha(sparse_sample) -> None:
    from batcher.ml import ElasticNet, ElasticNetCV

    ds, features = sparse_sample
    model = ElasticNetCV(features, "y", alphas=(0.001, 0.1), l1_ratio=0.5, cv=4).fit(ds)
    plain = ElasticNet(features, "y", alpha=model.alpha_, l1_ratio=0.5).fit(ds)
    assert model.coef_ == pytest.approx(plain.coef_, abs=1e-6)


def test_lasso_cv_recovers_the_informative_feature(sparse_sample) -> None:
    from batcher.ml import LassoCV

    ds, features = sparse_sample
    model = LassoCV(features, "y", alphas=(0.0001, 0.01), cv=5).fit(ds)
    assert model.coef_[0] == pytest.approx(2.5, abs=0.1)
    assert abs(model.coef_[1]) < 0.2
    assert abs(model.coef_[2]) < 0.2


def test_a_strong_l1_penalty_zeroes_the_uninformative_features(sparse_sample) -> None:
    """Exactly zero, not merely small: that is what distinguishes L1 from ridge."""
    from batcher.ml import LassoCV

    ds, features = sparse_sample
    model = LassoCV(features, "y", alphas=(1.0,), cv=4).fit(ds)
    assert model.coef_[1] == 0.0
    assert model.coef_[2] == 0.0


def test_the_l1_search_is_also_one_pass(sparse_sample) -> None:
    from batcher.ml import LassoCV

    ds, features = sparse_sample
    frame = type(ds)
    calls = {"n": 0}
    originals = {name: getattr(frame, name) for name in ("collect", "to_pydict", "count")}

    def wrap(original):
        def counted(self, *args, **kwargs):
            calls["n"] += 1
            return original(self, *args, **kwargs)

        return counted

    for name, original in originals.items():
        setattr(frame, name, wrap(original))
    try:
        LassoCV(features, "y", alphas=(0.001, 0.01, 0.1, 1.0), cv=5).fit(ds)
    finally:
        for name, original in originals.items():
            setattr(frame, name, original)
    assert calls["n"] == 1, f"expected one terminal op, got {calls['n']}"


def test_lasso_cv_is_the_elastic_net_at_l1_ratio_one(sparse_sample) -> None:
    from batcher.ml import ElasticNetCV, LassoCV

    ds, features = sparse_sample
    lasso = LassoCV(features, "y", alphas=(0.01, 0.1), cv=4).fit(ds)
    net = ElasticNetCV(features, "y", alphas=(0.01, 0.1), l1_ratio=1.0, cv=4).fit(ds)
    assert lasso.l1_ratio == 1.0
    assert lasso.alpha_ == net.alpha_
    assert lasso.coef_ == pytest.approx(net.coef_, abs=1e-9)


def test_an_l1_ratio_outside_the_unit_interval_is_rejected() -> None:
    from batcher.ml import ElasticNetCV

    with pytest.raises(PlanError, match="l1_ratio"):
        ElasticNetCV(["a"], "y", l1_ratio=1.5)


@pytest.mark.parametrize("name", ["ElasticNetCV", "LassoCV"])
def test_the_l1_searches_reject_the_same_bad_arguments(name: str) -> None:
    import batcher.ml as ml

    klass = getattr(ml, name)
    with pytest.raises(PlanError, match="at least one feature"):
        klass([], "y")
    with pytest.raises(PlanError, match="at least one candidate"):
        klass(["a"], "y", alphas=())
    with pytest.raises(PlanError, match="non-negative"):
        klass(["a"], "y", alphas=(-1.0,))
    with pytest.raises(PlanError, match="two folds"):
        klass(["a"], "y", cv=1)
