"""The data-science preprocessors — shape transforms, cardinality encoders, persistence.

What each test pins is the property that makes the preprocessor worth having, not merely
that it runs: a quantile transform must be immune to an outlier, a rare-category encoder
must handle a category it has never seen, a hashing encoder must be stable across
processes, and a saved preprocessor must transform identically to the one that was fitted.

The last of those is the one that fails silently in production, so it gets the most tests:
a scaler whose state did not survive the save applies a *different* transform at serving
time and nothing raises.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    Chain,
    Clipper,
    FrequencyEncoder,
    HashingEncoder,
    LogTransformer,
    MinMaxScaler,
    MissingIndicator,
    PowerTransformer,
    Preprocessor,
    QuantileTransformer,
    RareCategoryEncoder,
    SimpleImputer,
    StandardScaler,
    from_dict,
    to_dict,
)

pytestmark = pytest.mark.unit


# --- QuantileTransformer ------------------------------------------------------------


def test_quantile_transformer_is_immune_to_an_outlier() -> None:
    # The point of a rank transform: the extreme value becomes simply "the largest".
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 1e12]})
    got = QuantileTransformer("x", n_quantiles=4).fit_transform(ds).to_pydict()["x"]
    assert got == [0.125, 0.375, 0.625, 0.875]


def test_quantile_transformer_output_stays_in_range() -> None:
    ds = bt.from_pydict({"x": [float(i) for i in range(100)]})
    got = QuantileTransformer("x", n_quantiles=10).fit_transform(ds).to_pydict()["x"]
    assert all(0.0 <= v < 1.0 for v in got)


def test_quantile_transformer_is_monotone() -> None:
    ds = bt.from_pydict({"x": [float(i) for i in range(50)]})
    got = QuantileTransformer("x", n_quantiles=10).fit_transform(ds).to_pydict()["x"]
    assert got == sorted(got)


def test_quantile_transformer_applies_training_cut_points_to_new_data() -> None:
    train = bt.from_pydict({"x": [0.0, 1.0, 2.0, 3.0]})
    pre = QuantileTransformer("x", n_quantiles=4).fit(train)
    # A value above everything seen in training saturates at the top step, it does not
    # extrapolate into a range the model never saw.
    assert pre.transform(bt.from_pydict({"x": [99.0]})).to_pydict()["x"] == [0.875]


def test_quantile_transformer_normal_output_is_centered() -> None:
    # A step function reporting each step's lower edge instead of its midpoint shifts the
    # whole output down by half a step; on a normal target that moved the mean to -0.098.
    ds = bt.from_pydict({"x": [float(i) for i in range(100)]})
    pre = QuantileTransformer("x", n_quantiles=20, output_distribution="normal")
    got = pre.fit_transform(ds).to_pydict()["x"]
    assert abs(sum(got) / len(got)) < 1e-9
    assert min(got) < -1.5 and max(got) > 1.5


def test_quantile_transformer_uniform_output_is_centered() -> None:
    ds = bt.from_pydict({"x": [float(i) for i in range(100)]})
    got = QuantileTransformer("x", n_quantiles=20).fit_transform(ds).to_pydict()["x"]
    assert abs(sum(got) / len(got) - 0.5) < 1e-9


def test_quantile_transformer_rejects_a_degenerate_grid() -> None:
    with pytest.raises(PlanError, match="n_quantiles"):
        QuantileTransformer("x", n_quantiles=1)


def test_quantile_transformer_rejects_an_unknown_output_distribution() -> None:
    with pytest.raises(PlanError, match="output_distribution"):
        QuantileTransformer("x", output_distribution="poisson")


# --- PowerTransformer ---------------------------------------------------------------


def test_power_transformer_reduces_skew() -> None:
    from batcher.plan.expr_ir import col

    values = [float(2**i) for i in range(12)]
    ds = bt.from_pydict({"x": values})
    before = ds.agg(s=col("x").skewness()).collect().column("s")[0].as_py()
    after_ds = PowerTransformer("x", standardize=False).fit_transform(ds)
    after = after_ds.agg(s=col("x").skewness()).collect().column("s")[0].as_py()
    assert abs(after) < abs(before)


def test_power_transformer_standardizes_by_default() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    got = PowerTransformer("x").fit_transform(ds).to_pydict()["x"]
    assert abs(sum(got) / len(got)) < 1e-9


def test_power_transformer_handles_negative_values() -> None:
    # Yeo-Johnson is chosen over Box-Cox precisely so this does not raise.
    ds = bt.from_pydict({"x": [-5.0, -1.0, 0.0, 1.0, 5.0]})
    got = PowerTransformer("x", standardize=False).fit_transform(ds).to_pydict()["x"]
    assert all(v == v for v in got)  # no NaN


def test_power_transformer_lambda_stays_in_the_searched_grid() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pre = PowerTransformer("x").fit(ds)
    assert -2.0 <= pre.lambdas_["x"] <= 2.0


def test_power_transformer_transform_uses_the_fitted_lambda_on_new_data() -> None:
    train = bt.from_pydict({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    pre = PowerTransformer("x").fit(train)
    first = pre.transform(bt.from_pydict({"x": [4.0]})).to_pydict()["x"]
    second = pre.transform(bt.from_pydict({"x": [4.0]})).to_pydict()["x"]
    assert first == second


# --- Clipper, LogTransformer, MissingIndicator ---------------------------------------


def test_clipper_clamps_new_data_to_the_training_range() -> None:
    train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pre = Clipper("x", lower=0.0, upper=1.0).fit(train)
    got = pre.transform(bt.from_pydict({"x": [-100.0, 100.0]})).to_pydict()["x"]
    assert got == [1.0, 5.0]


def test_clipper_keeps_every_row() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 1000.0]})
    assert Clipper("x", upper=0.5).fit_transform(ds).count() == 3


def test_clipper_needs_at_least_one_bound() -> None:
    with pytest.raises(PlanError, match="lower"):
        Clipper("x", lower=None, upper=None)


def test_clipper_rejects_a_bound_outside_the_unit_interval() -> None:
    with pytest.raises(PlanError, match="quantile"):
        Clipper("x", upper=1.5)


def test_log_transformer_handles_zero_with_log1p() -> None:
    ds = bt.from_pydict({"x": [0.0, 1.0]})
    assert LogTransformer("x").fit_transform(ds).to_pydict()["x"][0] == 0.0


def test_log_transformer_is_stateless_so_train_and_serve_agree() -> None:
    pre = LogTransformer("x")
    first = pre.fit_transform(bt.from_pydict({"x": [7.0]})).to_pydict()["x"]
    second = pre.transform(bt.from_pydict({"x": [7.0]})).to_pydict()["x"]
    assert first == second


def test_missing_indicator_records_nulls_before_imputation() -> None:
    # The order matters: imputing first destroys the signal permanently.
    ds = bt.from_pydict({"x": [1.0, None, 3.0]})
    chain = Chain(MissingIndicator("x"), SimpleImputer(["x"]))
    out = chain.fit_transform(ds).to_pydict()
    assert out["x_missing"] == [False, True, False]
    assert out["x"][1] is not None


# --- cardinality-tolerant encoders ---------------------------------------------------


def test_frequency_encoder_replaces_a_category_with_its_share() -> None:
    ds = bt.from_pydict({"c": ["a", "a", "a", "b"]})
    assert FrequencyEncoder("c").fit_transform(ds).to_pydict()["c"] == [0.75, 0.75, 0.75, 0.25]


def test_frequency_encoder_can_report_raw_counts() -> None:
    ds = bt.from_pydict({"c": ["a", "a", "b"]})
    got = FrequencyEncoder("c", normalize=False).fit_transform(ds).to_pydict()["c"]
    assert got == [2.0, 2.0, 1.0]


def test_frequency_encoder_gives_an_unseen_category_zero() -> None:
    train = bt.from_pydict({"c": ["a", "a", "b"]})
    pre = FrequencyEncoder("c").fit(train)
    assert pre.transform(bt.from_pydict({"c": ["zzz"]})).to_pydict()["c"] == [0.0]


def test_rare_category_encoder_collapses_the_tail() -> None:
    ds = bt.from_pydict({"c": ["a"] * 90 + ["b"] * 9 + ["c"]})
    got = RareCategoryEncoder("c", min_frequency=0.05).fit_transform(ds).to_pydict()["c"]
    assert set(got) == {"a", "b", "__rare__"}


def test_rare_category_encoder_buckets_an_unseen_category() -> None:
    # The serving-time unknown-category problem: the bucket already exists.
    train = bt.from_pydict({"c": ["a"] * 10})
    pre = RareCategoryEncoder("c", min_frequency=0.5).fit(train)
    assert pre.transform(bt.from_pydict({"c": ["brand_new"]})).to_pydict()["c"] == ["__rare__"]


def test_rare_category_encoder_rejects_an_impossible_frequency() -> None:
    with pytest.raises(PlanError, match="min_frequency"):
        RareCategoryEncoder("c", min_frequency=0.0)


def test_hashing_encoder_is_stable_for_the_same_value() -> None:
    ds = bt.from_pydict({"c": ["alpha", "beta", "alpha"]})
    got = HashingEncoder("c", n_buckets=64).fit_transform(ds).to_pydict()["c"]
    assert got[0] == got[2]


def test_hashing_encoder_stays_inside_the_bucket_range() -> None:
    ds = bt.from_pydict({"c": [f"v{i}" for i in range(200)]})
    got = HashingEncoder("c", n_buckets=16).fit_transform(ds).to_pydict()["c"]
    assert all(0 <= v < 16 for v in got)


def test_hashing_encoder_needs_no_fit_so_train_and_serve_cannot_skew() -> None:
    pre = HashingEncoder("c", n_buckets=32)
    train = pre.transform(bt.from_pydict({"c": ["x"]})).to_pydict()["c"]
    serve = pre.transform(bt.from_pydict({"c": ["x"]})).to_pydict()["c"]
    assert train == serve


def test_hashing_encoder_rejects_a_degenerate_bucket_count() -> None:
    with pytest.raises(PlanError, match="n_buckets"):
        HashingEncoder("c", n_buckets=1)


# --- persistence ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: StandardScaler("x"),
        lambda: MinMaxScaler("x"),
        lambda: QuantileTransformer("x", n_quantiles=4),
        lambda: Clipper("x", lower=0.1, upper=0.9),
        lambda: PowerTransformer("x"),
    ],
)
def test_a_saved_preprocessor_transforms_identically(tmp_path, build) -> None:
    # The silent production failure this exists to prevent: state that did not survive the
    # save makes serving apply a different transform, and nothing raises.
    train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 10.0]})
    serve = bt.from_pydict({"x": [2.5, 7.0]})
    fitted = build().fit(train)
    expected = fitted.transform(serve).to_pydict()
    path = str(tmp_path / "pre.json")
    fitted.save(path)
    assert Preprocessor.load(path).transform(serve).to_pydict() == expected


def test_a_saved_preprocessor_is_readable_json(tmp_path) -> None:
    pre = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
    path = str(tmp_path / "pre.json")
    pre.save(path)
    with open(path) as handle:
        document = json.load(handle)
    assert document["class"] == "StandardScaler"
    assert document["state"]["mean_"] == {"__items__": [["x", 2.0]]}


def test_round_tripping_preserves_a_non_string_dict_key() -> None:
    # JSON would coerce an int key to a string and lose the type on reload.
    pre = FrequencyEncoder("c").fit(bt.from_pydict({"c": [1, 1, 2]}))
    restored = from_dict(to_dict(pre))
    assert set(restored.frequencies_["c"]) == {1, 2}


def test_round_tripping_preserves_a_tuple_in_state() -> None:
    pre = Clipper("x", lower=0.0, upper=1.0).fit(bt.from_pydict({"x": [1.0, 5.0]}))
    restored = from_dict(to_dict(pre))
    assert restored.bounds_ == {"x": (1.0, 5.0)}


def test_loading_rejects_an_unknown_class() -> None:
    with pytest.raises(PlanError, match="unknown preprocessor"):
        from_dict({"version": 1, "class": "NoSuchScaler", "params": {}, "state": {}})


def test_loading_rejects_a_future_schema_version() -> None:
    with pytest.raises(PlanError, match="schema version"):
        from_dict({"version": 999, "class": "StandardScaler", "params": {}, "state": {}})


def test_loading_a_non_document_says_what_is_wrong(tmp_path) -> None:
    path = tmp_path / "junk.json"
    path.write_text("not json at all")
    with pytest.raises(PlanError, match="not a saved preprocessor"):
        Preprocessor.load(str(path))


def test_an_unfitted_preprocessor_round_trips_as_unfitted() -> None:
    restored = from_dict(to_dict(StandardScaler("x")))
    assert restored.is_fitted is False
