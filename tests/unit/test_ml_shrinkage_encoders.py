"""`LeaveOneOutEncoder` and `JamesSteinEncoder`.

Both answer "how much should I believe this category's own mean?", so the tests are mostly
about the *shape* of that answer rather than about particular numbers: leave-one-out must
never let a row see its own target, and James-Stein must shrink a noisy category harder than
a clean one. A test pinning single values would pass for an encoder that shrank everything
to the prior.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    JamesSteinEncoder,
    LeaveOneOutEncoder,
    Preprocessor,
    TargetEncoder,
)

pytestmark = pytest.mark.unit


def test_leave_one_out_excludes_the_rows_own_target() -> None:
    ds = bt.from_pydict({"c": ["a", "a", "a"], "y": [0.0, 3.0, 6.0]})
    assert LeaveOneOutEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"] == [4.5, 3.0, 1.5]


def test_leave_one_out_is_the_mean_of_the_others_for_every_row() -> None:
    """Checked against the definition directly, over a frame with uneven category sizes."""
    categories = ["a", "a", "a", "b", "b", "c"]
    targets = [1.0, 2.0, 6.0, 4.0, 10.0, 7.0]
    ds = bt.from_pydict({"c": categories, "y": targets})
    got = LeaveOneOutEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"]
    prior = sum(targets) / len(targets)
    for index, category in enumerate(categories):
        others = [
            t
            for j, (c, t) in enumerate(zip(categories, targets, strict=True))
            if c == category and j != index
        ]
        want = sum(others) / len(others) if others else prior
        assert got[index] == pytest.approx(want)


def test_a_singleton_category_falls_back_to_the_prior() -> None:
    """Removing the only row leaves nothing to average, so it must not divide by zero."""
    ds = bt.from_pydict({"c": ["a", "a", "solo"], "y": [1.0, 3.0, 9.0]})
    encoded = LeaveOneOutEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"]
    assert encoded[2] == pytest.approx(13.0 / 3.0)


def test_leave_one_out_on_a_held_out_split_is_the_plain_mean() -> None:
    train = bt.from_pydict({"c": ["a", "a", "b"], "y": [1.0, 3.0, 5.0]})
    encoder = LeaveOneOutEncoder(["c"], "y").fit(train)
    assert encoder.transform(bt.from_pydict({"c": ["a", "b"]})).to_pydict()["c"] == [2.0, 5.0]


def test_an_unseen_category_encodes_as_the_prior() -> None:
    train = bt.from_pydict({"c": ["a", "a", "b"], "y": [1.0, 3.0, 5.0]})
    for encoder in (LeaveOneOutEncoder(["c"], "y"), JamesSteinEncoder(["c"], "y")):
        fitted = encoder.fit(train)
        got = fitted.transform(bt.from_pydict({"c": ["brand-new"]})).to_pydict()["c"]
        assert got == [pytest.approx(fitted.prior_)]


def test_leave_one_out_never_reproduces_the_rows_own_label() -> None:
    """The whole point: a perfect-signal column must not become a copy of the target."""
    ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
    plain = TargetEncoder(["c"], "y", smoothing=0.0).fit_transform(ds).to_pydict()["c"]
    loo = LeaveOneOutEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"]
    assert plain == [1.0, 1.0, 0.0, 0.0]  # the leak TargetEncoder has without cv
    assert loo == [1.0, 1.0, 0.0, 0.0]  # here both rows agree, so the value coincides
    uneven = bt.from_pydict({"c": ["a", "a", "a"], "y": [1.0, 1.0, 0.0]})
    assert LeaveOneOutEncoder(["c"], "y").fit_transform(uneven).to_pydict()["c"] == [
        0.5,
        0.5,
        1.0,
    ]


def test_james_stein_lies_between_the_category_mean_and_the_prior() -> None:
    ds = bt.from_pydict({"c": ["a"] * 5 + ["b"] * 5, "y": [1.0] * 5 + [0.0] * 5})
    fitted = JamesSteinEncoder(["c"], "y").fit(ds)
    assert fitted.prior_ == pytest.approx(0.5)
    for category, mean in (("a", 1.0), ("b", 0.0)):
        encoded = fitted.mapping_["c"][category]
        assert min(mean, fitted.prior_) <= encoded <= max(mean, fitted.prior_)


def test_james_stein_shrinks_a_noisy_category_harder_than_a_clean_one() -> None:
    """The property that distinguishes it from a fixed smoothing weight."""
    clean = ["clean"] * 20
    noisy = ["noisy"] * 20
    ds = bt.from_pydict(
        {
            "c": clean + noisy,
            "y": [1.0] * 20 + [1.0 if i % 2 else 0.0 for i in range(20)],
        }
    )
    fitted = JamesSteinEncoder(["c"], "y").fit(ds)
    prior = fitted.prior_
    clean_gap = abs(fitted.mapping_["c"]["clean"] - prior)
    noisy_gap = abs(fitted.mapping_["c"]["noisy"] - prior)
    assert clean_gap > noisy_gap


def test_james_stein_trusts_a_large_category_more_than_a_small_one() -> None:
    ds = bt.from_pydict({"c": ["big"] * 40 + ["small"] * 2, "y": [1.0] * 40 + [1.0, 1.0]})
    fitted = JamesSteinEncoder(["c"], "y").fit(ds)
    assert abs(fitted.mapping_["c"]["big"] - fitted.prior_) >= abs(
        fitted.mapping_["c"]["small"] - fitted.prior_
    )


@pytest.mark.parametrize("klass", [LeaveOneOutEncoder, JamesSteinEncoder])
def test_fit_is_independent_of_partitioning(klass) -> None:
    rows = {"c": ["a", "b", "a", "b", "c"] * 4, "y": [1.0, 0.0, 1.0, 0.5, 0.0] * 4}
    one = klass(["c"], "y").fit(bt.from_pydict(rows))
    many = klass(["c"], "y").fit(bt.from_pydict(rows).repartition(4))
    assert one.prior_ == pytest.approx(many.prior_)
    assert (
        one.transform(bt.from_pydict({"c": ["a", "b", "c"]})).to_pydict()
        == many.transform(bt.from_pydict({"c": ["a", "b", "c"]})).to_pydict()
    )


@pytest.mark.parametrize("klass", [LeaveOneOutEncoder, JamesSteinEncoder])
def test_transform_before_fit_names_the_class(klass) -> None:
    with pytest.raises(PlanError, match="must be fitted"):
        klass(["c"], "y").transform(bt.from_pydict({"c": ["a"], "y": [1.0]}))


@pytest.mark.parametrize("klass", [LeaveOneOutEncoder, JamesSteinEncoder])
def test_an_unbounded_category_set_is_refused(klass) -> None:
    ds = bt.from_pydict({"c": [f"id{i}" for i in range(50)], "y": [1.0] * 50})
    with pytest.raises(PlanError, match="max_categories"):
        klass(["c"], "y", max_categories=10).fit(ds)


@pytest.mark.parametrize("klass", [LeaveOneOutEncoder, JamesSteinEncoder])
def test_a_fitted_encoder_round_trips_through_save(klass, tmp_path) -> None:
    ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 0.0, 1.0, 1.0]})
    fitted = klass(["c"], "y").fit(ds)
    target = str(tmp_path / "encoder.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    assert restored.transform(ds).to_pydict()["c"] == fitted.transform(ds).to_pydict()["c"]


@pytest.mark.parametrize("klass", [LeaveOneOutEncoder, JamesSteinEncoder])
def test_encoders_compose_in_a_chain(klass) -> None:
    from batcher.ml.preprocessors import Chain, StandardScaler

    ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 0.0, 1.0, 1.0]})
    out = Chain(klass(["c"], "y"), StandardScaler(["c"])).fit_transform(ds)
    assert out.count() == 4


@pytest.mark.parametrize("klass", [LeaveOneOutEncoder, JamesSteinEncoder])
def test_a_null_category_takes_the_prior_rather_than_forming_a_group(klass) -> None:
    """A missing value is not a category, so it must not learn a mean of its own."""
    ds = bt.from_pydict({"c": ["a", "a", None], "y": [1.0, 3.0, 9.0]})
    fitted = klass(["c"], "y").fit(ds)
    learned = getattr(fitted, "counts_", None) or fitted.mapping_
    assert None not in learned["c"]
    encoded = fitted.transform(bt.from_pydict({"c": ["a", None]})).to_pydict()["c"]
    assert encoded[1] == pytest.approx(fitted.prior_)
