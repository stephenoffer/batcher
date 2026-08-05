"""Classification rates on the inputs that make their denominator zero.

`precision` is ``tp / (tp + fp)``. On a batch where nothing was predicted positive that is
0/0, and plain division answered NaN. NaN is defensible in isolation and wrong in place: the
docstrings promise a value in ``[0, 1]``, and it spreads. One fold with no positive
predictions makes the mean over folds NaN, so a cross-validation score or a streaming metric
goes quietly undefined because of a single batch - and on imbalanced data, a batch with no
positives is the ordinary case rather than a rare one.

scikit-learn answers 0.0 here (``zero_division=0``), so that is what these check, case by
case and against sklearn itself where it is installed.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

sk = pytest.importorskip("sklearn.metrics")

#: (label, y_true, y_pred) for every way a rate's denominator can reach zero.
DEGENERATE = [
    ("nothing positive anywhere", [0, 0, 0, 0], [0, 0, 0, 0]),
    ("positives exist, none predicted", [1, 0, 1, 0], [0, 0, 0, 0]),
    ("no true positives, some predicted", [0, 0, 0, 0], [1, 0, 1, 0]),
    ("nothing negative anywhere", [1, 1, 1, 1], [1, 1, 1, 1]),
]


def _value(y: list[int], p: list[int], expr) -> float:
    return bt.from_pydict({"y": y, "p": p}).agg(m=expr).to_pydict()["m"][0]


@pytest.mark.parametrize(("label", "y", "p"), DEGENERATE)
@pytest.mark.parametrize("name", ["precision", "recall", "specificity", "f1_score", "fbeta_score"])
def test_no_rate_is_nan_on_a_degenerate_batch(name: str, label: str, y: list, p: list) -> None:
    got = _value(y, p, getattr(bt, name)("y", "p"))
    assert got is not None, f"{name} returned null on {label}"
    assert not math.isnan(got), f"{name} returned NaN on {label}"
    assert 0.0 <= got <= 1.0, f"{name} returned {got}, outside the documented [0, 1]"


@pytest.mark.parametrize(("label", "y", "p"), DEGENERATE)
def test_precision_recall_and_f1_match_sklearn_exactly(label: str, y: list, p: list) -> None:
    got = (
        _value(y, p, bt.precision("y", "p")),
        _value(y, p, bt.recall("y", "p")),
        _value(y, p, bt.f1_score("y", "p")),
    )
    expected = (
        sk.precision_score(y, p, zero_division=0),
        sk.recall_score(y, p, zero_division=0),
        sk.f1_score(y, p, zero_division=0),
    )
    assert got == pytest.approx(expected), label


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_fbeta_matches_sklearn_when_undefined(beta: float) -> None:
    y, p = [1, 0, 1, 0], [0, 0, 0, 0]
    got = _value(y, p, bt.fbeta_score("y", "p", beta=beta))
    assert got == pytest.approx(sk.fbeta_score(y, p, beta=beta, zero_division=0))


@pytest.mark.parametrize(
    ("y", "p"),
    [([1, 0, 1, 0], [1, 0, 0, 0]), ([1, 1, 0, 0], [1, 0, 1, 0]), ([1, 0, 0, 1], [1, 0, 0, 1])],
)
def test_the_ordinary_cases_are_unchanged(y: list, p: list) -> None:
    """The guard must only affect the zero denominator, or it rewrites every score."""
    assert _value(y, p, bt.precision("y", "p")) == pytest.approx(
        sk.precision_score(y, p, zero_division=0)
    )
    assert _value(y, p, bt.recall("y", "p")) == pytest.approx(
        sk.recall_score(y, p, zero_division=0)
    )
    assert _value(y, p, bt.f1_score("y", "p")) == pytest.approx(sk.f1_score(y, p, zero_division=0))


def test_a_degenerate_batch_no_longer_poisons_an_average() -> None:
    """The reason this matters: NaN spreads, so one empty batch used to undefine the lot.

    This is the shape of a per-group score, a per-fold cross-validation score, and a
    streaming metric over windows - the batch with no positives is not an edge case on
    imbalanced data, it is most of them.
    """
    ds = bt.from_pydict(
        {
            "fold": ["a", "a", "b", "b", "c", "c"],
            "y": [1, 0, 0, 0, 1, 1],
            "p": [1, 0, 0, 0, 1, 0],
        }
    )
    per_fold = ds.group_by("fold").agg(score=bt.precision("y", "p")).to_pydict()
    scores = per_fold["score"]
    assert not any(math.isnan(v) for v in scores), f"a fold went NaN: {per_fold}"
    assert not math.isnan(sum(scores) / len(scores))


def test_the_false_negative_rate_is_zero_when_there_are_no_positives() -> None:
    assert _value([0, 0, 0, 0], [0, 0, 0, 0], bt.false_negative_rate("y", "p")) == 0.0


#: A constant target has no variance, so every variance-ratio score divides by zero.
CONSTANT_TARGET = [
    ("perfect prediction", [5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]),
    ("imperfect prediction", [5.0, 5.0, 5.0, 5.0], [4.0, 6.0, 5.0, 5.0]),
]


@pytest.mark.parametrize(("label", "y", "p"), CONSTANT_TARGET)
def test_r2_is_finite_on_a_constant_target(label: str, y: list, p: list) -> None:
    """It answered NaN when the prediction was perfect and -inf when it was not.

    A constant target is not exotic: a filtered group, a degenerate fold, and a
    single-valued segment in a `group_by` all produce one, and -inf spreads through a mean
    exactly as NaN does.
    """
    got = _value(y, p, bt.r2("y", "p"))
    assert not math.isnan(got), f"r2 was NaN on {label}"
    assert math.isfinite(got), f"r2 was {got} on {label}"


@pytest.mark.parametrize(("label", "y", "p"), CONSTANT_TARGET)
def test_r2_and_explained_variance_match_sklearn_on_a_constant_target(
    label: str, y: list, p: list
) -> None:
    assert _value(y, p, bt.r2("y", "p")) == pytest.approx(sk.r2_score(y, p)), label
    assert _value(y, p, bt.explained_variance("y", "p")) == pytest.approx(
        sk.explained_variance_score(y, p)
    ), label


@pytest.mark.parametrize(
    ("y", "p"),
    [
        ([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8]),
        ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_the_ordinary_variance_ratios_are_unchanged(y: list, p: list) -> None:
    """Including the anti-correlated case, whose r2 is -3: the guard must not clamp."""
    assert _value(y, p, bt.r2("y", "p")) == pytest.approx(sk.r2_score(y, p))
    assert _value(y, p, bt.explained_variance("y", "p")) == pytest.approx(
        sk.explained_variance_score(y, p)
    )


def test_a_constant_group_no_longer_poisons_a_grouped_score() -> None:
    ds = bt.from_pydict(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "y": [7.0, 7.0, 7.0, 1.0, 2.0, 3.0],
            "p": [7.0, 7.0, 7.0, 1.1, 2.1, 2.9],
        }
    )
    scores = ds.group_by("g").agg(score=bt.r2("y", "p")).to_pydict()["score"]
    assert all(math.isfinite(v) for v in scores), f"a group went non-finite: {scores}"
