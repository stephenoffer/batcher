"""Multinomial and Bernoulli naive Bayes.

Both fit from grouped feature sums and classify by a closed-form argmax, so the bar is an exact
prediction match with scikit-learn: `MultinomialNB` on integer count features, `BernoulliNB` on
the same features binarized. The smoothing and the absent-feature penalty are what the sklearn
match actually exercises.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.naive_bayes import BernoulliNB, MultinomialNB

pytestmark = pytest.mark.unit

sk_nb = pytest.importorskip("sklearn.naive_bayes")


@pytest.fixture(scope="module")
def counts() -> tuple[np.ndarray, np.ndarray, list[str], bt.Dataset]:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, (300, 4))
    y = rng.integers(0, 3, 300)
    names = [f"x{i}" for i in range(4)]
    ds = bt.from_pydict({**{n: x[:, i].tolist() for i, n in enumerate(names)}, "y": y.tolist()})
    return x, y, names, ds


def test_multinomial_matches_sklearn(counts) -> None:
    x, y, names, ds = counts
    got = np.array(MultinomialNB(names, "y").fit(ds).predict(ds).to_pydict()["prediction"])
    assert (got == sk_nb.MultinomialNB(alpha=1.0).fit(x, y).predict(x)).mean() == pytest.approx(1.0)


def test_bernoulli_matches_sklearn(counts) -> None:
    x, y, names, ds = counts
    got = np.array(BernoulliNB(names, "y").fit(ds).predict(ds).to_pydict()["prediction"])
    assert (got == sk_nb.BernoulliNB(alpha=1.0).fit(x, y).predict(x)).mean() == pytest.approx(1.0)


def test_multinomial_matches_sklearn_out_of_sample(counts) -> None:
    x, y, names, ds = counts
    model = MultinomialNB(names, "y").fit(ds)
    rng = np.random.default_rng(5)
    test = rng.integers(0, 5, (60, 4))
    dst = bt.from_pydict({n: test[:, i].tolist() for i, n in enumerate(names)})
    got = np.array(model.predict(dst).to_pydict()["prediction"])
    assert (got == sk_nb.MultinomialNB(alpha=1.0).fit(x, y).predict(test)).mean() == pytest.approx(
        1.0
    )


def test_bernoulli_penalizes_absence() -> None:
    # Class 0 always has feature a; class 1 always has feature b. A row with neither is decided
    # by which absence is more surprising, which only Bernoulli (not Multinomial) can express.
    ds = bt.from_pydict({"a": [1, 1, 0, 0], "b": [0, 0, 1, 1], "y": [0, 0, 1, 1]})
    model = BernoulliNB(["a", "b"], "y").fit(ds)
    labels = model.predict(bt.from_pydict({"a": [1, 0], "b": [0, 1]})).to_pydict()["prediction"]
    assert labels == [0, 1]


def test_rejects_no_features() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        MultinomialNB([], "y")
    with pytest.raises(PlanError, match="at least one feature"):
        BernoulliNB([], "y")


def test_names_a_missing_column(counts) -> None:
    _, _, _, ds = counts
    with pytest.raises(ColumnNotFoundError):
        MultinomialNB(["x0", "nope"], "y").fit(ds)


def test_predict_before_fit_raises() -> None:
    ds = bt.from_pydict({"x": [1], "y": [0]})
    with pytest.raises(PlanError, match="must be fitted"):
        BernoulliNB(["x"], "y").predict(ds)
