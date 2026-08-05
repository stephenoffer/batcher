"""The discriminants must fit in one pass, whatever the class count.

Both used to filter the frame per class and run a count, a mean aggregate and a covariance
over each — three scans per class, so a ten-class fit read the data thirty-one times, each
of them a filtered pass. On the distributed path every one of those is a pass across the
cluster.

A covariance is recoverable from sums and sums of products, which are ordinary mergeable
aggregates, so all of it now rides one `group_by(target)`. The pass count is pinned here;
the agreement tests beside it are what make the rewrite safe, since the shortcut form is
numerically different from the two-pass one and has to be shown to land in the same place.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset
from batcher.ml import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from batcher.ml.discriminant import class_moments

pytestmark = pytest.mark.unit

sklearn_da = pytest.importorskip("sklearn.discriminant_analysis")

MODELS = [LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis]


@pytest.fixture
def executions(monkeypatch) -> list[int]:
    tally = [0]
    for name in ("collect", "to_pydict"):
        original = getattr(Dataset, name)

        def counting(self, *args, _original=original, **kwargs):
            tally[0] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Dataset, name, counting)
    return tally


def _labelled(classes: int, rows: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(rows, 3))
    labels = rng.integers(0, classes, size=rows)
    features[labels == 0] += 3.0
    ds = bt.from_pydict(
        {f"f{i}": features[:, i].tolist() for i in range(3)} | {"y": labels.tolist()}
    )
    return ds, features, labels, [f"f{i}" for i in range(3)]


@pytest.mark.parametrize("klass", MODELS)
@pytest.mark.parametrize("classes", [2, 5, 10])
def test_the_fit_costs_one_pass_whatever_the_class_count(klass, classes: int, executions) -> None:
    ds, _, _, names = _labelled(classes)
    executions[0] = 0
    klass(names, "y").fit(ds)
    assert executions[0] == 1


@pytest.mark.parametrize("classes", [2, 3, 5])
def test_lda_still_matches_sklearn(classes: int) -> None:
    ds, features, labels, names = _labelled(classes)
    got = LinearDiscriminantAnalysis(names, "y").fit(ds).predict(ds).to_pydict()["prediction"]
    want = sklearn_da.LinearDiscriminantAnalysis().fit(features, labels).predict(features)
    assert got == want.tolist()


@pytest.mark.parametrize("classes", [2, 3, 5])
def test_qda_still_matches_sklearn(classes: int) -> None:
    ds, features, labels, names = _labelled(classes)
    got = QuadraticDiscriminantAnalysis(names, "y").fit(ds).predict(ds).to_pydict()["prediction"]
    want = sklearn_da.QuadraticDiscriminantAnalysis().fit(features, labels).predict(features)
    assert got == want.tolist()


def test_the_grouped_moments_equal_the_per_class_ones() -> None:
    """The shortcut covariance must land where a two-pass one does, not merely close by."""
    ds, features, labels, names = _labelled(3)
    classes, counts, means, covariances = class_moments(ds, names, "y")
    for label in classes:
        rows = features[labels == label]
        assert counts[label] == len(rows)
        np.testing.assert_allclose(means[label], rows.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(covariances[label], np.cov(rows, rowvar=False), rtol=1e-6)


def test_a_single_row_class_has_no_covariance() -> None:
    """A sample covariance divides by n-1, so one row cannot have one."""
    ds = bt.from_pydict(
        {"a": [0.0, 1.0, 5.0], "b": [0.0, 1.0, 5.0], "y": ["pair", "pair", "alone"]}
    )
    _, counts, _, covariances = class_moments(ds, ["a", "b"], "y")
    assert counts["alone"] == 1
    assert covariances["alone"] is None
    assert covariances["pair"] is not None


def test_qda_still_names_a_class_too_rare_to_fit() -> None:
    from batcher._internal.errors import PlanError

    ds = bt.from_pydict(
        {"a": [0.0, 1.0, 5.0], "b": [0.0, 1.0, 5.0], "y": ["pair", "pair", "alone"]}
    )
    with pytest.raises(PlanError, match="at least 2 examples per class"):
        QuadraticDiscriminantAnalysis(["a", "b"], "y").fit(ds)


def test_lda_tolerates_a_single_row_class() -> None:
    """It pools one covariance, so a class with one row contributes a mean and a prior only."""
    ds = bt.from_pydict(
        {"a": [0.0, 1.0, 0.5, 9.0], "b": [0.0, 1.0, 0.5, 9.0], "y": ["a", "a", "a", "b"]}
    )
    model = LinearDiscriminantAnalysis(["a", "b"], "y").fit(ds)
    assert set(model.classes_) == {"a", "b"}


@pytest.mark.parametrize("klass", MODELS)
def test_the_fit_is_independent_of_partitioning(klass) -> None:
    ds, _, _, names = _labelled(3)
    one = klass(names, "y").fit(ds)
    many = klass(names, "y").fit(ds.repartition(4))
    assert one.predict(ds).to_pydict() == many.predict(ds).to_pydict()
