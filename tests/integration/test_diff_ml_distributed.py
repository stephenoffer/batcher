"""A fitted estimator must be the same estimator whether it was fitted on one node or many.

`tests/differential/test_diff_ml_sklearn.py` holds `batcher.ml` against scikit-learn on one
node. That is half the contract. The other half is the one CI could not see until there was
a Ray lane: a fit runs through `fit_aggregate`, which is one mergeable global aggregate, so
distributing it must change *where* the sums happen and nothing else.

If it did not, the failure would be silent and severe. A scaler fitted across eight workers
that disagreed with the same scaler fitted locally would produce a model that scores
correctly in development and wrongly in production, with no error anywhere.

The oracle is carried through: the distributed fit is compared to the single-node fit *and*
to sklearn, so "both paths agree with each other" cannot pass by both being wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset

pytest.importorskip("ray", reason="the distributed arm needs Ray")
sklearn = pytest.importorskip("sklearn", reason="scikit-learn is the ML oracle")

pytestmark = pytest.mark.integration

RTOL = 1e-9
ATOL = 1e-9
WORKERS = 2


@pytest.fixture()
def forced_distributed(monkeypatch):
    """Make every terminal `collect` inside a fit run distributed.

    A fit calls `ds.agg(...).collect()` with no execution arguments, so it resolves
    ``distributed="auto"`` — which on a single-node test cluster correctly declines to
    distribute. Forcing the flag is what lets the mergeable path actually execute here
    rather than being asserted about.
    """
    original = Dataset.collect

    def always_distributed(self, *args, **kwargs):
        kwargs.setdefault("distributed", True)
        kwargs.setdefault("num_workers", WORKERS)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "collect", always_distributed)


@pytest.fixture()
def frame():
    """Enough rows, in enough batches, that a distributed run has something to partition."""
    rng = np.random.default_rng(20260826)
    return {
        "a": rng.normal(3.0, 2.0, 4000).tolist(),
        "b": rng.exponential(5.0, 4000).tolist(),
        "g": rng.choice(["x", "y", "z", "w"], 4000).tolist(),
    }


def _matrix(data: dict, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(data[c], dtype=float) for c in columns])


def test_standard_scaler_fits_identically_distributed(frame, forced_distributed):
    """Mean and scale from a distributed fit must match the local fit and sklearn.

    This is the mergeable-algebra invariant applied to a statistical fit: partitioning the
    input changes the summation order and nothing else.
    """
    from sklearn.preprocessing import StandardScaler as SkScaler

    from batcher.ml import StandardScaler

    columns = ["a", "b"]
    ds = bt.from_pydict(frame)
    distributed = StandardScaler(columns=columns).fit(ds)
    oracle = SkScaler().fit(_matrix(frame, columns))

    # Float reassociation is the one stated exception to single-node == distributed, so the
    # comparison is to the last bits rather than exact — which is what the contract says and
    # is still far tighter than any formula error could hide under.
    np.testing.assert_allclose(
        [distributed.mean_[c] for c in columns], oracle.mean_, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        [distributed.scale_[c] for c in columns], oracle.scale_, rtol=1e-10, atol=1e-10
    )


def test_a_distributed_fit_equals_the_single_node_fit(frame, monkeypatch):
    """The two paths compared directly, with the same input and the same estimator."""
    from batcher.ml import StandardScaler

    columns = ["a", "b"]
    ds = bt.from_pydict(frame)
    local = StandardScaler(columns=columns).fit(ds)

    original = Dataset.collect
    monkeypatch.setattr(
        Dataset,
        "collect",
        lambda self, *a, **k: original(
            self, *a, **{**k, "distributed": True, "num_workers": WORKERS}
        ),
    )
    distributed = StandardScaler(columns=columns).fit(ds)

    for c in columns:
        assert distributed.mean_[c] == pytest.approx(local.mean_[c], rel=1e-10, abs=1e-10)
        assert distributed.scale_[c] == pytest.approx(local.scale_[c], rel=1e-10, abs=1e-10)


def test_an_encoder_learns_the_same_categories_distributed(frame, forced_distributed):
    """Category discovery is a distinct, which is also mergeable; order must not drift.

    An encoder whose category *order* depended on partition arrival would produce a
    different integer for the same string on a different cluster size — the worst kind of
    distributed defect, because every individual run looks self-consistent.
    """
    from sklearn.preprocessing import OrdinalEncoder as SkOrdinal

    from batcher.ml import OrdinalEncoder

    ds = bt.from_pydict(frame)
    fitted = OrdinalEncoder(columns=["g"]).fit(ds)
    encoded = np.asarray(fitted.transform(ds).to_pydict()["g"], dtype=float)

    oracle = SkOrdinal().fit(np.asarray(frame["g"], dtype=object).reshape(-1, 1))
    expected = oracle.transform(np.asarray(frame["g"], dtype=object).reshape(-1, 1)).ravel()

    np.testing.assert_allclose(encoded, expected)
