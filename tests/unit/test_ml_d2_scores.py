"""The D² regression scores on the absolute-error and pinball scales.

Both reproduce scikit-learn's `d2_absolute_error_score` and `d2_pinball_score` exactly, since
those are the functions users compare against, and both are pinned to the defining property: a
perfect fit scores 1 and the optimal-constant baseline scores 0.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.ml.metrics import d2_absolute_error_score, d2_pinball_score

pytestmark = pytest.mark.unit

sk_metrics = pytest.importorskip("sklearn.metrics")


@pytest.fixture(scope="module")
def scored() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    y = rng.normal(5, 2, 200)
    p = y + rng.normal(0, 1, 200)
    return y, p, bt.from_pydict({"y": y.tolist(), "p": p.tolist()})


def test_d2_absolute_error_matches_sklearn(scored) -> None:
    y, p, ds = scored
    assert d2_absolute_error_score(ds, "y", "p") == pytest.approx(
        sk_metrics.d2_absolute_error_score(y, p)
    )


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_d2_pinball_matches_sklearn(scored, alpha: float) -> None:
    y, p, ds = scored
    assert d2_pinball_score(ds, "y", "p", alpha=alpha) == pytest.approx(
        sk_metrics.d2_pinball_score(y, p, alpha=alpha)
    )


def test_d2_absolute_error_is_one_for_a_perfect_fit() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 4.0]})
    assert d2_absolute_error_score(ds, "y", "p") == pytest.approx(1.0)


def test_d2_absolute_error_is_zero_for_the_median_baseline() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    median = float(np.median(y))
    ds = bt.from_pydict({"y": y.tolist(), "p": [median] * 300})
    assert d2_absolute_error_score(ds, "y", "p") == pytest.approx(0.0, abs=1e-12)


def test_d2_pinball_reduces_to_absolute_error_at_the_median(scored) -> None:
    _, _, ds = scored
    assert d2_pinball_score(ds, "y", "p", alpha=0.5) == pytest.approx(
        d2_absolute_error_score(ds, "y", "p")
    )
