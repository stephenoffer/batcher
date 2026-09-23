"""A feature that separates the classes perfectly is the best one, not an undefined one.

`_anova_from_moments` returned NaN whenever the within-group sum of squares was zero, which
is exactly the perfectly separating case. NaN then sorted last, so `SelectKBest(f_classif)`
dropped the single most informative feature. sklearn scores it `inf`, and NaN only when the
feature is constant everywhere.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt
from batcher.ml import SelectKBest
from batcher.ml.feature_scores import f_classif_scores

pytestmark = pytest.mark.unit


def _data():
    return bt.from_pydict(
        {
            "perfect": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "noisy": [0.1, 0.9, 0.4, 0.6, 0.2, 0.8],
            "constant": [5.0] * 6,
            "y": [0, 0, 0, 1, 1, 1],
        }
    )


def test_a_perfectly_separating_feature_scores_inf_and_a_constant_one_nan():
    scores = f_classif_scores(_data(), "y", ["perfect", "noisy", "constant"])
    assert scores["perfect"] == math.inf
    assert math.isnan(scores["constant"])
    assert 0 < scores["noisy"] < math.inf


def test_select_k_best_keeps_the_separating_feature():
    fitted = SelectKBest("y", k=1, features=["perfect", "noisy", "constant"]).fit(_data())
    out = fitted.transform(_data()).columns
    assert "perfect" in out and "noisy" not in out
