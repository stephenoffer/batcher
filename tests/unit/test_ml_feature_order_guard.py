"""The feature-order guard on `ds.ml.predict`, for models with and without recorded names.

A tabular model scores by position. Where a model records its training names, a reordering
or a trained name in another slot raises. Where it records none (fitted from a bare matrix),
the width it recorded is still checked, which used to be skipped entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from batcher._internal.errors import PlanError
from batcher.ml.tabular.registry import check_feature_names, get_adapter

sklearn_linear = pytest.importorskip("sklearn.linear_model")


class _Named:
    def __init__(self, names):
        self.names = names


class _NamesAdapter:
    def feature_names(self, model):
        return model.names


def test_a_model_without_names_still_has_its_width_checked():
    model = sklearn_linear.LinearRegression().fit(np.zeros((4, 3)), np.zeros(4))
    adapter = get_adapter("sklearn")
    assert adapter.feature_names(model) is None
    with pytest.raises(PlanError, match="trained on 3 features"):
        check_feature_names(adapter, model, ["a", "b"])
    check_feature_names(adapter, model, ["a", "b", "c"])


@pytest.mark.parametrize("framework", ["xgboost", "lightgbm"])
def test_a_booster_fitted_from_a_matrix_has_its_width_checked(framework):
    if framework == "xgboost":
        lib = pytest.importorskip("xgboost")
        model = lib.XGBRegressor(n_estimators=1).fit(np.random.rand(20, 3), np.random.rand(20))
    else:
        lib = pytest.importorskip("lightgbm")
        model = lib.LGBMRegressor(n_estimators=1, verbose=-1).fit(
            np.random.rand(20, 3), np.random.rand(20)
        )
    with pytest.raises(PlanError, match="trained on 3 features"):
        check_feature_names(get_adapter(framework), model, ["a", "b", "c", "d"])


def test_a_trained_name_in_another_slot_is_refused():
    with pytest.raises(PlanError, match=r"\['income'\] at a different position"):
        check_feature_names(_NamesAdapter(), _Named(["age", "income"]), ["income", "zip"])


def test_renamed_columns_in_the_trained_slots_pass():
    check_feature_names(_NamesAdapter(), _Named(["age", "income"]), ["age_years", "income"])


def test_generic_names_warn_that_the_order_cannot_be_verified():
    with pytest.warns(UserWarning, match="generic feature names"):
        check_feature_names(_NamesAdapter(), _Named(["f0", "f1"]), ["a", "b"])
