"""Gradient-boosted-tree adapters — XGBoost, LightGBM, CatBoost.

The three libraries that dominate tabular ML. They agree on the important thing (score a
dense ``(n_rows, n_features)`` float matrix) and disagree on everything else: what a
prediction method is called, how to ask for raw margins or leaf indices, where the
training feature names live, and which knob caps the thread pool. Each adapter is the
translation of one library into the four questions `registry.TabularAdapter` asks.

Method vocabulary, deliberately uniform across all three so a pipeline can switch
frameworks without rewriting the call:

``predict``
    The model's natural output — a probability for a classifier, a value for a regressor.
``predict_proba``
    Class probabilities, one column per class.
``raw``
    The untransformed margin / raw score (pre-sigmoid, pre-softmax). What you want when
    the calibration or the ensembling happens downstream.
``leaf``
    The leaf index each tree routed the row to. The feature vector for a
    boosted-trees-plus-linear-model stack.
``contrib``
    Per-feature SHAP contributions, ``n_features + 1`` values per row (the trailing one is
    the bias). Explanation at batch scale, which is exactly where a row-at-a-time SHAP
    call is unusable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.optional import require
from batcher.ml.tabular.registry import BaseAdapter, register

if TYPE_CHECKING:
    import numpy as np

__all__ = ["CatBoostAdapter", "LightGBMAdapter", "XGBoostAdapter"]

_BOOSTER_METHODS = ("predict", "predict_proba", "raw", "leaf", "contrib")


class XGBoostAdapter(BaseAdapter):
    """Scores an ``xgboost`` ``Booster`` or scikit-learn wrapper (``XGBClassifier``, …)."""

    name = "xgboost"
    modules = ("xgboost",)
    methods = _BOOSTER_METHODS
    suffixes = ("ubj", "bst", "json", "model")

    def load(self, path: str) -> Any:
        """Load a saved XGBoost model (``.ubj``, ``.json``, or the legacy binary)."""
        xgb = require(
            "xgboost", feature="XGBoost batch inference", provides="xgboost", extra="xgboost"
        )
        booster = xgb.Booster()
        booster.load_model(path)
        return booster

    def feature_names(self, model: Any) -> list[str] | None:
        """The booster's recorded ``feature_names``, or None when it kept none."""
        names = getattr(model, "feature_names", None)
        if names is None and hasattr(model, "get_booster"):
            names = getattr(model.get_booster(), "feature_names", None)
        return list(names) if names else None

    def configure_threads(self, model: Any, threads: int) -> None:
        """Cap the booster's OpenMP thread pool (``nthread``) to `threads`."""
        try:
            if hasattr(model, "set_param"):
                model.set_param({"nthread": threads})
            elif hasattr(model, "set_params"):
                model.set_params(n_jobs=threads)
        except Exception:  # pragma: no cover - a wrapper without the knob is fine
            return

    def output_width(self, model: Any, method: str, n_features: int) -> int | None:
        """Derive the width from the booster's own ``num_class`` when it has no wrapper."""
        if hasattr(model, "classes_") or hasattr(model, "n_outputs_"):
            return super().output_width(model, method, n_features)
        classes = _xgboost_num_class(model)
        if method == "contrib":
            return (n_features + 1) * (classes if classes > 1 else 1)
        if method == "leaf":
            rounds = _xgboost_rounds(model)
            return None if rounds is None else rounds * (classes if classes > 1 else 1)
        if method in ("predict", "raw"):
            return classes if classes > 1 else 1
        return super().output_width(model, method, n_features)

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Score `matrix`, translating the uniform `method` to XGBoost's own keywords."""
        self._check_method(method)
        if method == "predict_proba" and not hasattr(model, "predict_proba"):
            raise PlanError(
                "method='predict_proba' needs an XGBClassifier; a bare Booster's predict() "
                "already returns the probability. Use method='predict'."
            )
        if hasattr(model, "get_booster") and method in ("predict", "predict_proba"):
            # A scikit-learn wrapper: its own predict applies the label transform (a
            # classifier returns *classes*, not the raw probability), which is what a
            # caller asking for `predict` on a fitted estimator means.
            if method == "predict_proba" and hasattr(model, "predict_proba"):
                return model.predict_proba(matrix)
            return model.predict(matrix)
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        xgb = require(
            "xgboost", feature="XGBoost batch inference", provides="xgboost", extra="xgboost"
        )
        dmatrix = xgb.DMatrix(matrix, missing=options.get("missing", float("nan")))
        kwargs: dict[str, Any] = {}
        if method == "raw":
            kwargs["output_margin"] = True
        elif method == "leaf":
            kwargs["pred_leaf"] = True
        elif method == "contrib":
            kwargs["pred_contribs"] = True
        rounds = options.get("iteration_range")
        if rounds is not None:
            kwargs["iteration_range"] = tuple(rounds)
        return booster.predict(dmatrix, **kwargs)


def _xgboost_num_class(booster: Any) -> int:
    """The booster's trained ``num_class`` (0 or 1 for a regressor / binary classifier)."""
    import json

    try:
        config = json.loads(booster.save_config())
        raw = config["learner"]["learner_model_param"].get("num_class", "0")
        return int(raw)
    except Exception:  # pragma: no cover - an older or non-standard booster
        return 0


def _xgboost_rounds(booster: Any) -> int | None:
    """The number of boosting rounds in the booster, or None when it cannot be read."""
    try:
        return int(booster.num_boosted_rounds())
    except Exception:  # pragma: no cover - an older booster without the accessor
        return None


class LightGBMAdapter(BaseAdapter):
    """Scores a ``lightgbm`` ``Booster`` or scikit-learn wrapper (``LGBMClassifier``, …)."""

    name = "lightgbm"
    modules = ("lightgbm",)
    methods = _BOOSTER_METHODS
    suffixes = ("txt", "lgb")

    def load(self, path: str) -> Any:
        """Load a LightGBM model from its text model file."""
        lgb = require(
            "lightgbm", feature="LightGBM batch inference", provides="lightgbm", extra="lightgbm"
        )
        return lgb.Booster(model_file=path)

    def feature_names(self, model: Any) -> list[str] | None:
        """The booster's ``feature_name()``, or None when they are LightGBM's generic ones."""
        booster = model.booster_ if hasattr(model, "booster_") else model
        try:
            names = list(booster.feature_name())
        except Exception:
            return None
        # LightGBM invents `Column_0…` when trained from a bare matrix; those are not real
        # names and comparing against them would raise on a correct pipeline.
        if all(n.startswith("Column_") for n in names):
            return None
        return names or None

    def configure_threads(self, model: Any, threads: int) -> None:
        """Cap LightGBM's thread pool (``num_threads``) to `threads`."""
        try:
            if hasattr(model, "params") and isinstance(model.params, dict):
                model.params["num_threads"] = threads
            elif hasattr(model, "set_params"):
                model.set_params(n_jobs=threads)
        except Exception:  # pragma: no cover - a wrapper without the knob is fine
            return

    def output_width(self, model: Any, method: str, n_features: int) -> int | None:
        """Derive the width from ``num_model_per_iteration`` for a bare ``Booster``."""
        if hasattr(model, "classes_") or hasattr(model, "n_outputs_"):
            return super().output_width(model, method, n_features)
        try:
            per_iter = int(model.num_model_per_iteration())
            trees = int(model.num_trees())
        except Exception:  # pragma: no cover - not a Booster after all
            return super().output_width(model, method, n_features)
        if method == "contrib":
            return (n_features + 1) * per_iter
        if method == "leaf":
            return trees
        if method in ("predict", "raw"):
            return per_iter
        return super().output_width(model, method, n_features)

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Score `matrix`, translating the uniform `method` to LightGBM's own keywords."""
        self._check_method(method)
        if method == "predict_proba" and not hasattr(model, "predict_proba"):
            raise PlanError(
                "method='predict_proba' needs an LGBMClassifier; a bare Booster's predict() "
                "already returns the probability. Use method='predict'."
            )
        if method in ("predict", "predict_proba") and hasattr(model, "booster_"):
            if method == "predict_proba" and hasattr(model, "predict_proba"):
                return model.predict_proba(matrix)
            return model.predict(matrix)
        booster = model.booster_ if hasattr(model, "booster_") else model
        kwargs: dict[str, Any] = {}
        if method == "raw":
            kwargs["raw_score"] = True
        elif method == "leaf":
            kwargs["pred_leaf"] = True
        elif method == "contrib":
            kwargs["pred_contrib"] = True
        rounds = options.get("num_iteration")
        if rounds is not None:
            kwargs["num_iteration"] = int(rounds)
        return booster.predict(matrix, **kwargs)


class CatBoostAdapter(BaseAdapter):
    """Scores a ``catboost`` model: ``CatBoostClassifier``, ``CatBoostRegressor``, ``CatBoost``."""

    name = "catboost"
    modules = ("catboost",)
    methods = _BOOSTER_METHODS
    suffixes = ("cbm",)

    def load(self, path: str) -> Any:
        """Load a CatBoost model from its ``.cbm`` binary."""
        catboost = require(
            "catboost", feature="CatBoost batch inference", provides="catboost", extra="catboost"
        )
        model = catboost.CatBoost()
        model.load_model(path)
        return model

    def feature_names(self, model: Any) -> list[str] | None:
        """The model's ``feature_names_``, or None when it recorded none."""
        names = getattr(model, "feature_names_", None)
        return list(names) if names else None

    def configure_threads(self, model: Any, threads: int) -> None:
        """CatBoost takes its thread count per prediction call, so nothing is set here."""
        _ = model, threads

    def predict(self, model: Any, matrix: np.ndarray, method: str, options: dict[str, Any]) -> Any:
        """Score `matrix`, translating the uniform `method` to CatBoost's ``prediction_type``."""
        self._check_method(method)
        threads = options.get("threads")
        kwargs: dict[str, Any] = {} if threads is None else {"thread_count": int(threads)}
        if method == "predict_proba":
            if not hasattr(model, "predict_proba"):
                raise PlanError(
                    "method='predict_proba' needs a CatBoostClassifier; this model has no "
                    "predict_proba. Use method='predict' or 'raw'."
                )
            return model.predict_proba(matrix, **kwargs)
        if method == "raw":
            return model.predict(matrix, prediction_type="RawFormulaVal", **kwargs)
        if method == "leaf":
            return model.calc_leaf_indexes(matrix)
        if method == "contrib":
            return model.get_feature_importance(
                data=_catboost_pool(matrix), type="ShapValues", **kwargs
            )
        return model.predict(matrix, **kwargs)


def _catboost_pool(matrix: np.ndarray) -> Any:
    """Wrap `matrix` in a CatBoost ``Pool``, which SHAP computation requires."""
    catboost = require(
        "catboost", feature="CatBoost batch inference", provides="catboost", extra="catboost"
    )
    return catboost.Pool(matrix)


register(XGBoostAdapter())
register(LightGBMAdapter())
register(CatBoostAdapter())
