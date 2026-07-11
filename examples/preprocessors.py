"""Feature engineering with fit/transform preprocessor objects.

Builds a model-ready feature matrix from a raw customer table using the
scikit-learn-style preprocessors in ``batcher.ml.preprocessors``: impute missing
values, scale numerics, one-hot encode a category, bin a continuous column, encode the
target label, compose the lot with ``Chain``, and assemble a single tensor column with
``Concatenator``.

The point of a preprocessor being an *object* is that ``fit`` learns state (a median, a
mean, a category set, bin edges) that must be reused on held-out data: fit on train,
then ``transform`` — never ``fit_transform`` — the test split, so it inherits the
training statistics. Every step here does exactly that. ``fit`` runs one mergeable
aggregate in the engine; ``transform`` is a lazy ``Expr`` rewrite. No row is touched in
Python.

This is the preprocessor-object counterpart of ``examples/feature_engineering.py``,
which does the same workflow with raw expressions (broadcast aggregates, ``when/then``
bucketing, one-hot via boolean casts). The docs tutorial is
``docs/tutorials/feature-engineering.md``.

    python examples/preprocessors.py
"""

from __future__ import annotations

import batcher as bt
from batcher.ml.preprocessors import (
    Chain,
    Concatenator,
    KBinsDiscretizer,
    LabelEncoder,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def _raw() -> tuple[bt.Dataset, bt.Dataset]:
    """A tiny customer table, split into train and a held-out test set.

    The train set has a missing ``age``; the test set has a missing ``age`` and a
    ``plan`` value (``"student"``) never seen at fit time — both are cases a real
    feature pipeline must survive.
    """
    train = bt.from_pydict(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "age": [25.0, 40.0, None, 33.0, 52.0, 19.0],  # a null to impute
            "tenure": [2.0, 8.0, 5.0, 12.0, 20.0, 1.0],
            "plan": ["free", "pro", "free", "enterprise", "pro", "free"],
            "spend": [10.0, 55.0, 12.0, 90.0, 70.0, 5.0],
            "churned": ["yes", "no", "yes", "no", "no", "yes"],  # the target label
        }
    )
    test = bt.from_pydict(
        {
            "user_id": [7, 8],
            "age": [None, 45.0],
            "tenure": [3.0, 15.0],
            "plan": ["pro", "student"],  # "student" is unseen at fit time
            "spend": [20.0, 80.0],
            "churned": ["no", "yes"],
        }
    )
    return train, test


def _fit_on_train_not_test(train: bt.Dataset, test: bt.Dataset) -> None:
    """A scaler learns different statistics from train vs test — so only ever fit train."""
    on_train = StandardScaler(["tenure"]).fit(train)
    on_test = StandardScaler(["tenure"]).fit(test)
    assert on_train.mean_["tenure"] == 8.0
    assert on_test.mean_["tenure"] == 9.0  # fitting on test would leak its distribution


def _step_by_step(train: bt.Dataset, test: bt.Dataset) -> None:
    """Each preprocessor on its own: fit on train, transform both splits."""
    # Impute: the test null is filled with the *training* median, not its own.
    imputer = SimpleImputer(["age"], strategy="median").fit(train)
    assert imputer.statistics_ == {"age": 33.0}
    assert imputer.transform(train).to_pydict()["age"] == [25.0, 40.0, 33.0, 33.0, 52.0, 19.0]
    assert imputer.transform(test).to_pydict()["age"] == [33.0, 45.0]

    # Scale on the imputed data: the two imputed ages share one standardized value.
    imputed_train = imputer.transform(train)
    scaler = StandardScaler(["age", "tenure"]).fit(imputed_train)
    scaled = [round(v, 3) for v in scaler.transform(imputed_train).to_pydict()["age"]]
    assert scaled == [-0.822, 0.601, -0.063, -0.063, 1.738, -1.391]

    # One-hot: the unseen "student" plan encodes as all-zero indicators.
    encoder = OneHotEncoder(["plan"]).fit(train)
    assert encoder.categories_ == {"plan": ["enterprise", "free", "pro"]}
    enc_test = encoder.transform(test).to_pydict()
    assert enc_test["plan_enterprise"] == [0, 0]
    assert enc_test["plan_free"] == [0, 0]
    assert enc_test["plan_pro"] == [1, 0]  # row 8 ("student") is all zeros

    # Bin a continuous column into equal-width integer bins.
    binner = KBinsDiscretizer(["spend"], n_bins=3, strategy="uniform").fit(train)
    assert binner.transform(train).to_pydict()["spend"] == [0, 1, 0, 2, 2, 0]
    assert binner.transform(test).to_pydict()["spend"] == [0, 2]

    # Encode the target label to 0..k-1 in sorted class order.
    target = LabelEncoder("churned").fit(train)
    assert target.classes_ == ["no", "yes"]
    assert target.transform(train).to_pydict()["churned"] == [1, 0, 1, 0, 0, 1]


def _compose(train: bt.Dataset, test: bt.Dataset) -> tuple[bt.Dataset, bt.Dataset]:
    """Compose every step with ``Chain``: fit on train, transform both splits."""
    pipeline = Chain(
        SimpleImputer(["age"], strategy="median"),
        StandardScaler(["age", "tenure"]),
        KBinsDiscretizer(["spend"], n_bins=3, strategy="uniform"),
        OneHotEncoder(["plan"]),
        LabelEncoder("churned"),
    ).fit(train)

    assert len(pipeline) == 5
    assert repr(pipeline) == (
        "Chain(SimpleImputer, StandardScaler, KBinsDiscretizer, OneHotEncoder, LabelEncoder)"
    )

    train_features = pipeline.transform(train)
    test_features = pipeline.transform(test)

    # The one-hot step drops "plan" and appends its indicators; everything else stays.
    assert train_features.collect().column_names == [
        "user_id",
        "age",
        "tenure",
        "spend",
        "churned",
        "plan_enterprise",
        "plan_free",
        "plan_pro",
    ]

    # A fitted step stays introspectable, and held-out rows carry the training scale.
    assert round(pipeline[1].mean_["age"], 3) == 33.667
    assert [round(v, 3) for v in test_features.to_pydict()["age"]] == [-0.063, 1.075]
    # The unseen "student" plan is still all-zero after the whole chain.
    te = test_features.to_pydict()
    assert (te["plan_enterprise"][1], te["plan_free"][1], te["plan_pro"][1]) == (0, 0, 0)

    return train_features, test_features


def _assemble(train_features: bt.Dataset, test_features: bt.Dataset) -> None:
    """Stack the numeric feature columns into one tensor column for a training loop."""
    feature_cols = ["age", "tenure", "spend", "plan_enterprise", "plan_free", "plan_pro"]
    assembler = Concatenator(feature_cols, output_column="features", drop=True)

    model_ready = assembler.fit_transform(train_features)
    assert model_ready.collect().column_names == ["user_id", "churned", "features"]

    first = model_ready.to_pydict()["features"][0]
    assert [round(v, 3) for v in first] == [-0.822, -0.922, 0.0, 0.0, 1.0, 0.0]

    # The same assembler applies to the held-out matrix (stateless, so fit is a no-op).
    test_ready = assembler.transform(test_features)
    assert test_ready.collect().column_names == ["user_id", "churned", "features"]
    assert len(test_ready.to_pydict()["features"][0]) == len(feature_cols)

    # From here `model_ready.ml.iter_torch_batches(batch_size=..., columns=[...])` streams
    # {column: tensor} batches to a PyTorch loop (needs torch, so not run here).


def main() -> None:
    train, test = _raw()
    _fit_on_train_not_test(train, test)
    _step_by_step(train, test)
    train_features, test_features = _compose(train, test)
    _assemble(train_features, test_features)
    print("preprocessor feature pipeline OK")


if __name__ == "__main__":
    main()
