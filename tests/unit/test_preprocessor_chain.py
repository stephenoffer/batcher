"""`Chain` fits each step on the previous step's output — and never on the test split.

The failure `Chain` exists to prevent is silent: fit a scaler on the *untransformed*
frame, or on the concatenation of train and test, and the features still compute, the
model still trains, and the reported accuracy is simply wrong. So the tests assert the
two things a hand-written loop gets wrong — the *threading* of each step's output into
the next `fit`, and the reuse of the train-fitted state on held-out data — by pinning
`Chain` against the correct manual sequence.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import Chain, MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler

pytestmark = pytest.mark.unit


@pytest.fixture
def train():
    return bt.from_pydict({"age": [10.0, 20.0, None, 40.0]})


@pytest.fixture
def holdout():
    # Deliberately different statistics from `train`: if a step refits here, the
    # numbers move and the equivalence assertions below fail.
    return bt.from_pydict({"age": [20.0, 220.0]})


def _manual(train_ds, target):
    """The correct hand-written sequence `Chain` encapsulates."""
    imputer = SimpleImputer(["age"]).fit(train_ds)
    scaler = StandardScaler(["age"]).fit(imputer.transform(train_ds))
    return scaler.transform(imputer.transform(target)).to_pydict()["age"]


def _chained(train_ds, target):
    chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"])).fit(train_ds)
    return chain.transform(target).to_pydict()["age"]


def test_chain_equals_the_correct_manual_sequence_on_train(train):
    assert _chained(train, train) == pytest.approx(_manual(train, train))


def test_chain_equals_the_correct_manual_sequence_on_holdout(train, holdout):
    """The anti-leak property: the holdout is scaled by the *train* statistics."""
    assert _chained(train, holdout) == pytest.approx(_manual(train, holdout))


def test_second_step_is_fitted_on_the_first_steps_output(train):
    """`StandardScaler` must learn from the *imputed* column, not the raw one.

    Mean-imputation would not separate the two cases (the imputed mean equals the raw
    mean by construction), so impute with a far-away constant: [10, 20, 1000, 40] has
    mean 267.5, while the raw non-null values have mean 23.3. Fitting on the first
    step's output puts the imputed row at +1.73 standard deviations; fitting on the raw
    column would put it at +78.3.
    """
    imputer = SimpleImputer(["age"], strategy="constant", fill_value=1000.0)
    out = Chain(imputer, StandardScaler(["age"])).fit_transform(train).to_pydict()["age"]
    assert out[2] == pytest.approx(1.7315, abs=1e-3)
    assert sum(out) / len(out) == pytest.approx(0.0, abs=1e-9)


def test_fit_transform_equals_fit_then_transform(train):
    a = Chain(SimpleImputer(["age"]), StandardScaler(["age"])).fit_transform(train)
    chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"])).fit(train)
    b = chain.transform(train)
    assert a.to_pydict() == pytest.approx(b.to_pydict())


def test_transform_stays_lazy(train):
    chain = Chain(SimpleImputer(["age"])).fit(train)
    assert isinstance(chain.transform(train), bt.Dataset)


def test_chain_nests(train):
    inner = Chain(SimpleImputer(["age"]))
    outer = Chain(inner, StandardScaler(["age"]))
    flat = Chain(SimpleImputer(["age"]), StandardScaler(["age"]))
    assert outer.fit_transform(train).to_pydict() == pytest.approx(
        flat.fit_transform(train).to_pydict()
    )


def test_chain_carries_a_multi_column_encoder(train):
    ds = bt.from_pydict({"age": [1.0, 2.0], "city": ["a", "b"]})
    out = Chain(MinMaxScaler(["age"]), OneHotEncoder(["city"])).fit_transform(ds).to_pydict()
    assert out["age"] == pytest.approx([0.0, 1.0])
    assert out["city_a"] == [1, 0]


def test_steps_are_introspectable(train):
    chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"]))
    assert len(chain) == 2
    assert isinstance(chain[0], SimpleImputer)
    assert [type(s).__name__ for s in chain] == ["SimpleImputer", "StandardScaler"]
    assert repr(chain) == "Chain(SimpleImputer, StandardScaler)"


def test_transform_before_fit_raises(train):
    with pytest.raises(PlanError, match="must be fitted"):
        Chain(StandardScaler(["age"])).transform(train)


def test_empty_chain_raises():
    with pytest.raises(PlanError, match="at least one preprocessor"):
        Chain()


def test_non_preprocessor_step_raises():
    with pytest.raises(PlanError, match="must be Preprocessor"):
        Chain(SimpleImputer(["age"]), "not-a-preprocessor")


def test_chain_is_itself_a_preprocessor():
    from batcher.ml import Preprocessor

    assert isinstance(Chain(SimpleImputer(["age"])), Preprocessor)


def test_a_fitted_chain_saves_and_loads(tmp_path):
    """Persisting a fitted *pipeline* is the thing users actually save, and it could not be.

    Two independent gaps made `to_dict(Chain(...))` impossible. The encoder had no case for a
    nested `Preprocessor`, so a chain hit "cannot serialize fitted state of type
    StandardScaler" — naming a step the user never asked about. And `get_params` reports
    `Chain`'s `*steps` by name, which is exactly how a var-positional parameter cannot be
    passed back, so reconstruction raised `TypeError` and was reported as a version mismatch
    that had not happened.
    """
    from batcher.ml import OutlierClipper, load, save

    train = bt.from_pydict({"age": [10.0, 20.0, None, 40.0], "score": [1.0, 5.0, 2.0, 90.0]})
    chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"]), OutlierClipper(["score"]))
    chain.fit(train)
    expected = chain.transform(train).to_pydict()

    path = str(tmp_path / "pipeline.json")
    save(chain, path)
    restored = load(path)

    assert isinstance(restored, Chain)
    assert restored.is_fitted
    assert [type(s).__name__ for s in restored.get_params()["steps"]] == [
        "SimpleImputer",
        "StandardScaler",
        "OutlierClipper",
    ]
    # The point of persisting it: the loaded pipeline must transform held-out data exactly as
    # the fitted one did. A step restored unfitted would still produce numbers, just wrong ones.
    assert restored.transform(train).to_pydict() == expected


def test_every_reachable_preprocessor_can_be_loaded_back():
    """The registry is built from `__all__` so a new preprocessor "cannot be forgotten".

    It could: it scanned only the `preprocessors` package, and `OutlierClipper` lives in
    `batcher.ml.outliers` beside the outlier functions. It was a `Preprocessor`, exported from
    `batcher.ml` like every other one, and the single one `save`/`load` could not reconstruct —
    discovered at load time, after the fitting was done.
    """
    import inspect

    from batcher import ml
    from batcher.ml import Preprocessor
    from batcher.ml.preprocessors.persistence import _registry

    known = _registry()
    reachable = [
        name
        for name in ml.__all__
        if inspect.isclass(getattr(ml, name))
        and issubclass(getattr(ml, name), Preprocessor)
        and getattr(ml, name) is not Preprocessor
    ]
    assert reachable, "the ml facade should export preprocessors"
    assert [n for n in reachable if n not in known] == []
