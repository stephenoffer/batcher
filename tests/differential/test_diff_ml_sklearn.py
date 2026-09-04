"""`batcher.ml` against scikit-learn, the way relational operators are held against DuckDB.

`ml/` is 177 modules and roughly 48,000 lines — scalers, encoders, decompositions, linear
models, metrics — and `tests/differential/` held nothing for any of it. The relational
engine is safe because every operator is checked against an oracle on every change; the
statistical surface had no equivalent, so a drift in a variance formula or an encoder's
category ordering would have been found by a user.

scikit-learn is that oracle, and it is a fair one: these classes carry sklearn's names and
sklearn's fitted-attribute convention (`mean_`, `scale_`, `components_`), so agreement is
the stated contract rather than a coincidence to be argued about.

**The oracle lives here and never on an execution path.** sklearn is single-node NumPy with
no partial/combine/finalize form, so routing a fit through it would silently cap every
estimator at one machine. Batcher's own mergeable implementations stay; this file is what
makes them checkable. The distributed arm of the same contract is
`test_diff_ml_distributed.py`, which asserts one-node and many-node fits agree with each
other and with this oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt

sklearn = pytest.importorskip("sklearn", reason="scikit-learn is the ML oracle")

pytestmark = pytest.mark.differential

# Tolerances. Batcher accumulates in a mergeable form (Neumaier-compensated sums, Chan's
# parallel variance) and sklearn does not, so the two agree to near the last bits rather
# than exactly. These are tight enough that a formula error cannot hide under them.
RTOL = 1e-9
ATOL = 1e-9


@pytest.fixture()
def numeric():
    """A small numeric frame with a deliberately non-uniform spread per column."""
    rng = np.random.default_rng(20260826)
    a = rng.normal(3.0, 2.0, 64)
    b = rng.exponential(5.0, 64)
    c = rng.integers(-50, 50, 64).astype(float)
    return {"a": a.tolist(), "b": b.tolist(), "c": c.tolist()}


def _matrix(data: dict, columns: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(data[c], dtype=float) for c in columns])


def _transformed(ds, columns: list[str]) -> np.ndarray:
    out = ds.to_pydict()
    return np.column_stack([np.asarray(out[c], dtype=float) for c in columns])


# --- scalers ---------------------------------------------------------------------------


def test_standard_scaler_matches_sklearn(numeric):
    """Both the learned mean/scale and the transformed values must agree."""
    from sklearn.preprocessing import StandardScaler as SkScaler

    from batcher.ml import StandardScaler

    columns = ["a", "b", "c"]
    ds = bt.from_pydict(numeric)
    fitted = StandardScaler(columns=columns).fit(ds)
    oracle = SkScaler().fit(_matrix(numeric, columns))

    np.testing.assert_allclose(
        [fitted.mean_[c] for c in columns], oracle.mean_, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        [fitted.scale_[c] for c in columns], oracle.scale_, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        _transformed(fitted.transform(ds), columns),
        oracle.transform(_matrix(numeric, columns)),
        rtol=RTOL,
        atol=ATOL,
    )


def test_standard_scaler_uses_the_population_denominator(numeric):
    """The n-vs-n-1 choice, pinned explicitly because it is invisible at scale.

    sklearn divides by `n`. Dividing by `n - 1` changes the scale by a factor that vanishes
    as rows grow, so a 64-row test catches it and a million-row one would not. This is the
    single most likely silent divergence in a variance implementation.
    """
    from batcher.ml import StandardScaler

    columns = ["a"]
    fitted = StandardScaler(columns=columns).fit(bt.from_pydict(numeric))
    values = np.asarray(numeric["a"], dtype=float)

    population = float(np.sqrt(((values - values.mean()) ** 2).sum() / values.size))
    sample = float(np.sqrt(((values - values.mean()) ** 2).sum() / (values.size - 1)))

    assert fitted.scale_["a"] == pytest.approx(population, rel=RTOL)
    assert fitted.scale_["a"] != pytest.approx(sample, rel=RTOL)


def test_min_max_scaler_matches_sklearn(numeric):
    from sklearn.preprocessing import MinMaxScaler as SkMinMax

    from batcher.ml import MinMaxScaler

    columns = ["a", "b", "c"]
    ds = bt.from_pydict(numeric)
    fitted = MinMaxScaler(columns=columns).fit(ds)
    oracle = SkMinMax().fit(_matrix(numeric, columns))

    np.testing.assert_allclose(
        _transformed(fitted.transform(ds), columns),
        oracle.transform(_matrix(numeric, columns)),
        rtol=RTOL,
        atol=ATOL,
    )


def test_max_abs_scaler_matches_sklearn(numeric):
    from sklearn.preprocessing import MaxAbsScaler as SkMaxAbs

    from batcher.ml import MaxAbsScaler

    columns = ["a", "b", "c"]
    ds = bt.from_pydict(numeric)
    fitted = MaxAbsScaler(columns=columns).fit(ds)
    oracle = SkMaxAbs().fit(_matrix(numeric, columns))

    np.testing.assert_allclose(
        _transformed(fitted.transform(ds), columns),
        oracle.transform(_matrix(numeric, columns)),
        rtol=RTOL,
        atol=ATOL,
    )


def test_a_constant_column_does_not_divide_by_zero():
    """sklearn scales a zero-variance column by 1 rather than by 0. So must Batcher.

    The edge case every scaler gets wrong once, and it produces inf or nan rather than an
    error, so it travels a long way downstream before anyone notices.
    """
    from sklearn.preprocessing import StandardScaler as SkScaler

    from batcher.ml import StandardScaler

    data = {"k": [7.0] * 16}
    ds = bt.from_pydict(data)
    fitted = StandardScaler(columns=["k"]).fit(ds)
    oracle = SkScaler().fit(_matrix(data, ["k"]))

    assert fitted.scale_["k"] == pytest.approx(float(oracle.scale_[0]))
    got = _transformed(fitted.transform(ds), ["k"])
    assert np.isfinite(got).all(), "a constant column must not transform to inf or nan"
    np.testing.assert_allclose(got, oracle.transform(_matrix(data, ["k"])), atol=ATOL)


# --- encoders --------------------------------------------------------------------------


def test_ordinal_encoder_matches_sklearn_category_order():
    """Categories are ordered lexicographically by sklearn; a different order is a bug.

    An encoder that ordered by first appearance would produce a perfectly reasonable-looking
    result that disagrees with every model trained elsewhere on the same data.
    """
    from sklearn.preprocessing import OrdinalEncoder as SkOrdinal

    from batcher.ml import OrdinalEncoder

    data = {"g": ["delta", "alpha", "charlie", "alpha", "bravo", "delta"]}
    ds = bt.from_pydict(data)
    fitted = OrdinalEncoder(columns=["g"]).fit(ds)
    oracle = SkOrdinal().fit(np.asarray(data["g"], dtype=object).reshape(-1, 1))

    got = np.asarray(fitted.transform(ds).to_pydict()["g"], dtype=float).reshape(-1, 1)
    np.testing.assert_allclose(
        got, oracle.transform(np.asarray(data["g"], dtype=object).reshape(-1, 1))
    )


def test_one_hot_encoder_matches_sklearn():
    """The same categories, in the same order, producing the same indicator columns."""
    from sklearn.preprocessing import OneHotEncoder as SkOneHot

    from batcher.ml import OneHotEncoder

    data = {"g": ["y", "x", "z", "x", "y"]}
    ds = bt.from_pydict(data)
    fitted = OneHotEncoder(columns=["g"]).fit(ds)
    out = fitted.transform(ds).to_pydict()

    oracle = SkOneHot(sparse_output=False).fit(np.asarray(data["g"], dtype=object).reshape(-1, 1))
    expected = oracle.transform(np.asarray(data["g"], dtype=object).reshape(-1, 1))
    categories = [str(c) for c in oracle.categories_[0]]

    produced = [name for name in out if name != "g"]
    assert len(produced) == len(categories), (
        f"expected one indicator per category {categories}, produced {produced}"
    )
    # Match by category suffix rather than by an assumed naming scheme: the contract is the
    # encoding, not the column name.
    for index, category in enumerate(categories):
        column = next(name for name in produced if name.endswith(category))
        np.testing.assert_allclose(
            np.asarray(out[column], dtype=float), expected[:, index], atol=ATOL
        )


# --- decomposition ---------------------------------------------------------------------


def test_pca_matches_sklearn_up_to_component_sign(numeric):
    """PCA agrees on explained variance exactly, and on components up to sign.

    A singular vector and its negation are both valid, and the sign is an artefact of the
    solver, so comparing components directly would be a test of LAPACK's tie-breaking. The
    explained variance carries no such freedom and is compared exactly.
    """
    from sklearn.decomposition import PCA as SkPCA

    from batcher.ml import PCA

    columns = ["a", "b", "c"]
    ds = bt.from_pydict(numeric)
    fitted = PCA(columns=columns, n_components=2).fit(ds)
    oracle = SkPCA(n_components=2).fit(_matrix(numeric, columns))

    np.testing.assert_allclose(
        np.asarray(fitted.explained_variance_ratio_, dtype=float),
        oracle.explained_variance_ratio_,
        rtol=1e-6,
        atol=1e-9,
    )
    got = np.abs(np.asarray(fitted.components_, dtype=float))
    np.testing.assert_allclose(got, np.abs(oracle.components_), rtol=1e-6, atol=1e-8)


# --- linear models ---------------------------------------------------------------------


def test_linear_regression_matches_sklearn(numeric):
    """Coefficients and intercept from the same normal equations."""
    from sklearn.linear_model import LinearRegression as SkLinear

    from batcher.ml import LinearRegression

    rng = np.random.default_rng(7)
    target = (
        2.5 * np.asarray(numeric["a"]) - 0.75 * np.asarray(numeric["b"]) + rng.normal(0, 0.1, 64)
    )
    data = {**numeric, "y": target.tolist()}
    ds = bt.from_pydict(data)

    fitted = LinearRegression(features=["a", "b", "c"], target="y").fit(ds)
    oracle = SkLinear().fit(_matrix(data, ["a", "b", "c"]), target)

    np.testing.assert_allclose(
        np.asarray(fitted.coef_, dtype=float).ravel(), oracle.coef_, rtol=1e-6, atol=1e-8
    )
    assert float(np.ravel(fitted.intercept_)[0]) == pytest.approx(
        float(oracle.intercept_), rel=1e-6, abs=1e-8
    )

    # And the predictions, not only the parameters: a correct fit paired with a wrong
    # prediction expression is a shape this test would otherwise pass straight over.
    predicted = np.asarray(fitted.predict(ds).to_pydict()["prediction"], dtype=float)
    np.testing.assert_allclose(
        predicted, oracle.predict(_matrix(data, ["a", "b", "c"])), rtol=1e-6, atol=1e-7
    )
