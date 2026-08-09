"""`SplineTransformer` and `FunctionTransformer`.

The spline basis is checked against SciPy's `BSpline` rather than against itself, because
"it looks like a bump" is not a property. The two structural invariants — partition of unity
and non-negativity — are checked over *every* row, which is what catches the endpoint: an
early version of this returned an all-zero row at the maximum value, and a spot check of the
first few rows saw nothing wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    FunctionTransformer,
    Preprocessor,
    SplineTransformer,
)

pytestmark = pytest.mark.unit

scipy_interpolate = pytest.importorskip("scipy.interpolate")

XS = [i / 4 for i in range(17)]


def _ds(values: list[float] | None = None) -> bt.Dataset:
    return bt.from_pydict({"x": XS if values is None else values})


def _basis(pre: SplineTransformer, ds: bt.Dataset) -> np.ndarray:
    out = pre.transform(ds).to_pydict()
    names = sorted((c for c in out if c.startswith("x_sp")), key=lambda c: int(c[4:]))
    return np.array([out[c] for c in names], dtype=float).T


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("n_knots", [3, 5, 8])
def test_the_basis_matches_scipy(degree: int, n_knots: int) -> None:
    ds = _ds()
    pre = SplineTransformer("x", n_knots=n_knots, degree=degree, knots="uniform").fit(ds)
    got = _basis(pre, ds)

    knots = np.asarray(pre.knots_["x"])
    padded = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    count = len(padded) - degree - 1
    identity = np.eye(count)
    want = np.nan_to_num(
        np.array(
            [
                scipy_interpolate.BSpline(padded, identity[i], degree, extrapolate=False)(XS)
                for i in range(count)
            ]
        ).T
    )
    # SciPy's basis is half-open at the right end, so the maximum value evaluates to NaN
    # there. Batcher closes that last interval on purpose, and the value it gives is the
    # left-hand limit — which is what this compares against.
    want[-1] = np.nan_to_num(
        np.array(
            [
                scipy_interpolate.BSpline(padded, identity[i], degree, extrapolate=False)(
                    XS[-1] - 1e-9
                )
                for i in range(count)
            ]
        )
    )
    np.testing.assert_allclose(got, want, atol=1e-7)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_the_basis_is_a_partition_of_unity_on_every_row(degree: int) -> None:
    """Every row, not the first few: the endpoint is exactly where this used to break."""
    ds = _ds()
    matrix = _basis(SplineTransformer("x", n_knots=5, degree=degree).fit(ds), ds)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(len(XS)))
    assert matrix.min() >= 0.0


def test_the_maximum_value_is_not_an_all_zero_row() -> None:
    ds = _ds()
    matrix = _basis(SplineTransformer("x", n_knots=4, degree=3).fit(ds), ds)
    assert matrix[-1].sum() == pytest.approx(1.0)


@pytest.mark.parametrize(("n_knots", "degree"), [(3, 1), (5, 3), (8, 2)])
def test_the_basis_has_n_knots_plus_degree_minus_one_columns(n_knots: int, degree: int) -> None:
    ds = _ds()
    out = SplineTransformer("x", n_knots=n_knots, degree=degree).fit_transform(ds)
    assert len([c for c in out.columns if c.startswith("x_sp")]) == n_knots + degree - 1


def test_uniform_knots_span_the_observed_range() -> None:
    assert SplineTransformer("x", n_knots=3, knots="uniform").fit(_ds()).knots_["x"] == [
        0.0,
        2.0,
        4.0,
    ]


def test_quantile_knots_follow_the_density_not_the_range() -> None:
    """A skewed column puts its interior knots where the data is, which uniform cannot."""
    skewed = [float(i) for i in range(10)] + [1000.0]
    quantile = SplineTransformer("x", n_knots=4, knots="quantile").fit(_ds(skewed)).knots_["x"]
    uniform = SplineTransformer("x", n_knots=4, knots="uniform").fit(_ds(skewed)).knots_["x"]
    assert quantile[0] == uniform[0] and quantile[-1] == uniform[-1]
    # Both interior knots sit inside the bulk, where uniform spacing puts neither.
    assert quantile[1] < uniform[1] and quantile[2] < uniform[2]


def test_repeated_quantiles_collapse_instead_of_making_collinear_columns() -> None:
    """A point mass repeats a knot, and repeated knots duplicate basis functions.

    Rank is checked on a dense grid rather than on the fitted column: that column has two
    distinct values, so *any* basis evaluated over it has rank 2 and the assertion would
    hold for a duplicated basis too.
    """
    ds = _ds([0.0] * 30 + [1.0])
    pre = SplineTransformer("x", n_knots=5, knots="quantile").fit(ds)
    assert pre.knots_["x"] == sorted(set(pre.knots_["x"]))
    matrix = _basis(pre, _ds([i / 20 for i in range(21)]))
    assert np.linalg.matrix_rank(matrix) == matrix.shape[1]


def test_a_constant_column_still_produces_a_usable_basis() -> None:
    ds = _ds([3.0] * 10)
    matrix = _basis(SplineTransformer("x", n_knots=3, degree=1).fit(ds), ds)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(10))


def test_drop_original_removes_the_source_column() -> None:
    out = SplineTransformer("x", n_knots=3, drop_original=True).fit_transform(_ds())
    assert "x" not in out.columns


def test_fit_is_independent_of_partitioning() -> None:
    one = SplineTransformer("x", n_knots=4, knots="uniform").fit(_ds())
    many = SplineTransformer("x", n_knots=4, knots="uniform").fit(_ds().repartition(4))
    assert one.knots_ == many.knots_


def test_an_all_null_column_says_so() -> None:
    ds = bt.from_pydict({"x": [None, None]})
    with pytest.raises(PlanError, match="no non-null values"):
        SplineTransformer("x", n_knots=3, knots="uniform").fit(ds)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_knots": 1}, "n_knots must be at least 2"),
        ({"degree": 0}, "degree must be at least 1"),
        ({"knots": "even"}, "knots must be"),
    ],
)
def test_spline_configuration_is_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        SplineTransformer("x", **kwargs)


def test_a_fitted_spline_round_trips_through_save(tmp_path) -> None:
    ds = _ds()
    fitted = SplineTransformer("x", n_knots=4, degree=2).fit(ds)
    target = str(tmp_path / "spline.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    assert restored.knots_ == fitted.knots_
    np.testing.assert_allclose(_basis(restored, ds), _basis(fitted, ds))


def test_spline_transform_streams() -> None:
    ds = _ds()
    pre = SplineTransformer("x", n_knots=4).fit(ds)
    collected = pre.transform(ds).to_pydict()["x_sp0"]
    streamed = [
        v
        for batch in pre.transform(ds).iter_batches(batch_size=4)
        for v in batch.to_pydict()["x_sp0"]
    ]
    assert collected == streamed


def test_function_transformer_replaces_a_column_in_place() -> None:
    ds = bt.from_pydict({"x": [1.0, 4.0, 9.0]})
    assert FunctionTransformer("x", lambda c: c.sqrt()).fit_transform(ds).to_pydict() == {
        "x": [1.0, 2.0, 3.0]
    }


def test_function_transformer_can_add_a_column_instead() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0]})
    out = FunctionTransformer("x", lambda c: c * 10, suffix="_scaled").fit_transform(ds)
    assert out.to_pydict() == {"x": [1.0, 2.0], "x_scaled": [10.0, 20.0]}


def test_function_transformer_applies_to_every_named_column() -> None:
    ds = bt.from_pydict({"a": [1.0], "b": [2.0]})
    out = FunctionTransformer(["a", "b"], lambda c: c + 1).fit_transform(ds)
    assert out.to_pydict() == {"a": [2.0], "b": [3.0]}


def test_function_transformer_rejects_a_scalar_style_function() -> None:
    """A lambda written against a value, not an Expr, is the obvious mistake to make."""
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(PlanError, match="not an expression"):
        FunctionTransformer("x", lambda c: 1.0).fit_transform(ds)


def test_function_transformer_rejects_a_non_callable() -> None:
    with pytest.raises(PlanError, match="needs a callable"):
        FunctionTransformer("x", "sqrt")


def test_both_compose_in_a_chain() -> None:
    from batcher.ml.preprocessors import Chain, StandardScaler

    ds = _ds()
    out = Chain(
        FunctionTransformer("x", lambda c: c + 1),
        SplineTransformer("x", n_knots=3, degree=1),
        StandardScaler(["x_sp0"]),
    ).fit_transform(ds)
    assert "x_sp0" in out.columns
    assert out.count() == len(XS)
