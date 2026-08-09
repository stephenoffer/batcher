"""Random projection and kernel approximation.

These are approximations, so the tests check the property each one claims rather than exact
values: a random projection must preserve pairwise distances, and a kernel map must make
dot products approximate the kernel it names. A test that only checked the output shape
would pass for a transform that returned noise.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.preprocessors import (
    GaussianRandomProjection,
    Nystroem,
    Preprocessor,
    RBFSampler,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

pytestmark = pytest.mark.unit

PROJECTIONS = [GaussianRandomProjection, SparseRandomProjection]


def _wide(rows: int = 60, width: int = 40, seed: int = 0) -> tuple[bt.Dataset, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, width))
    ds = bt.from_pydict({f"f{i}": matrix[:, i].tolist() for i in range(width)})
    return ds, matrix


def _projected(pre: Preprocessor, ds: bt.Dataset, prefix: str) -> np.ndarray:
    out = pre.transform(ds).to_pydict()
    names = sorted((c for c in out if c.startswith(prefix)), key=lambda c: int(c[len(prefix) :]))
    return np.array([out[c] for c in names], dtype=float).T


def _pairwise(matrix: np.ndarray) -> np.ndarray:
    difference = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt((difference**2).sum(axis=2))


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_a_random_projection_preserves_pairwise_distances(klass) -> None:
    """The Johnson-Lindenstrauss property, which is the whole reason to use these."""
    ds, matrix = _wide()
    pre = klass(list(ds.columns), n_components=256, seed=3).fit(ds)
    projected = _projected(pre, ds, "rp")

    original = _pairwise(matrix)
    after = _pairwise(projected)
    upper = np.triu_indices_from(original, k=1)
    ratio = after[upper] / original[upper]
    assert 0.75 < float(ratio.mean()) < 1.25
    assert float(ratio.std()) < 0.2


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_a_projection_is_reproducible_from_its_seed(klass) -> None:
    """Serving must reproduce training's feature space exactly, so the seed is the contract."""
    ds, _ = _wide(rows=5, width=6)
    first = klass(list(ds.columns), n_components=4, seed=11).fit(ds)
    second = klass(list(ds.columns), n_components=4, seed=11).fit(ds)
    assert first.components_ == second.components_
    other = klass(list(ds.columns), n_components=4, seed=12).fit(ds)
    assert other.components_ != first.components_


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_a_projection_reads_no_rows(klass) -> None:
    """Only the column names matter, so an empty frame still fits."""
    empty = bt.from_pydict({"a": [], "b": []})
    pre = klass(["a", "b"], n_components=3, seed=0).fit(empty)
    assert len(pre.components_) == 3


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_a_projection_emits_the_requested_width(klass) -> None:
    ds, _ = _wide(rows=8, width=5)
    out = klass(list(ds.columns), n_components=3, seed=0).fit_transform(ds)
    assert len([c for c in out.columns if c.startswith("rp")]) == 3


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_drop_original_removes_the_source_block(klass) -> None:
    ds, _ = _wide(rows=4, width=4)
    out = klass(list(ds.columns), n_components=2, seed=0, drop_original=True).fit_transform(ds)
    assert sorted(out.columns) == ["rp0", "rp1"]


def test_a_sparse_projection_really_is_sparse() -> None:
    """Zeroed entries are skipped when lowering, so sparsity is a cost saving, not a detail."""
    ds, _ = _wide(rows=4, width=100)
    pre = SparseRandomProjection(list(ds.columns), n_components=10, seed=0).fit(ds)
    zeros = sum(1 for row in pre.components_ for value in row if value == 0.0)
    assert zeros > 0.5 * 10 * 100


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_a_projection_too_wide_to_lower_is_refused(klass) -> None:
    ds, _ = _wide(rows=2, width=40)
    with pytest.raises(PlanError, match="multiply-add terms"):
        klass(list(ds.columns), n_components=100, max_terms=100).fit(ds)


@pytest.mark.parametrize("klass", PROJECTIONS)
def test_a_missing_column_is_named(klass) -> None:
    with pytest.raises(ColumnNotFoundError):
        klass(["nope"], n_components=2).fit(bt.from_pydict({"a": [1.0]}))


@pytest.mark.parametrize("klass", [*PROJECTIONS, RBFSampler, Nystroem])
def test_zero_components_is_refused(klass) -> None:
    with pytest.raises(PlanError, match="n_components must be at least 1"):
        klass(["a"], n_components=0)


def test_sparse_density_is_validated() -> None:
    with pytest.raises(PlanError, match="density must be in"):
        SparseRandomProjection(["a"], density=0.0)


def test_johnson_lindenstrauss_grows_with_precision_not_with_width() -> None:
    assert johnson_lindenstrauss_min_dim(10_000, eps=0.2) < johnson_lindenstrauss_min_dim(
        10_000, eps=0.1
    )
    assert johnson_lindenstrauss_min_dim(100, eps=0.1) < johnson_lindenstrauss_min_dim(
        100_000, eps=0.1
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"eps": 0.0}, "eps must be in"), ({"eps": 1.0}, "eps must be in")],
)
def test_johnson_lindenstrauss_validates_eps(kwargs: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        johnson_lindenstrauss_min_dim(100, **kwargs)


def test_johnson_lindenstrauss_validates_the_row_count() -> None:
    with pytest.raises(PlanError, match="n_samples must be positive"):
        johnson_lindenstrauss_min_dim(0)


def _kernel(matrix: np.ndarray, gamma: float) -> np.ndarray:
    squared = ((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * squared)


def test_rbf_sampler_dot_products_approximate_the_kernel() -> None:
    """The claim these make: <z(x), z(y)> ~ exp(-gamma ||x - y||^2)."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(30, 4))
    ds = bt.from_pydict({f"f{i}": matrix[:, i].tolist() for i in range(4)})
    gamma = 0.5
    mapped = _projected(
        RBFSampler(list(ds.columns), n_components=2000, gamma=gamma, seed=1).fit(ds), ds, "rbf"
    )
    approximated = mapped @ mapped.T
    exact = _kernel(matrix, gamma)
    assert float(np.abs(approximated - exact).mean()) < 0.05


def test_nystroem_dot_products_approximate_the_kernel() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(30, 3))
    ds = bt.from_pydict({f"f{i}": matrix[:, i].tolist() for i in range(3)})
    gamma = 0.5
    mapped = _projected(
        Nystroem(list(ds.columns), n_components=25, gamma=gamma, seed=1).fit(ds), ds, "ny"
    )
    approximated = mapped @ mapped.T
    exact = _kernel(matrix, gamma)
    assert float(np.abs(approximated - exact).mean()) < 0.05


def test_nystroem_needs_fewer_components_than_the_sampler() -> None:
    """The point of paying for a fit pass: landmarks sit where the data actually is."""
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 3))
    ds = bt.from_pydict({f"f{i}": matrix[:, i].tolist() for i in range(3)})
    gamma = 0.5
    exact = _kernel(matrix, gamma)

    def error(pre: Preprocessor, prefix: str) -> float:
        mapped = _projected(pre.fit(ds), ds, prefix)
        return float(np.abs(mapped @ mapped.T - exact).mean())

    landmarks = error(Nystroem(list(ds.columns), n_components=20, gamma=gamma, seed=1), "ny")
    fourier = error(RBFSampler(list(ds.columns), n_components=20, gamma=gamma, seed=1), "rbf")
    assert landmarks < fourier


@pytest.mark.parametrize("klass", [RBFSampler, Nystroem])
def test_a_kernel_map_is_reproducible_from_its_seed(klass) -> None:
    ds = bt.from_pydict({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]})
    prefix = "rbf" if klass is RBFSampler else "ny"
    first = _projected(klass(["a", "b"], n_components=3, seed=5).fit(ds), ds, prefix)
    second = _projected(klass(["a", "b"], n_components=3, seed=5).fit(ds), ds, prefix)
    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("klass", [RBFSampler, Nystroem])
def test_a_kernel_map_validates_gamma(klass) -> None:
    with pytest.raises(PlanError, match="gamma must be positive"):
        klass(["a"], gamma=0.0)


def test_nystroem_says_so_when_every_row_has_a_null() -> None:
    ds = bt.from_pydict({"a": [None, 1.0], "b": [1.0, None]})
    with pytest.raises(PlanError, match="no complete rows"):
        Nystroem(["a", "b"], n_components=2).fit(ds)


def test_nystroem_tolerates_duplicate_landmarks() -> None:
    """Duplicates make the landmark kernel singular; the eigenvalue floor is what saves it."""
    ds = bt.from_pydict({"a": [1.0] * 8, "b": [2.0] * 8})
    mapped = _projected(Nystroem(["a", "b"], n_components=4, seed=0).fit(ds), ds, "ny")
    assert np.isfinite(mapped).all()


@pytest.mark.parametrize(
    ("klass", "prefix"),
    [
        (GaussianRandomProjection, "rp"),
        (SparseRandomProjection, "rp"),
        (RBFSampler, "rbf"),
        (Nystroem, "ny"),
    ],
)
def test_a_fitted_map_round_trips_through_save(klass, prefix: str, tmp_path) -> None:
    ds = bt.from_pydict({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]})
    fitted = klass(["a", "b"], n_components=2, seed=0).fit(ds)
    target = str(tmp_path / "map.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    np.testing.assert_allclose(_projected(restored, ds, prefix), _projected(fitted, ds, prefix))


@pytest.mark.parametrize("klass", [*PROJECTIONS, RBFSampler, Nystroem])
def test_transform_before_fit_names_the_class(klass) -> None:
    with pytest.raises(PlanError, match="must be fitted"):
        klass(["a", "b"], n_components=2).transform(bt.from_pydict({"a": [1.0], "b": [2.0]}))


def test_projections_compose_in_a_chain() -> None:
    from batcher.ml.preprocessors import Chain, StandardScaler

    ds, _ = _wide(rows=20, width=8)
    out = Chain(
        StandardScaler(list(ds.columns)),
        GaussianRandomProjection(list(ds.columns), n_components=4, seed=0, drop_original=True),
    ).fit_transform(ds)
    assert sorted(out.columns) == ["rp0", "rp1", "rp2", "rp3"]
