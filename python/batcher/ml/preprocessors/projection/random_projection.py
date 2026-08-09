"""Random projection — cut a wide feature block down without looking at the data.

`PCA` finds the directions that carry the most variance, which costs a covariance pass and
an eigendecomposition. The Johnson-Lindenstrauss lemma says you often do not need them: a
*random* projection into enough dimensions preserves every pairwise distance to within a
small factor, with a probability that depends only on the target width and not on the data
at all.

That makes it the right tool in three places `PCA` is awkward. It needs no data pass, so it
works on a stream. Its matrix depends only on a seed and the input width, so training and
serving cannot disagree. And it costs nothing to fit on a block far too wide to build a
covariance matrix for.

Both projections lower to plain arithmetic over the source columns, so the transform is
column-wise and the JIT can compile it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir import Expr, col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "MAX_PROJECTION_TERMS",
    "GaussianRandomProjection",
    "SparseRandomProjection",
    "johnson_lindenstrauss_min_dim",
]

#: The ceiling on how many multiply-add terms a projection may lower to. Every term is
#: arithmetic the engine evaluates over every row, so ``n_features * n_components`` is both
#: the size of the expression tree and the per-row cost. 20,000 admits 200 features onto 100
#: components while still catching the accidental projection of a thousand-column frame.
MAX_PROJECTION_TERMS = 20_000


def johnson_lindenstrauss_min_dim(n_samples: int, *, eps: float = 0.1) -> int:
    """The width a random projection needs to hold every pairwise distance within `eps`.

    The Johnson-Lindenstrauss bound, ``4 ln(n) / (eps^2/2 - eps^3/3)``. Note what it does
    *not* mention: the number of input features. A million-column frame and a ten-column one
    need the same target width for the same number of rows, which is the whole reason random
    projection is worth using on very wide data.

    Args:
        n_samples: How many rows the distances must be preserved between.
        eps: The tolerated relative distortion, in ``(0, 1)``.

    Returns:
        The minimum number of components.

    Raises:
        PlanError: If `eps` is outside ``(0, 1)`` or `n_samples` is not positive.

    Examples:
        .. doctest::

            >>> from batcher.ml.preprocessors import johnson_lindenstrauss_min_dim
            >>> johnson_lindenstrauss_min_dim(10_000, eps=0.2)
            2125
    """
    import math

    if not 0.0 < eps < 1.0:
        raise PlanError(f"johnson_lindenstrauss_min_dim: eps must be in (0, 1), got {eps!r}")
    if n_samples < 1:
        raise PlanError(
            f"johnson_lindenstrauss_min_dim: n_samples must be positive, got {n_samples}"
        )
    denominator = eps**2 / 2.0 - eps**3 / 3.0
    return int(4.0 * math.log(n_samples) / denominator)


class _RandomProjection(Preprocessor):
    """The fit/transform machinery the two random projections share.

    Only the distribution the matrix is drawn from differs, so that is the one method a
    subclass supplies.
    """

    __slots__ = (
        "columns",
        "components_",
        "drop_original",
        "max_terms",
        "n_components",
        "output_prefix",
        "seed",
    )

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_components: int = 32,
        seed: int = 0,
        output_prefix: str = "rp",
        drop_original: bool = False,
        max_terms: int = MAX_PROJECTION_TERMS,
    ) -> None:
        what = type(self).__name__
        self.columns = columns_arg(columns, what=what)
        if n_components < 1:
            raise PlanError(f"{what}: n_components must be at least 1, got {n_components}")
        self.n_components = n_components
        self.seed = seed
        self.output_prefix = output_prefix
        self.drop_original = drop_original
        self.max_terms = max_terms
        self.components_: list[list[float]] = []

    def _draw(self, rows: int, columns: int) -> Any:
        """Draw the projection matrix, shaped ``(n_components, n_features)``."""
        raise NotImplementedError

    def fit(self, ds: Dataset) -> _RandomProjection:
        """Draw the projection matrix. No data is read — only the column count matters.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import GaussianRandomProjection
                >>> ds = bt.from_pydict({"a": [1.0], "b": [2.0], "c": [3.0]})
                >>> pre = GaussianRandomProjection(["a", "b", "c"], n_components=2).fit(ds)
                >>> len(pre.components_), len(pre.components_[0])
                (2, 3)

        Args:
            ds: The dataset, read only for its column names.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the projection would lower to more terms than `max_terms`.
            ColumnNotFoundError: If a named column is missing.
        """
        self._check_numeric(ds)
        available = ds.columns
        present = set(available)
        for name in self.columns:
            if name not in present:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, available, hint="Pass numeric columns.")
                )
        terms = self.n_components * len(self.columns)
        if terms > self.max_terms:
            raise PlanError(
                f"{type(self).__name__}: projecting {len(self.columns)} column(s) onto "
                f"{self.n_components} component(s) is {terms} multiply-add terms, above "
                f"max_terms={self.max_terms}. Every term is arithmetic evaluated over every "
                "row. Lower n_components, project fewer columns, or raise max_terms."
            )
        matrix = self._draw(self.n_components, len(self.columns))
        self.components_ = [[float(v) for v in row] for row in matrix]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append the projected columns ``<prefix>0 … <prefix>{n-1}``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import GaussianRandomProjection
                >>> ds = bt.from_pydict({"a": [1.0], "b": [2.0]})
                >>> out = GaussianRandomProjection(["a", "b"], n_components=2).fit_transform(ds)
                >>> [c for c in out.columns if c.startswith("rp")]
                ['rp0', 'rp1']

        Args:
            ds: The dataset to project.

        Returns:
            A new lazy `Dataset` with the projected columns appended.
        """
        self._require_fitted()
        projections: dict[str, Expr] = {}
        for index, weights in enumerate(self.components_):
            total = lit(0.0)
            for weight, name in zip(weights, self.columns, strict=True):
                if weight:  # a sparse projection zeroes most entries; skip the dead terms
                    total = total + lit(weight) * col(name).cast("float64")
            projections[f"{self.output_prefix}{index}"] = total
        out = ds.with_columns(**projections)
        return out.drop(*self.columns) if self.drop_original else out


class GaussianRandomProjection(_RandomProjection):
    """Project a block of numeric columns through a dense Gaussian random matrix.

    Entries are drawn from ``N(0, 1 / n_components)``, which is the scaling that makes the
    projection preserve squared distances in expectation rather than shrinking them.

    Use `johnson_lindenstrauss_min_dim` to pick `n_components` from the number of rows and
    the distortion you can accept, rather than guessing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import GaussianRandomProjection
            >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
            >>> out = GaussianRandomProjection(["a", "b", "c"], n_components=2, seed=1)
            >>> sorted(c for c in out.fit_transform(ds).columns if c.startswith("rp"))
            ['rp0', 'rp1']

    Args:
        columns: The numeric columns to project.
        n_components: The target width.
        seed: Seed for the matrix, so the projection is reproducible and serving matches
            training.
        output_prefix: The prefix of the emitted column names.
        drop_original: Remove the source columns after projecting them.
        max_terms: The ceiling on ``n_features * n_components``.
    """

    numeric_only = True

    __slots__ = ()

    def _draw(self, rows: int, columns: int) -> Any:
        """Draw a dense ``N(0, 1/n_components)`` matrix."""
        import numpy as np

        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, 1.0 / np.sqrt(rows), size=(rows, columns))


class SparseRandomProjection(_RandomProjection):
    """Project through a sparse random matrix, which is cheaper and nearly as good.

    Achlioptas's construction, generalized by Li: each entry is ``+s``, ``-s`` or zero, with
    the non-zero probability set by `density`. At the default density of
    ``1 / sqrt(n_features)`` most entries are zero, so the projection lowers to far fewer
    multiply-add terms than the Gaussian one while preserving distances to the same bound.

    Prefer this over `GaussianRandomProjection` on a wide block: the expression tree is what
    the engine evaluates per row, and this one is much smaller.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import SparseRandomProjection
            >>> ds = bt.from_pydict({f"f{i}": [float(i)] for i in range(9)})
            >>> pre = SparseRandomProjection(list(ds.columns), n_components=3, seed=0)
            >>> sorted(c for c in pre.fit_transform(ds).columns if c.startswith("rp"))
            ['rp0', 'rp1', 'rp2']

    Args:
        columns: The numeric columns to project.
        n_components: The target width.
        density: The share of non-zero entries, or ``None`` for ``1 / sqrt(n_features)``.
        seed: Seed for the matrix.
        output_prefix: The prefix of the emitted column names.
        drop_original: Remove the source columns after projecting them.
        max_terms: The ceiling on ``n_features * n_components``.
    """

    numeric_only = True

    __slots__ = ("density",)

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_components: int = 32,
        density: float | None = None,
        seed: int = 0,
        output_prefix: str = "rp",
        drop_original: bool = False,
        max_terms: int = MAX_PROJECTION_TERMS,
    ) -> None:
        super().__init__(
            columns,
            n_components=n_components,
            seed=seed,
            output_prefix=output_prefix,
            drop_original=drop_original,
            max_terms=max_terms,
        )
        if density is not None and not 0.0 < density <= 1.0:
            raise PlanError(f"SparseRandomProjection: density must be in (0, 1], got {density!r}")
        self.density = density

    def _draw(self, rows: int, columns: int) -> Any:
        """Draw the ``{-s, 0, +s}`` matrix at the configured density."""
        import numpy as np

        rng = np.random.default_rng(self.seed)
        density = self.density if self.density is not None else 1.0 / np.sqrt(columns)
        density = float(min(max(density, 1.0 / columns), 1.0))
        # The surviving entries are scaled by 1/sqrt(density * n_components) so the
        # projection has the same second moment as the dense Gaussian one.
        scale = 1.0 / np.sqrt(density * rows)
        draws = rng.random(size=(rows, columns))
        signs = rng.integers(0, 2, size=(rows, columns)) * 2 - 1
        return np.where(draws < density, signs * scale, 0.0)
