"""Gaussian mixture models — soft clustering and density estimation by expectation-maximization.

Where k-means assigns each row hard to one cluster, a Gaussian mixture models the data as a blend
of Gaussians and gives each row a *probability* of belonging to each — which is what you want when
clusters overlap, have different shapes, or when the goal is a density rather than a partition. The
learned model scores how likely any point is, so it doubles as an anomaly detector.

It is fitted by expectation-maximization, and each half of each step maps onto the engine: the
E-step is a per-row expression (the responsibilities, normalized by a numerically-stable
log-sum-exp), and the M-step is a set of weighted aggregates (the mixing weights, means, and full
covariances). Only the small per-component matrix inverse and determinant run on the driver.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml._estimator import require_fitted
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.functions.horizontal import max_horizontal, sum_horizontal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["GaussianMixture"]


class GaussianMixture:
    """A mixture of `n_components` full-covariance Gaussians, fitted by expectation-maximization.

    Learns each component's mixing weight, mean, and full covariance so the data is modeled as
    their weighted sum. `predict` gives the most-likely component per row (soft clustering),
    `predict_proba` the membership probabilities, and `score_samples` the log-likelihood of each
    row — the last making the fitted model an anomaly detector, since a genuinely unusual point
    scores far below the rest.

    Because expectation-maximization only finds a local optimum, the initialization matters: the
    means are seeded from a reproducible content-hash sample of the rows, so a fit is deterministic
    from its seed. On well-separated data it recovers the components; where clusters overlap it
    gives the calibrated soft assignment that a hard clusterer cannot.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.mixture import GaussianMixture
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.2, 0.1, 9.0, 9.2, 9.1], "y": [0.0, 0.1, 0.2, 9.0, 9.1, 8.9]}
            ... )
            >>> gm = GaussianMixture(["x", "y"], n_components=2, seed=0).fit(ds)
            >>> labels = gm.predict(ds).to_pydict()["component"]
            >>> labels[0] == labels[1] and labels[3] == labels[4] and labels[0] != labels[3]
            True

    Args:
        columns: The numeric feature columns.
        n_components: How many Gaussian components to fit.
        max_iter: The maximum number of EM iterations.
        tol: The convergence tolerance on the mean log-likelihood between iterations.
        reg_covar: A ridge added to each covariance diagonal for numerical stability.
        output_column: The name of the component-label column `predict` appends.
        seed: Seed for the mean initialization.
    """

    __slots__ = (
        "columns",
        "converged_",
        "covariances_",
        "log_likelihood_",
        "max_iter",
        "means_",
        "n_components",
        "n_iter_",
        "output_column",
        "reg_covar",
        "seed",
        "tol",
        "weights_",
    )

    def __init__(
        self,
        columns: Sequence[str],
        *,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        output_column: str = "component",
        seed: int = 0,
    ) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("GaussianMixture needs at least one feature column.")
        if n_components < 1:
            raise PlanError(f"n_components must be at least 1, got {n_components}.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.output_column = output_column
        self.seed = seed
        self.weights_: list[float] = []
        self.means_: list[list[float]] = []
        self.covariances_: list = []
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = float("nan")

    def _component_log_density(self, mean, covariance):
        """The Gaussian log-density expression ``log N(x | mean, covariance)`` for one component."""
        import numpy as np

        d = len(self.columns)
        precision = np.linalg.inv(covariance)
        _, logdet = np.linalg.slogdet(covariance)
        constant = -0.5 * (d * math.log(2 * math.pi) + logdet)
        centered = [col(name) - lit(float(mean[i])) for i, name in enumerate(self.columns)]
        quadratic = lit(0.0)
        for i in range(d):
            for j in range(d):
                weight = float(precision[i, j])
                if weight != 0.0:
                    quadratic = quadratic + lit(weight) * centered[i] * centered[j]
        return lit(constant) - lit(0.5) * quadratic

    def _responsibility_columns(self, ds: Dataset) -> Dataset:
        """Append per-component log-weighted-density and normalized responsibility columns."""
        weighted = {
            f"__bt_lp{k}": lit(math.log(self.weights_[k]))
            + self._component_log_density(self.means_[k], self.covariances_[k])
            for k in range(self.n_components)
        }
        ds = ds.with_columns(**weighted)
        log_terms = [col(f"__bt_lp{k}") for k in range(self.n_components)]
        max_log = max_horizontal(*log_terms)
        ds = ds.with_columns(__bt_maxlog=max_log)
        exps = {
            f"__bt_e{k}": (col(f"__bt_lp{k}") - col("__bt_maxlog")).exp()
            for k in range(self.n_components)
        }
        ds = ds.with_columns(**exps)
        denom = sum_horizontal(*[col(f"__bt_e{k}") for k in range(self.n_components)])
        ds = ds.with_columns(__bt_denom=denom)
        # row log-likelihood = maxlog + log(sum exp) ; responsibility_k = e_k / denom
        responsibilities = {
            f"__bt_r{k}": col(f"__bt_e{k}") / col("__bt_denom") for k in range(self.n_components)
        }
        ds = ds.with_columns(
            __bt_ll=col("__bt_maxlog") + col("__bt_denom").ln(), **responsibilities
        )
        return ds

    def fit(self, ds: Dataset) -> GaussianMixture:
        """Fit the mixture by expectation-maximization to a local optimum.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.mixture import GaussianMixture
                >>> ds = bt.from_pydict({"x": [0.0, 0.1, 9.0, 9.1], "y": [0.0, 0.1, 9.0, 9.1]})
                >>> gm = GaussianMixture(["x", "y"], n_components=2, seed=0).fit(ds)
                >>> len(gm.means_)
                2

        Args:
            ds: The dataset to fit.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
            PlanError: If there are fewer rows than components.
        """

        from batcher.api.dataset._build import split_key
        from batcher.plan.functions.aggregate import mean as mean_
        from batcher.plan.functions.aggregate import sum as sum_

        for name in self.columns:
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        d = len(self.columns)
        n = ds.count()
        if n < self.n_components:
            raise PlanError(f"GaussianMixture needs at least {self.n_components} rows, got {n}.")
        sample = (
            ds.with_columns(__bt_hash=split_key(ds, None, self.seed))
            .sort("__bt_hash")
            .select(*self.columns)
            .limit(self.n_components)
            .to_pydict()
        )
        self.means_ = [
            [float(sample[name][k]) for name in self.columns] for k in range(self.n_components)
        ]
        global_cov = self._global_covariance(ds)
        self.covariances_ = [global_cov.copy() for _ in range(self.n_components)]
        self.weights_ = [1.0 / self.n_components] * self.n_components
        previous = -math.inf
        for iteration in range(self.max_iter):
            responsible = self._responsibility_columns(ds)
            aggregates: dict[str, object] = {
                "__bt_ll": mean_(col("__bt_ll")),
            }
            for k in range(self.n_components):
                aggregates[f"nk{k}"] = sum_(col(f"__bt_r{k}"))
                for i, name in enumerate(self.columns):
                    aggregates[f"m{k}_{i}"] = sum_(col(f"__bt_r{k}") * col(name))
            row = responsible.agg(**aggregates).collect()
            log_likelihood = float(row.column("__bt_ll")[0].as_py())
            counts = [float(row.column(f"nk{k}")[0].as_py()) for k in range(self.n_components)]
            self.weights_ = [max(c, 1e-12) / n for c in counts]
            self.means_ = [
                [
                    float(row.column(f"m{k}_{i}")[0].as_py()) / max(counts[k], 1e-12)
                    for i in range(d)
                ]
                for k in range(self.n_components)
            ]
            self.covariances_ = self._update_covariances(responsible, counts, d)
            self.n_iter_ = iteration + 1
            if abs(log_likelihood - previous) < self.tol:
                self.converged_ = True
                self.log_likelihood_ = log_likelihood
                break
            previous = log_likelihood
            self.log_likelihood_ = log_likelihood
        return self

    def _global_covariance(self, ds: Dataset):
        """The overall covariance of the feature block, used to seed every component."""
        import numpy as np

        from batcher.ml.stats.multivariate import covariance_matrix

        covariance = covariance_matrix(ds, self.columns).to_pydict()
        matrix = np.array([covariance[name] for name in self.columns], dtype=float).T
        return matrix + self.reg_covar * np.eye(len(self.columns))

    def _update_covariances(self, responsible: Dataset, counts, d: int):
        """The M-step weighted full covariance of each component."""
        import numpy as np

        from batcher.plan.functions.aggregate import sum as sum_

        centered = {
            k: [col(name) - lit(self.means_[k][i]) for i, name in enumerate(self.columns)]
            for k in range(self.n_components)
        }
        aggregates = {}
        for k in range(self.n_components):
            for i in range(d):
                for j in range(i, d):
                    aggregates[f"c{k}_{i}_{j}"] = sum_(
                        col(f"__bt_r{k}") * centered[k][i] * centered[k][j]
                    )
        row = responsible.agg(**aggregates).collect()
        covariances = []
        for k in range(self.n_components):
            matrix = np.zeros((d, d))
            for i in range(d):
                for j in range(i, d):
                    value = float(row.column(f"c{k}_{i}_{j}")[0].as_py()) / max(counts[k], 1e-12)
                    matrix[i, j] = matrix[j, i] = value
            covariances.append(matrix + self.reg_covar * np.eye(d))
        return covariances

    def predict(self, ds: Dataset) -> Dataset:
        """Append the most-likely component index for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.mixture import GaussianMixture
                >>> ds = bt.from_pydict({"x": [0.0, 0.1, 9.0, 9.1], "y": [0.0, 0.1, 9.0, 9.1]})
                >>> gm = GaussianMixture(["x", "y"], n_components=2, seed=0).fit(ds)
                >>> "component" in gm.predict(ds).columns
                True

        Args:
            ds: The dataset to label.

        Returns:
            A new lazy `Dataset` with the component-label column appended.
        """
        require_fitted(self, self.means_)
        scores = [
            lit(math.log(self.weights_[k]))
            + self._component_log_density(self.means_[k], self.covariances_[k])
            for k in range(self.n_components)
        ]
        prediction = lit(0)
        best = scores[0]
        for k in range(1, self.n_components):
            closer = scores[k] > best
            prediction = when(closer).then(lit(k)).otherwise(prediction)
            best = when(closer).then(scores[k]).otherwise(best)
        return ds.with_columns(**{self.output_column: prediction})

    def score_samples(self, ds: Dataset, *, output_column: str = "log_likelihood") -> Dataset:
        """Append each row's log-likelihood under the fitted mixture — the anomaly score.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.mixture import GaussianMixture
                >>> ds = bt.from_pydict({"x": [0.0, 0.1, 9.0, 9.1], "y": [0.0, 0.1, 9.0, 9.1]})
                >>> gm = GaussianMixture(["x", "y"], n_components=2, seed=0).fit(ds)
                >>> "log_likelihood" in gm.score_samples(ds).columns
                True

        Args:
            ds: The dataset to score.
            output_column: The name of the log-likelihood column to append.

        Returns:
            A new lazy `Dataset` with the per-row log-likelihood appended.
        """
        require_fitted(self, self.means_, "score_samples")
        log_terms = [
            lit(math.log(self.weights_[k]))
            + self._component_log_density(self.means_[k], self.covariances_[k])
            for k in range(self.n_components)
        ]
        max_log = max_horizontal(*log_terms)
        scored = ds.with_columns(__bt_maxlog=max_log)
        exps = sum_horizontal(*[(term - col("__bt_maxlog")).exp() for term in log_terms])
        return scored.with_columns(**{output_column: col("__bt_maxlog") + exps.ln()}).drop(
            "__bt_maxlog"
        )
