"""Dimensionality reduction and kernel approximation that need no covariance matrix.

`PCA` finds the directions carrying the most variance, which costs a covariance pass and an
eigendecomposition over the full width. The four transforms here get similar work done
without one, and all of them lower to plain arithmetic over the source columns, so the
transform is column-wise and the JIT can compile it.

`GaussianRandomProjection` / `SparseRandomProjection`
    Cut a wide block down by multiplying through a random matrix. The Johnson-Lindenstrauss
    lemma bounds the distortion using only the target width, not the input width or the
    data, so these need no data pass and work on a stream.
`RBFSampler` / `Nystroem`
    Map into a space where an ordinary dot product approximates an RBF kernel, so a linear
    model gets most of a kernel SVM's accuracy without its quadratic cost.
"""

from __future__ import annotations

from batcher.ml.preprocessors.projection.kernel import Nystroem, RBFSampler
from batcher.ml.preprocessors.projection.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
    johnson_lindenstrauss_min_dim,
)

__all__ = [
    "GaussianRandomProjection",
    "Nystroem",
    "RBFSampler",
    "SparseRandomProjection",
    "johnson_lindenstrauss_min_dim",
]
