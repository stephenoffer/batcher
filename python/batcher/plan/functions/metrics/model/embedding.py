"""Embedding-quality metrics — scoring vector columns for retrieval, similarity, and drift.

An embedding pipeline produces fixed-width vector columns, and the questions you ask of them are
numeric: how similar is a query to its retrieved document, how far has today's embedding
distribution moved from training, are the vectors normalized the way the index expects. Each metric
here is a per-row vector operation that aggregates to a corpus score in one scan — the similarity or
distance is computed in Rust over the Arrow list, never materialized on the driver.

The pairwise metrics take two vector columns and report the mean over rows; the single-column
metrics report a distribution property (norm, unit-normalization rate) that catches a mis-scaled or
degenerate embedding before it poisons a similarity search.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "embedding_dim_drift",
    "mean_angular_distance",
    "mean_cosine_distance",
    "mean_cosine_similarity",
    "mean_dot_product",
    "mean_embedding_norm",
    "mean_euclidean_distance",
    "mean_hamming_distance",
    "mean_manhattan_distance",
    "unit_norm_rate",
    "zero_vector_rate",
]


def mean_cosine_similarity(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean cosine similarity between two vector columns, over the corpus.

    The headline retrieval-quality number: for a column of query embeddings and a column of the
    embeddings they retrieved (or of paraphrase pairs, or of a prediction against a reference
    vector), how aligned they are on average, in ``[-1, 1]``. A high mean means the pairs point the
    same way; a drop over time is the first sign an embedding model or its inputs have drifted.

    Args:
        left: A fixed-width numeric vector column.
        right: The paired vector column.

    Returns:
        The mean cosine similarity over the corpus, in ``[-1, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"a": [[1.0, 0.0], [1.0, 1.0]], "b": [[1.0, 0.0], [0.0, 1.0]]}
            ... )
            >>> round(ds.agg(m=bt.mean_cosine_similarity("a", "b")).to_pydict()["m"][0], 4)
            0.8536
    """
    return _as_column(left).list.cosine_similarity(_as_column(right)).mean()


def mean_euclidean_distance(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean Euclidean (L2) distance between two vector columns, over the corpus.

    The magnitude-sensitive counterpart of `mean_cosine_similarity`: it grows when vectors move
    apart in the embedding space, and unlike cosine it is affected by their norms. Use it when the
    vectors are not normalized and their scale is meaningful, or as a distance a threshold-based
    duplicate or near-match rule is set on.

    Args:
        left: A fixed-width numeric vector column.
        right: The paired vector column.

    Returns:
        The mean L2 distance over the corpus, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"a": [[0.0, 0.0], [3.0, 0.0]], "b": [[0.0, 0.0], [0.0, 4.0]]}
            ... )
            >>> ds.agg(m=bt.mean_euclidean_distance("a", "b")).to_pydict()["m"][0]
            2.5
    """
    return _as_column(left).list.euclidean_distance(_as_column(right)).mean()


def mean_dot_product(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean dot product between two vector columns, over the corpus.

    The raw inner product, which is the similarity a maximum-inner-product-search index actually
    ranks by. On unit-normalized vectors it equals the cosine similarity; on unnormalized ones it
    also rewards larger norms, which is sometimes the intended behavior (a longer, more confident
    document scoring higher) and sometimes a bug to catch with `unit_norm_rate`.

    Args:
        left: A fixed-width numeric vector column.
        right: The paired vector column.

    Returns:
        The mean dot product over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"a": [[1.0, 2.0], [3.0, 0.0]], "b": [[1.0, 0.0], [1.0, 1.0]]}
            ... )
            >>> ds.agg(m=bt.mean_dot_product("a", "b")).to_pydict()["m"][0]
            2.0
    """
    return _as_column(left).list.dot(_as_column(right)).mean()


def mean_embedding_norm(column: IntoExpr) -> Expr:
    """The mean L2 norm of a vector column — the average embedding magnitude.

    A distribution check on one embedding column: a mean norm far from 1 on vectors that should be
    normalized, or one that drifts across batches, points at a preprocessing or model problem before
    it shows up as bad retrieval. It is also the denominator that turns a dot product into a cosine.

    Args:
        column: A fixed-width numeric vector column.

    Returns:
        The mean L2 norm over the corpus, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [[3.0, 4.0], [1.0, 0.0]]})
            >>> ds.agg(m=bt.mean_embedding_norm("v")).to_pydict()["m"][0]
            3.0
    """
    return _as_column(column).list.l2_norm().mean()


def unit_norm_rate(column: IntoExpr) -> Expr:
    """The fraction of a vector column that is unit-normalized (L2 norm ~ 1).

    The precondition a cosine-similarity index assumes: if the vectors are not unit length, an
    inner-product search silently ranks by magnitude instead of angle. This is the one-number check
    that they are — a rate below 1 says some vectors slipped through un-normalized and the search is
    subtly wrong for them.

    Args:
        column: A fixed-width numeric vector column.

    Returns:
        The fraction of unit-norm vectors, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [[1.0, 0.0], [3.0, 4.0]]})
            >>> ds.agg(m=bt.unit_norm_rate("v")).to_pydict()["m"][0]
            0.5
    """
    return count_if(_as_column(column).list.is_unit_norm()) / count_if(lit(True))


def zero_vector_rate(column: IntoExpr) -> Expr:
    """The fraction of a vector column that is the zero vector — degenerate embeddings.

    A zero vector has no direction, so it has an undefined cosine similarity to everything and is a
    silent hole in a retrieval index. It usually means an empty input, a failed model call, or a
    padding row that leaked through. This rate surfaces them so they can be filtered before indexing
    rather than returning garbage neighbors.

    Args:
        column: A fixed-width numeric vector column.

    Returns:
        The fraction of zero vectors, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]})
            >>> round(ds.agg(m=bt.zero_vector_rate("v")).to_pydict()["m"][0], 4)
            0.6667
    """
    return count_if(_as_column(column).list.is_zero_vector()) / count_if(lit(True))


def mean_cosine_distance(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean cosine distance between two vector columns — ``1 - cosine_similarity``.

    The distance form of `mean_cosine_similarity`, in ``[0, 2]``, for when a monitor reads more
    naturally as "how far apart" than "how aligned": it rises as pairs diverge. It is the metric a
    cosine-space ANN index (the common default) ranks by, so it is the right drift number when the
    index is cosine.

    Args:
        left: A fixed-width numeric vector column.
        right: The paired vector column.

    Returns:
        The mean cosine distance over the corpus, in ``[0, 2]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"a": [[1.0, 0.0], [1.0, 0.0]], "b": [[1.0, 0.0], [0.0, 1.0]]}
            ... )
            >>> ds.agg(m=bt.mean_cosine_distance("a", "b")).to_pydict()["m"][0]
            0.5
    """
    return _as_column(left).list.cosine_distance(_as_column(right)).mean()


def mean_manhattan_distance(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean Manhattan (L1) distance between two vector columns, over the corpus.

    The sum of absolute per-dimension differences, averaged over rows. It is the distance a
    Manhattan-metric index uses, and it is more robust to a single large-magnitude dimension than L2
    so it is the better drift number when one embedding dimension can dominate the Euclidean sum.

    Args:
        left: A fixed-width numeric vector column.
        right: The paired vector column.

    Returns:
        The mean L1 distance over the corpus, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"a": [[0.0, 0.0], [1.0, 2.0]], "b": [[0.0, 0.0], [4.0, 6.0]]}
            ... )
            >>> ds.agg(m=bt.mean_manhattan_distance("a", "b")).to_pydict()["m"][0]
            3.5
    """
    return _as_column(left).list.l1_distance(_as_column(right)).mean()


def mean_angular_distance(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean angular distance between two vector columns — the normalized angle in ``[0, 1]``.

    The angle between the vectors, rescaled so orthogonal is a clean 0.5 and opposite is 1. Unlike
    cosine similarity, angular distance is a true metric (it satisfies the triangle inequality),
    which is why some ANN indexes build on it; use it when you want a proper distance rather than a
    similarity score.

    Args:
        left: A fixed-width numeric vector column.
        right: The paired vector column.

    Returns:
        The mean angular distance over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [[1.0, 0.0]], "b": [[0.0, 1.0]]})
            >>> round(ds.agg(m=bt.mean_angular_distance("a", "b")).to_pydict()["m"][0], 4)
            0.5
    """
    return _as_column(left).list.angular_distance(_as_column(right)).mean()


def mean_hamming_distance(left: IntoExpr, right: IntoExpr) -> Expr:
    """The mean Hamming distance between two vector columns — the count of differing positions.

    The number of dimensions where the two vectors differ, averaged over rows. It is the metric for
    *binary* or product-quantized embeddings, where a vector is a bit pattern and similarity is bit
    agreement, so it is the drift and duplicate signal for a compressed index, not a float one.

    Args:
        left: A fixed-width vector column, typically binary or quantized.
        right: The paired vector column.

    Returns:
        The mean Hamming distance over the corpus, at least 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [[1.0, 0.0, 1.0]], "b": [[1.0, 1.0, 0.0]]})
            >>> ds.agg(m=bt.mean_hamming_distance("a", "b")).to_pydict()["m"][0]
            2.0
    """
    return _as_column(left).list.hamming_distance(_as_column(right)).mean()


def embedding_dim_drift(column: IntoExpr, expected: int) -> Expr:
    """The fraction of vectors whose dimension is not `expected` — the index-corruption check.

    A vector index is built for one dimension. A column that mixes two is not a degraded index,
    it is a broken one: the mismatched rows either fail to insert or are silently dropped, and
    the queries that should have matched them return the next-nearest thing with a
    confident-looking distance. Nothing else in a pipeline notices, because both dimensions are
    valid embeddings.

    It happens for one reason above all others — a re-embed with a different model, or a mixed
    read across a corpus embedded in two passes. Run this on ingest, before the index build.

    A null row counts as wrong. A failed encode leaves a null where a vector should be, and it
    cannot enter the index any more than a mis-sized one can.

    Args:
        column: The embedding column, a list of numbers per row.
        expected: The dimension every vector must have.

    Returns:
        The wrong-dimension rate over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `expected` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"v": [[1.0, 2.0, 3.0], [1.0, 2.0], [0.0, 1.0, 0.0]]})
            >>> round(ds.agg(d=bt.embedding_dim_drift("v", 3)).to_pydict()["d"][0], 4)
            0.3333
    """
    from batcher._internal.errors import PlanError

    if expected < 1:
        raise PlanError(f"embedding_dim_drift: expected must be at least 1, got {expected}")
    dims = _as_column(column).list.len()
    # A null row counts as wrong, not as missing. A failed encode leaves a null where a vector
    # should be, and it cannot go into the index any more than a mis-sized one can — letting the
    # null fall out of the numerator would report a clean corpus for a column of failures.
    return count_if((dims != lit(expected)).fill_null(lit(True))) / count_if(lit(True))
