"""Maximal marginal relevance — reranking a candidate set for diversity, not just relevance.

A vector search returns the `k` passages closest to the query, and on a real corpus several of
them are the same passage. Documents get republished, chunks overlap by design, and a boilerplate
paragraph matches everything. The context window then holds one fact repeated four times, and the
model reads that repetition as emphasis.

Maximal marginal relevance is the standard fix. It builds the result greedily: at each step it
takes the candidate maximizing

    lambda * relevance(query, candidate) - (1 - lambda) * max_similarity(candidate, already_chosen)

so a candidate that duplicates something already selected is penalized however relevant it is.
`lambda_mult` is the dial: 1.0 is pure relevance (the ranking you already had), 0.0 is pure
diversity, and the useful range is 0.5 to 0.8.

The selection is greedy and sequential per query, which cannot be an `Expr`. It is a batch UDF
instead, vectorized with NumPy over the whole batch's flattened embedding buffer: the similarity
matrix for one query's candidates is one matrix multiply, and the greedy loop runs over `k`
selections, not over rows of data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa

__all__ = ["mmr_rerank_udf"]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length, leaving a zero row as zeros rather than dividing by zero."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def _select(candidates: np.ndarray, relevance: np.ndarray, k: int, lam: float) -> list[int]:
    """The greedy MMR selection over one query's candidates, as indices into them.

    `candidates` is already unit-normalized, so a dot product is the cosine similarity and the
    whole pairwise matrix is one multiply.
    """
    import numpy as np

    n = len(relevance)
    if n == 0:
        return []
    similarity = candidates @ candidates.T
    chosen: list[int] = [int(np.argmax(relevance))]
    # The running max similarity to anything already chosen, updated per selection rather
    # than recomputed — the difference between O(k·n) and O(k²·n) on a wide candidate set.
    redundancy = similarity[chosen[0]].copy()
    for _ in range(min(k, n) - 1):
        score = lam * relevance - (1.0 - lam) * redundancy
        score[chosen] = -np.inf  # never pick the same candidate twice
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break  # every candidate is already chosen
        chosen.append(best)
        redundancy = np.maximum(redundancy, similarity[best])
    return chosen


def mmr_rerank_udf(
    *,
    embedding_column: str,
    score_column: str | None = None,
    rerank_columns: tuple[str, ...] = (),
    k: int = 5,
    lambda_mult: float = 0.7,
) -> type:
    """A class UDF that reranks each row's candidate list for relevance *and* diversity.

    Every column named in `rerank_columns`, plus `embedding_column` and `score_column`, must be a
    list column holding one entry per candidate for that row — the shape a vector search leaves
    behind. Each is rewritten to the selected candidates, in selection order, so the passages,
    their ids, and their scores stay aligned.

    Similarity between candidates is cosine, computed on normalized copies of the embeddings, so
    an unnormalized index works without a separate normalization pass. When `score_column` is
    omitted the candidates' existing order is taken as the relevance ranking, which is what a
    search result already gives you.

    Args:
        embedding_column: A list-of-list column of candidate embeddings, one list per candidate.
        score_column: A list column of relevance scores, higher being better. Omit to use the
            candidates' existing order.
        rerank_columns: Other per-candidate list columns to reorder alongside, such as the
            passage text and its id.
        k: Candidates to keep per row. A row with fewer keeps all of them.
        lambda_mult: The relevance/diversity trade-off in ``[0, 1]``. 1.0 is pure relevance,
            0.0 pure diversity; 0.5 to 0.8 is the useful range.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch with each listed
        column reduced to the selected candidates.

    Raises:
        PlanError: If `k` is below 1, or `lambda_mult` is outside ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import mmr_rerank_udf
            >>> # Two near-identical candidates and one different; k=2 must not pick both twins.
            >>> ds = bt.from_pydict(
            ...     {
            ...         "vecs": [[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]],
            ...         "docs": [["twin a", "twin b", "different"]],
            ...         "scores": [[0.9, 0.89, 0.5]],
            ...     }
            ... )
            >>> udf = mmr_rerank_udf(
            ...     embedding_column="vecs",
            ...     score_column="scores",
            ...     rerank_columns=("docs",),
            ...     k=2,
            ... )
            >>> ds.ml.map_batches(udf).to_pydict()["docs"]
            [['twin a', 'different']]
    """
    if k < 1:
        raise PlanError(f"mmr_rerank_udf: k must be at least 1, got {k}")
    if not 0.0 <= lambda_mult <= 1.0:
        raise PlanError(f"mmr_rerank_udf: lambda_mult must be in [0, 1], got {lambda_mult}")

    class _MmrRerank:
        """Stateless, but a class so it matches the load-once UDF shape used elsewhere."""

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return _rerank_batch(
                batch,
                embedding_column=embedding_column,
                score_column=score_column,
                rerank_columns=rerank_columns,
                k=k,
                lambda_mult=lambda_mult,
            )

    return _MmrRerank


def _require(batch: pa.RecordBatch, name: str) -> None:
    """Fail naming the columns the batch does have, rather than with an index error."""
    if name not in batch.schema.names:
        from batcher._internal.errors import ColumnNotFoundError, unknown_message

        raise ColumnNotFoundError(
            unknown_message("column", name, list(batch.schema.names), hint="Pass a list column.")
        )


def _rerank_batch(
    batch: pa.RecordBatch,
    *,
    embedding_column: str,
    score_column: str | None,
    rerank_columns: tuple[str, ...],
    k: int,
    lambda_mult: float,
) -> pa.RecordBatch:
    """One batch reranked: pick per row, then rebuild every per-candidate column."""
    import numpy as np
    import pyarrow as pa

    for name in (embedding_column, *rerank_columns, *([score_column] if score_column else [])):
        _require(batch, name)

    embeddings = batch.column(batch.schema.get_field_index(embedding_column)).to_pylist()
    scores = (
        batch.column(batch.schema.get_field_index(score_column)).to_pylist()
        if score_column
        else [None] * batch.num_rows
    )

    picks: list[list[int]] = []
    for row_vectors, row_scores in zip(embeddings, scores, strict=True):
        if not row_vectors:
            picks.append([])
            continue
        matrix = _unit_rows(np.asarray(row_vectors, dtype=np.float64))
        if row_scores is None:
            # No score column: the candidates' existing order is the ranking, so a
            # descending sequence reproduces it exactly.
            relevance = np.arange(len(row_vectors), 0, -1, dtype=np.float64)
        else:
            relevance = np.asarray(
                [0.0 if s is None else float(s) for s in row_scores], dtype=np.float64
            )
        picks.append(_select(matrix, relevance, k, lambda_mult))

    # Every per-candidate column has to be reduced by the same picks; anything else rides
    # through untouched, so a query id or a tenant tag survives the rerank.
    per_candidate = {embedding_column, *rerank_columns}
    if score_column:
        per_candidate.add(score_column)
    arrays = []
    for index, name in enumerate(batch.schema.names):
        column = batch.column(index)
        if name in per_candidate:
            values = column.to_pylist()
            arrays.append(
                pa.array(
                    [
                        None if row is None else [row[i] for i in chosen]
                        for row, chosen in zip(values, picks, strict=True)
                    ],
                    type=column.type,
                )
            )
        else:
            arrays.append(column)
    return pa.RecordBatch.from_arrays(arrays, names=list(batch.schema.names))
