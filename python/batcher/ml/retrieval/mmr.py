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

The selection is greedy and sequential per query, so it cannot be an `Expr`. It is a batch UDF
instead, and the work inside one query is NumPy: the candidates' pairwise similarity is a single
matrix multiply, and the running redundancy is updated per selection rather than recomputed,
which is the difference between `O(k*n)` and `O(k^2*n)` on a wide candidate set.

The loop over *rows* is still Python, one iteration per query. That is honest rather than ideal
— a rerank is `k` candidates per query, not a column of tokens, so the per-row overhead is small
against the matrix multiply it wraps; but on a batch of a million queries it is a million Python
iterations, and if that ever dominates, the fix is a kernel rather than a faster loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_names

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
    require_names(batch.schema.names, name, hint="Pass a list column.")


def _require_grouped_candidates(batch: pa.RecordBatch, embedding_column: str) -> None:
    """Fail here if `embedding_column` holds one vector per row rather than a candidate list.

    MMR reranks *within* a query, so its input is one row per query whose embedding column
    holds that query's whole candidate set — a ``list<list<double>>``. The natural mistake is
    to hand it the retrieval output before grouping, one candidate per row, which is a
    ``list<double>``: the same column name, the same element type, one nesting level short.

    NumPy's answer to that was ``AxisError: axis 1 is out of bounds for array of dimension
    1``, raised from `_unit_rows` three frames inside the UDF — a message about an array the
    caller never built, naming neither the column nor the shape it needed. Checked once per
    batch off the schema, not per row, so it costs nothing.
    """
    import pyarrow as pa

    field = batch.schema.field(batch.schema.get_field_index(embedding_column))
    dtype = field.type
    if not pa.types.is_list(dtype) and not pa.types.is_large_list(dtype):
        raise PlanError(
            f"mmr_rerank expects {embedding_column!r} to hold each query's candidate vectors "
            f"as a list, but it is {dtype}."
        )
    inner = dtype.value_type
    if (
        pa.types.is_list(inner)
        or pa.types.is_large_list(inner)
        or pa.types.is_fixed_size_list(inner)
        # An all-empty candidate column infers `list<null>`: with no elements there is nothing
        # to type, so the nesting cannot be read off the schema. That is unresolved, not
        # wrong, and a batch of empty candidate sets is a legitimate (if degenerate) input the
        # row loop already handles. Only a type positively recognizable as a single vector per
        # row is rejected.
        or pa.types.is_null(inner)
    ):
        return
    raise PlanError(
        f"mmr_rerank expects {embedding_column!r} to hold one *candidate set* per row — a "
        f"list of vectors, so `list<list<...>>` — but it is {dtype}, which is a single vector "
        "per row. Reranking happens within a query, so group the retrieved candidates by the "
        "query first (for example `.group_by(query_key).agg(...)` collecting the embeddings "
        "and scores into lists), then rerank the grouped rows."
    )


def _take(candidates: list | None, chosen: list[int], name: str, row: int) -> list | None:
    """The chosen candidates of one row's per-candidate list, or null when the row is null.

    A per-candidate column shorter than the embedding column would index past its end, and
    a bare `IndexError` from inside a comprehension names neither the column nor the row.
    The lists are per-candidate *by declaration* (`rerank_columns`), so a mismatch is the
    caller having grouped them apart, and that is what the message says.
    """
    if candidates is None:
        return None
    if chosen and max(chosen) >= len(candidates):
        raise PlanError(
            f"mmr_rerank: row {row} of {name!r} has {len(candidates)} values but the row has "
            f"{max(chosen) + 1} or more candidates. Every per-candidate column must line up "
            "one-for-one with the embedding column; group them in the same aggregation."
        )
    return [candidates[i] for i in chosen]


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
    _require_grouped_candidates(batch, embedding_column)

    embeddings = batch.column(batch.schema.get_field_index(embedding_column)).to_pylist()
    scores = (
        batch.column(batch.schema.get_field_index(score_column)).to_pylist()
        if score_column
        else [None] * batch.num_rows
    )

    picks: list[list[int]] = []
    for row, (row_vectors, row_scores) in enumerate(zip(embeddings, scores, strict=True)):
        if not row_vectors:
            picks.append([])
            continue
        if row_scores is not None and len(row_scores) != len(row_vectors):
            # Grouping the embeddings and the scores separately — a different filter on
            # either side, a join that dropped a candidate — leaves the two lists out of
            # step for that query. Left to NumPy it surfaced as "operands could not be
            # broadcast together with shapes (2,) (3,)", which names neither the columns,
            # nor the row, nor the fix.
            raise PlanError(
                f"mmr_rerank: row {row} has {len(row_vectors)} candidate vectors in "
                f"{embedding_column!r} but {len(row_scores)} scores in {score_column!r}. "
                "Each row's per-candidate lists must line up one-for-one; group them in "
                "the same aggregation so a dropped candidate drops from both."
            )
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
                        _take(candidates, chosen, name, row)
                        for row, (candidates, chosen) in enumerate(zip(values, picks, strict=True))
                    ],
                    type=column.type,
                )
            )
        else:
            arrays.append(column)
    return pa.RecordBatch.from_arrays(arrays, names=list(batch.schema.names))
