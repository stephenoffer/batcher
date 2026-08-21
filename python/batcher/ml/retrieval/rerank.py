"""Cross-encoder reranking — the second stage that decides what a model actually reads.

Retrieval is a two-stage problem and Batcher only had the first stage. A bi-encoder
(`ml.embed`) turns every passage into a vector *once*, offline, and a vector search then
finds the nearest few hundred in milliseconds. That scales because the query and the
passage never meet: the passage's vector was computed without knowing the query, so it
cannot encode anything about how the two relate.

A **cross-encoder** is the other trade. It reads the query and one passage *together* and
scores that pair, so it sees the interaction the bi-encoder had to throw away — and it is
far more accurate for it. It also cannot be precomputed, so it can only ever run over a
candidate set the first stage already narrowed. Retrieve 100 with vectors, rerank to 5 with
a cross-encoder, and the 5 that reach the model are meaningfully better than the top 5 the
vector search alone would have returned.

That shape is what this module is built around, and the reason it is a batch UDF rather
than an expression: the model call must be **one forward pass over every (query, passage)
pair in the whole batch**, not one per row. A batch of 64 queries with 100 candidates each
is 6,400 pairs — a single well-filled GPU batch, or 64 tiny ones that leave the device
mostly idle. The flatten/score/regroup here is what turns the first into the second.

`ml.retrieval.mmr` reranks for *diversity* using vectors you already have, at no model
cost. This reranks for *relevance* using a model. They compose: rerank 100 to 20 with the
cross-encoder, then MMR 20 to 5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_names

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["CrossEncoderScorer", "cross_encoder_rerank_udf"]

CrossEncoderScorer = Callable[[list[tuple[str, str]]], Sequence[float]]
"""Scores a list of ``(query, passage)`` pairs, returning one relevance score per pair."""


def cross_encoder_rerank_udf(
    model: str | Callable[[], CrossEncoderScorer],
    *,
    query_column: str,
    document_column: str,
    rerank_columns: tuple[str, ...] = (),
    score_column: str | None = "rerank_score",
    k: int | None = None,
    max_length: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    activation: str | None = None,
) -> type:
    """A load-once class UDF that rescores each row's candidate passages with a cross-encoder.

    `document_column` is a **list** column holding one candidate per entry — the shape a
    vector search leaves behind, and the same shape `mmr_rerank_udf` consumes. Every column
    in `rerank_columns` is reordered alongside it, so the passages, their ids and their
    first-stage scores stay aligned. With `k`, only the top candidates survive.

    Every ``(query, passage)`` pair in the batch is scored in **one** model call, so the
    device sees a full batch rather than one tiny forward per row. Rows are then regrouped and
    sorted, which is pure NumPy over the flat score array.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import cross_encoder_rerank_udf
            >>> ds = bt.from_pydict(
            ...     {
            ...         "question": ["how tall is everest"],
            ...         "passages": [["everest is 8849 m", "k2 is 8611 m", "a recipe for soup"]],
            ...     }
            ... )
            >>> def scorer():  # a stand-in for a real cross-encoder
            ...     return lambda pairs: [float("everest" in p) for _, p in pairs]
            >>> udf = cross_encoder_rerank_udf(
            ...     scorer, query_column="question", document_column="passages", k=1
            ... )
            >>> ds.ml.map_batches(udf).to_pydict()["passages"]
            [['everest is 8849 m']]

    Args:
        model: a ``sentence-transformers`` cross-encoder model id (loaded once per worker),
            or a zero-arg factory returning a `CrossEncoderScorer` — anything that maps a
            list of ``(query, passage)`` pairs to one score each.
        query_column: the string column holding each row's query.
        document_column: the list column holding each row's candidate passages.
        rerank_columns: other per-candidate list columns to reorder alongside, such as the
            passage id or the first-stage score.
        score_column: the list column the reranker's own scores are written to, aligned with
            the reordered candidates. `None` to leave the scores out.
        k: candidates to keep per row, best first. `None` keeps and reorders all of them.
        max_length: token limit per pair, for the model-id path. A long passage is truncated
            rather than allowed to blow the model's window.
        device: torch device for the model-id path; defaults to the detected accelerator.
        batch_size: pairs per forward pass on the model-id path. Defaults to the whole
            batch's pairs, because the point of flattening them is to fill the device.
        activation: activation applied to the model's logits on the model-id path —
            ``"sigmoid"`` maps them to ``[0, 1]``, which is what a threshold wants. `None`
            keeps the raw logits, whose *order* is identical either way.

    Returns:
        A class for ``ds.ml.map_batches(...)`` — the cross-encoder loads once per worker.

    Raises:
        PlanError: if `k` is given and below 1.
    """
    if k is not None and k < 1:
        raise PlanError(f"cross_encoder_rerank_udf: k must be at least 1, got {k}")
    columns = tuple(dict.fromkeys((document_column, *rerank_columns)))

    class _CrossEncoderRerank:
        def __init__(self) -> None:
            self._score = (
                _sentence_transformer_scorer(
                    model,
                    max_length=max_length,
                    device=device,
                    batch_size=batch_size,
                    activation=activation,
                )
                if isinstance(model, str)
                else model()
            )

        def close(self) -> None:
            """Release the model when the worker is done, if it holds anything to release."""
            import contextlib

            close = getattr(self._score, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            return _rerank_batch(
                batch,
                self._score,
                query_column=query_column,
                columns=columns,
                score_column=score_column,
                k=k,
            )

    return _CrossEncoderRerank


def _rerank_batch(
    batch: pa.RecordBatch,
    score: CrossEncoderScorer,
    *,
    query_column: str,
    columns: tuple[str, ...],
    score_column: str | None,
    k: int | None,
) -> pa.RecordBatch:
    """Score every pair in `batch` at once, then reorder each row's aligned list columns."""
    import pyarrow as pa

    for name in (query_column, *columns):
        require_names(batch.schema.names, name, hint="Pass the retrieval output's column.")
    queries = batch.column(query_column).to_pylist()
    passages = batch.column(columns[0]).to_pylist()
    pairs, owners = _flatten_pairs(queries, passages)
    scores = _as_float_list(score(pairs) if pairs else [], len(pairs))
    order, kept = _per_row_order(scores, owners, len(queries), k)
    arrays = [batch.column(i) for i in range(batch.num_columns)]
    names = list(batch.schema.names)
    for name in columns:
        arrays[names.index(name)] = _reordered(batch.column(name).to_pylist(), order)
    if score_column is not None:
        column = pa.array(kept, type=pa.list_(pa.float64()))
        if score_column in names:
            arrays[names.index(score_column)] = column
        else:
            arrays.append(column)
            names.append(score_column)
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _flatten_pairs(
    queries: list[Any], passages: list[Any]
) -> tuple[list[tuple[str, str]], list[int]]:
    """Every ``(query, passage)`` pair in the batch, plus the row each pair came from.

    One flat list is the whole point: it is what lets a batch of 64 queries with 100
    candidates reach the model as 6,400 pairs in one forward rather than 64 forwards of 100.
    A null query or a null candidate renders as the empty string, matching what the embedding
    and prompt paths do, because a `None` reaching a tokenizer fails the entire batch over one
    bad row.
    """
    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for row, candidates in enumerate(passages):
        query = "" if queries[row] is None else str(queries[row])
        for candidate in candidates or ():
            pairs.append((query, "" if candidate is None else str(candidate)))
            owners.append(row)
    return pairs, owners


def _as_float_list(scores: Sequence[float], expected: int) -> list[float]:
    """The scorer's output as a plain float list, refusing a length that cannot be aligned.

    A scorer returning the wrong count would otherwise silently pair scores with the wrong
    passages from that point on — a reranking that looks entirely plausible and is wrong.
    """
    values = [float(v) for v in scores]
    if len(values) != expected:
        raise PlanError(
            f"cross-encoder scorer returned {len(values)} scores for {expected} "
            "(query, passage) pairs. It must return exactly one score per pair, in order."
        )
    return values


def _per_row_order(
    scores: list[float], owners: list[int], rows: int, k: int | None
) -> tuple[list[list[int]], list[list[float]]]:
    """Per-row candidate indices best-first (and their scores), truncated to `k`.

    The sort is stable, so candidates the model scored identically keep the first stage's
    order rather than an arbitrary one — which is what makes a rerank reproducible.
    """
    per_row: list[list[tuple[float, int]]] = [[] for _ in range(rows)]
    position = [0] * rows
    for score, row in zip(scores, owners, strict=True):
        per_row[row].append((score, position[row]))
        position[row] += 1
    order: list[list[int]] = []
    kept: list[list[float]] = []
    for candidates in per_row:
        ranked = sorted(candidates, key=lambda pair: -pair[0])
        if k is not None:
            ranked = ranked[:k]
        order.append([index for _, index in ranked])
        kept.append([value for value, _ in ranked])
    return order, kept


def _reordered(values: list[Any], order: list[list[int]]) -> pa.Array:
    """One list column rewritten to each row's selected candidates, in selection order."""
    import pyarrow as pa

    out: list[list[Any] | None] = []
    for row, indices in enumerate(order):
        candidates = values[row]
        if candidates is None:
            out.append(None)
            continue
        out.append([candidates[i] for i in indices if i < len(candidates)])
    return pa.array(out)


def _sentence_transformer_scorer(
    model: str,
    *,
    max_length: int | None,
    device: str | None,
    batch_size: int | None,
    activation: str | None,
) -> CrossEncoderScorer:
    """A `CrossEncoderScorer` backed by ``sentence_transformers.CrossEncoder``.

    Defaults are chosen for bulk reranking rather than for a single `predict` call: the device
    is the detected accelerator instead of whatever the library picks, and the forward batch is
    the whole flattened pair list instead of the library's internal default of 32 — which would
    otherwise cap device occupancy at 32 pairs however many the batch produced.
    """
    from batcher._internal.optional import require

    CrossEncoder = require(
        "sentence_transformers",
        "CrossEncoder",
        feature="ds.ml.rerank(<model id>)",
        provides="sentence-transformers",
        extra="st",
    )
    from batcher.ml.gpu import torch_device

    kwargs: dict[str, Any] = {"device": device or torch_device()}
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoder = CrossEncoder(model, **kwargs)
    activate = _activation(activation)

    def score(pairs: list[tuple[str, str]]) -> Sequence[float]:
        raw = encoder.predict(
            [list(pair) for pair in pairs],
            batch_size=batch_size or max(len(pairs), 1),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return activate(raw)

    score.close = getattr(encoder, "close", lambda: None)  # type: ignore[attr-defined]
    return score


def _activation(name: str | None) -> Callable[[Any], Any]:
    """The post-processing applied to a cross-encoder's raw logits.

    Order is unchanged by any monotonic activation, so this never affects the *ranking* — it
    affects whether the score you keep is comparable across queries or thresholdable.

    Raises:
        PlanError: If `name` is not `None` or ``"sigmoid"``.
    """
    if name is None:
        return lambda values: values
    if name != "sigmoid":
        raise PlanError(
            f"cross_encoder_rerank_udf: activation must be None or 'sigmoid', got {name!r}"
        )

    def sigmoid(values: Any) -> Any:
        import numpy as np

        array = np.asarray(values, dtype="float64")
        # Branch by *mask*, not with `np.where`, which evaluates both arms: `exp` of a large
        # positive logit overflows to inf, and the arm that would have produced NaN is
        # computed anyway — a warning, and a NaN, for exactly the most confident predictions
        # even though the selected value is right.
        out = np.empty_like(array)
        positive = array >= 0.0
        out[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
        exponent = np.exp(array[~positive])
        out[~positive] = exponent / (1.0 + exponent)
        return out

    return sigmoid
