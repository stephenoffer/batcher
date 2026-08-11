"""Turning a per-row list of term codes into a bag-of-words row, without a Python loop.

A vectorizer's `transform` has to collapse each document's term list into ``(index, value)``
pairs, and the obvious implementation — a `Counter` per row — is a Python loop over every
token in the corpus. This module does the same work with Arrow and NumPy kernels over the
*flattened* list column: one `index_in` lookup for the whole batch, one sort to group
``(row, code)`` pairs, and one `reduceat` for the per-row norms. The cost is O(tokens) in
compiled code and the control plane never sees a token.

The output is the compressed-sparse-row pair a bag of words actually is: an `indices` list
column and a `values` list column, aligned element-for-element per row. Batcher has no
sparse-vector type to hide that behind, and inventing one here would be a bespoke columnar
format the rest of the engine could not read — so the two list columns *are* the contract,
and `scipy.sparse` or a torch sparse tensor is one call away from them. A `dense` mode is
offered for a small vocabulary, where a fixed-width list column is what a downstream
trainer wants anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

__all__ = ["bag_of_words", "set_columns"]


def set_columns(batch: Any, columns: dict[str, Any]) -> Any:
    """Replace or append each named column on `batch`, preserving the rest.

    Args:
        batch: The Arrow `RecordBatch` to extend.
        columns: The ``{name: array}`` columns to write.

    Returns:
        A new `RecordBatch` carrying the requested columns.
    """
    # One rebuild, not one per column: `set_column`/`append_column` each copy the whole
    # column list, and `Schema.names` materializes a fresh list per membership test, so
    # this was O(columns x batch width) on a path that runs per batch.
    from batcher.ml.tabular.features import append_columns

    return append_columns(batch, columns)


def _row_segments(list_array: Any) -> tuple[Any, np.ndarray]:
    """The flattened values of a list column and each row's element count.

    `list_flatten` and `list_value_length` are used rather than the array's own
    ``values``/``offsets`` buffers because those describe the *unsliced* array: a batch
    handed to a UDF is routinely a zero-copy slice of a larger one, and reading the raw
    buffers would silently mix in a neighbouring row's tokens.
    """
    import numpy as np
    import pyarrow.compute as pc

    lengths = pc.fill_null(pc.list_value_length(list_array), 0)
    return pc.list_flatten(list_array), lengths.to_numpy(zero_copy_only=False).astype(np.int64)


def _counts_per_row(
    rows: np.ndarray, codes: np.ndarray, n_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group ``(row, code)`` pairs into per-row runs, returning ``(offsets, codes, counts)``.

    Packing the pair into one integer key lets a single sort do the grouping, which is the
    whole reason this is O(tokens) rather than a dictionary per document.
    """
    import numpy as np

    if rows.size == 0:
        return np.zeros(n_rows + 1, dtype=np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    width = int(codes.max()) + 1 if codes.size else 1
    keys = rows.astype(np.int64) * width + codes.astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    out_rows, out_codes = np.divmod(unique, width)
    # `unique` is sorted, so rows are already grouped and ascending: the boundary of each
    # row's run is a search for its first key.
    offsets = np.searchsorted(out_rows, np.arange(n_rows + 1), side="left").astype(np.int64)
    return offsets, out_codes, counts.astype(np.int64)


def _normalize(values: np.ndarray, offsets: np.ndarray, norm: str | None) -> np.ndarray:
    """Scale each row's values to unit norm in place, leaving an all-zero row alone."""
    import numpy as np

    if norm is None or values.size == 0:
        return values
    lengths = np.diff(offsets)
    starts = offsets[:-1][lengths > 0]
    if starts.size == 0:
        return values
    magnitude = np.abs(values) if norm == "l1" else values * values
    totals = np.add.reduceat(magnitude, starts)
    if norm == "l2":
        totals = np.sqrt(totals)
    scale = np.where(totals > 0, 1.0 / np.where(totals > 0, totals, 1.0), 1.0)
    return values * np.repeat(scale, lengths[lengths > 0])


def bag_of_words(
    list_array: Any,
    *,
    vocabulary: Any,
    n_features: int,
    binary: bool = False,
    sublinear_tf: bool = False,
    weights: np.ndarray | None = None,
    norm: str | None = None,
    dense: bool = False,
) -> dict[str, Any]:
    """Vectorize one batch's term lists into aligned sparse (or dense) Arrow columns.

    Args:
        list_array: The ``List<Utf8>`` term column, or a ``List<Int64>`` of codes when
            `vocabulary` is ``None`` (the hashing path, where the codes are already final).
        vocabulary: The Arrow array of terms to look codes up in, or ``None`` when
            `list_array` already holds codes.
        n_features: The width of the feature space; a code outside it is dropped.
        binary: Record presence as ``1.0`` rather than the term count.
        sublinear_tf: Replace a count ``n`` with ``1 + log(n)``.
        weights: A per-feature multiplier (the IDF vector), or ``None``.
        norm: ``"l1"``, ``"l2"``, or ``None`` for no per-row normalization.
        dense: Return one fixed-width ``List<Float64>`` column instead of the index/value
            pair. Only sane for a small `n_features`.

    Returns:
        ``{"indices": array, "values": array}``, or ``{"values": array}`` when `dense`.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    n_rows = len(list_array)
    flat, lengths = _row_segments(list_array)
    codes_array = flat if vocabulary is None else pc.index_in(flat, value_set=vocabulary)
    codes = pc.fill_null(codes_array, -1).to_numpy(zero_copy_only=False).astype(np.int64)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), lengths)
    # An out-of-vocabulary term, and a hashed code outside the feature space, are both
    # dropped rather than folded into a bucket: scikit-learn ignores unseen terms, and a
    # catch-all bucket would quietly make one feature mean "everything I have not seen".
    keep = (codes >= 0) & (codes < n_features)
    offsets, out_codes, counts = _counts_per_row(rows[keep], codes[keep], n_rows)

    values = np.ones(counts.shape, dtype=np.float64) if binary else counts.astype(np.float64)
    if sublinear_tf and not binary:
        values = 1.0 + np.log(values)
    if weights is not None and values.size:
        values = values * weights[out_codes]
    values = _normalize(values, offsets, norm)

    if dense:
        matrix = np.zeros((n_rows, n_features), dtype=np.float64)
        if out_codes.size:
            matrix[np.repeat(np.arange(n_rows), np.diff(offsets)), out_codes] = values
        dense_offsets = np.arange(n_rows + 1, dtype=np.int64) * n_features
        return {
            "values": pa.ListArray.from_arrays(pa.array(dense_offsets), pa.array(matrix.ravel()))
        }
    return {
        "indices": pa.ListArray.from_arrays(pa.array(offsets), pa.array(out_codes)),
        "values": pa.ListArray.from_arrays(pa.array(offsets), pa.array(values)),
    }
