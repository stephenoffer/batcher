"""Ordering a corpus so a training batch is not mostly padding.

A training batch is a rectangle. Every sequence in it is padded to the longest one, and every
padded position costs the same attention arithmetic as a real token while teaching the model
nothing. On a corpus whose lengths vary — which is every natural-language corpus — a randomly
ordered batch pairs a 40-token example with a 2,000-token one and spends most of its compute on
padding.

Sorting the whole corpus by length fixes that and breaks the training: the model then sees all
the short examples first and all the long ones last, which is a curriculum nobody chose and a
gradient distribution that shifts across the epoch.

Length-grouped shuffling is the standard compromise. Shuffle, cut the shuffled stream into
*megabatches* of many batches each, and sort by length only inside a megabatch. Batches are then
length-homogeneous, while the epoch order stays random at the scale that matters for training.

`padding_waste` measures what the ordering is worth, because the answer depends entirely on the
corpus's length distribution: on a uniform-length corpus it is zero either way, and reaching for
this would be complexity for nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["length_grouped_order", "padding_waste"]

_LENGTH = "__bt_length"
_ROW = "__bt_row"
_MEGABATCH = "__bt_megabatch"


def _require_column(ds: Dataset, column: str) -> None:
    """Fail at the API edge, naming the columns that do exist."""
    if column not in ds.columns:
        from batcher._internal.errors import ColumnNotFoundError, unknown_message

        raise ColumnNotFoundError(
            unknown_message("column", column, ds.columns, hint="Pass an existing column.")
        )


def _length_of(ds: Dataset, column: str):
    """A row's length, whichever way the column expresses one.

    A tokenized corpus carries a list of ids and a raw one carries text, and both are ordinary
    things to order by. Choosing on the column's Arrow type rather than asking the caller means
    the same call works before and after tokenization.
    """
    import pyarrow as pa

    from batcher.plan.expr_ir.constructors import col

    dtype = ds.schema.field(column).type
    if (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    ):
        return col(column).list.len()
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return col(column).str.len()
    raise PlanError(
        f"length_grouped_order: {column!r} is {dtype}, which has no length. Pass a list column "
        f"of token ids or a text column."
    )


def length_grouped_order(
    ds: Dataset,
    column: str,
    *,
    batch_size: int,
    megabatch_factor: int = 50,
    seed: int = 0,
) -> Dataset:
    """Order a corpus so each batch holds similar-length rows, without sorting the epoch.

    Shuffles, cuts the shuffled stream into megabatches of ``batch_size * megabatch_factor``
    rows, and sorts by length only *within* a megabatch. Batches come out length-homogeneous —
    so little of each is padding — while the order across the epoch stays random.

    `megabatch_factor` is the trade-off. Larger means more length-homogeneous batches and less
    padding, and also a longer stretch of similar-length examples in a row; the usual range is
    20 to 100. At 1 it is a plain shuffle, which is the point of comparison rather than a
    degenerate case.

    The length comes from the column's type: a list column measures its elements (token ids),
    a text column its characters. Consume the result in order — the ordering is the product,
    and re-shuffling downstream discards it.

    Args:
        ds: The corpus to order.
        column: The list-of-token-ids or text column to measure.
        batch_size: Rows per training batch.
        megabatch_factor: Batches per megabatch, the window length-sorting is confined to.
        seed: Seed for the shuffle, so an epoch is reproducible.

    Returns:
        A new dataset in the length-grouped order, with no extra columns.

    Raises:
        PlanError: If `batch_size` or `megabatch_factor` is below 1, or the column has no
            length.
        ColumnNotFoundError: If `column` is not in the dataset.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import length_grouped_order
            >>> corpus = bt.from_pydict({"tokens": [[1] * n for n in (1, 9, 2, 8, 3, 7)]})
            >>> ordered = length_grouped_order(corpus, "tokens", batch_size=2)
            >>> ordered.columns
            ['tokens']
            >>> ordered.count()
            6
    """
    from batcher.plan.expr_ir.constructors import col

    _require_column(ds, column)
    if batch_size < 1:
        raise PlanError(f"length_grouped_order: batch_size must be at least 1, got {batch_size}")
    if megabatch_factor < 1:
        raise PlanError(
            f"length_grouped_order: megabatch_factor must be at least 1, got {megabatch_factor}"
        )
    original = list(ds.columns)
    window = batch_size * megabatch_factor
    return (
        ds.shuffle(seed=seed)
        .with_row_index(_ROW)
        .with_columns(**{_LENGTH: _length_of(ds, column)})
        .with_columns(**{_MEGABATCH: col(_ROW) // window})
        .sort(_MEGABATCH, _LENGTH)
        .select(*original)
    )


def padding_waste(ds: Dataset, column: str, *, batch_size: int) -> float:
    """The fraction of a batched run's positions that would be padding, in the current order.

    Batches the corpus in its present order, pads each batch to its own longest row, and
    returns the padded share of the total. It is the number that says whether reordering is
    worth anything: measure it before `length_grouped_order` and after, and if the two are
    close the corpus's lengths are already uniform and the ordering buys nothing.

    Order matters, so call it on the dataset you would actually train on. A lazily-ordered
    dataset is materialized here, since padding is a property of the realized sequence of rows.

    Args:
        ds: The corpus, in the order it would be consumed.
        column: The list-of-token-ids or text column to measure.
        batch_size: Rows per batch.

    Returns:
        The padded fraction of all positions, in ``[0, 1)``. Zero when every batch is uniform.

    Raises:
        PlanError: If `batch_size` is below 1, or the column has no length.
        ColumnNotFoundError: If `column` is not in the dataset.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import length_grouped_order, padding_waste
            >>> lengths = [1, 9, 2, 8, 3, 7, 4, 6]
            >>> corpus = bt.from_pydict({"tokens": [[1] * n for n in lengths]})
            >>> # Adjacent short/long pairs waste most of every batch.
            >>> round(padding_waste(corpus, "tokens", batch_size=2), 3)
            0.333
            >>> grouped = length_grouped_order(corpus, "tokens", batch_size=2)
            >>> round(padding_waste(grouped, "tokens", batch_size=2), 3)
            0.091
    """
    _require_column(ds, column)
    if batch_size < 1:
        raise PlanError(f"padding_waste: batch_size must be at least 1, got {batch_size}")
    lengths = ds.select(**{_LENGTH: _length_of(ds, column)}).to_pydict()[_LENGTH]
    real = 0
    padded = 0
    for start in range(0, len(lengths), batch_size):
        batch = [0 if n is None else n for n in lengths[start : start + batch_size]]
        if not batch:
            continue
        real += sum(batch)
        padded += max(batch) * len(batch)
    if padded == 0:
        return 0.0
    return (padded - real) / padded
