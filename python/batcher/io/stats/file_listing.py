"""Statistics for a source where one row *is* one file, read from the listing alone.

A blob, media, or document corpus has a property no columnar format has: the reader
already knows the exact row count and the exact size of every row before it opens
anything, because a row is a file and the listing reports both. Nothing has to be
scanned, sampled, or estimated.

That matters more than it sounds, and the reason is `plan.types.widths.column_bytes`.
Asked how wide a `binary` column is, it can only answer from the *type*, and its prior is
36 bytes. For a directory of 200 MB videos that is wrong by six orders of magnitude, and
the figure feeds broadcast eligibility, split sizing, spill budgeting and join build-side
choice — so a media corpus and a thumbnail corpus were costed identically. Declaring
`content_byte_size=True` is what tells the width estimator that `byte_size / row_count` is
the real answer here and not a stored-encoding artefact to be ignored (see
`SourceStatistics.content_byte_size`, which explains why a *columnar* source must not
claim it).

The `size` bounds are EXACT in the strong sense: they are minima and maxima over the
actual values, not per-chunk bounds that happen to contain them, so `WHERE size > ...`
prunes files outright.

This lives here rather than on either caller because both `multimodal.media` and
`unstructured.binary` produce the identical shape from the identical inputs, and they sit
in different format packages — the one arrangement where a second copy is the only way to
share, and therefore the one to avoid.
"""

from __future__ import annotations

from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["whole_file_statistics"]


def whole_file_statistics(sizes: list[int], *, size_column: str = "size") -> SourceStatistics:
    """`SourceStatistics` for a one-row-per-file source, from its files' sizes.

    Args:
        sizes: Every file's byte size, one per row the source will produce.
        size_column: The name of the column carrying each row's byte size, which receives
            the exact zone map. Pass a different name for a schema that spells it
            otherwise; pass one absent from the schema and it is simply never consulted.

    Returns:
        The statistics: an exact row count, the summed byte size marked as row *content*,
        and an exact `[min, max]` on the size column.
    """
    columns: dict[str, ColumnStat] = {}
    if sizes:
        columns[size_column] = ColumnStat(
            min=min(sizes), max=max(sizes), null_count=0, provenance=Provenance.EXACT
        )
    return SourceStatistics(
        row_count=len(sizes),
        byte_size=sum(sizes) or None,
        columns=columns,
        exact_rows=True,
        content_byte_size=True,
    )
