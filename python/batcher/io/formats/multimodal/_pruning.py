"""Prune media files at plan time using the metadata a listing already knows.

A media relation's cheap columns — ``uri``, ``size``, ``mime`` — are known from the
directory listing and a stat, before a single payload byte is read. So a predicate over
*only* those columns can be evaluated against the file list itself, and the files it
excludes never become a split, never get scheduled, and never get opened. On a corpus
where the interesting rows are a small fraction (``WHERE mime = 'video/mp4' AND size <
50000000``) that is the difference between reading a directory and reading a terabyte.

Unlike Parquet zone-map skipping, this is **exact** rather than conservative: `size` and
`mime` are the values themselves, not bounds over a chunk, so a surviving file is a file
that genuinely has matching rows. The engine keeps its `Filter` regardless, so pruning
only ever affects how much I/O happens — which is why refusing to prune (returning every
file) is always the safe answer and is what every unsupported predicate shape does.

Vectorized over the file dimension with `pyarrow.compute`, never a Python loop per file —
the pattern `io/stats/file_skipping.py` establishes.
"""

from __future__ import annotations

import mimetypes
from typing import Any

import pyarrow as pa

from batcher.io.predicate import to_pyarrow_expression

__all__ = ["prunable_columns", "prune_files"]

# The columns a media listing can answer without opening the file. A predicate touching
# anything else (an image's width, the payload itself) cannot be decided here.
_PRUNABLE = frozenset({"uri", "size", "mime"})


def prunable_columns(ir: dict[str, Any] | None) -> bool:
    """Whether every column `ir` references is answerable from the file listing alone.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.multimodal._pruning import prunable_columns
            >>> col = {"e": "col", "name": "size"}
            >>> lit = {"e": "lit", "value": {"int": 10}}
            >>> prunable_columns({"e": "binary", "op": "gt", "left": col, "right": lit})
            True
            >>> other = {"e": "col", "name": "width"}
            >>> prunable_columns({"e": "binary", "op": "gt", "left": other, "right": lit})
            False

    Args:
        ir: A predicate's IR dictionary, or None.

    Returns:
        True when the predicate can be decided from ``uri``/``size``/``mime`` alone.
    """
    if not ir:
        return False
    return _columns(ir) <= _PRUNABLE


def _columns(ir: Any) -> set[str]:
    """Every column name referenced anywhere in a predicate IR."""
    if not isinstance(ir, dict):
        return set()
    if ir.get("e") == "col":
        return {ir["name"]}
    found: set[str] = set()
    for value in ir.values():
        if isinstance(value, dict):
            found |= _columns(value)
        elif isinstance(value, list):
            for item in value:
                found |= _columns(item)
    return found


def prune_files(
    files: list[str], sizes: list[int], predicate: dict[str, Any] | None
) -> tuple[list[str], list[int]] | None:
    """Drop the files a listing-only predicate proves cannot match.

    Args:
        files: The candidate paths.
        sizes: Each path's size in bytes, positionally aligned with `files`.
        predicate: The pushed filter as its IR dictionary, or None.

    Returns:
        The surviving `(files, sizes)`, or None when the predicate cannot be decided
        from the listing — the caller then reads everything, which is always correct.
    """
    if not files or not prunable_columns(predicate):
        return None
    table = pa.table(
        {
            "uri": pa.array(files, pa.string()),
            "size": pa.array(sizes, pa.int64()),
            "mime": pa.array(
                [mimetypes.guess_type(f)[0] or "application/octet-stream" for f in files],
                pa.string(),
            ),
        }
    )
    expression = to_pyarrow_expression(predicate, table.schema)
    if expression is None:
        return None
    try:
        kept = table.filter(expression)
    except Exception:
        # A shape pyarrow declines to evaluate (an unusual literal type, say) must not
        # fail the read — fall back to reading everything.
        return None
    surviving = kept.column("uri").to_pylist()
    return surviving, kept.column("size").to_pylist()
