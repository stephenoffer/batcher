"""A rank-limited window drops rows, so `count()` must not answer from metadata.

`distinct(subset=…, keep="last", order_by=…)` lowers to
``Filter(Window([row_number]), rn == 1)``, and Kyber's fusion rule rewrites that into a
`Window` carrying ``rank_limit=1`` — the `Filter` disappears from the plan entirely. The
estimator then described `Window` as row-preserving and passed the child's **EXACT**
provenance through, which is what made this a wrong answer rather than a bad guess: the
metadata-answer path saw an EXACT row count and returned it *without executing*.

So `count()` reported the number of rows going **in** to the deduplication rather than the
number coming out. `collect()` was right the whole time, which is the signature of this bug
class — the two disagreed, and only the cheap one lied.

It surfaced through the merge: a source deduplicated with `distinct(subset=…)` (the exact
remedy `merge()` recommends for a cardinality violation) reported more rows than distinct
keys, and the merge rejected its own advice.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir import col


@pytest.mark.parametrize(
    ("table", "subset", "order_by", "expected"),
    [
        # Two rows, one key → one survivor.
        ({"id": [2, 2], "v": [98, 99], "seq": [1, 2]}, ["id"], "seq", 1),
        # Three distinct groups of differing size → three survivors.
        ({"g": [1, 1, 1, 2, 2, 3], "s": [1, 2, 3, 1, 2, 1]}, ["g"], "s", 3),
        # Every row its own key → nothing is dropped.
        ({"g": [1, 2, 3], "s": [1, 1, 1]}, ["g"], "s", 3),
        # A partition smaller than the rank limit still contributes only its own rows.
        ({"g": [1], "s": [7]}, ["g"], "s", 1),
    ],
)
def test_count_agrees_with_collect_after_keyed_distinct(table, subset, order_by, expected) -> None:
    ds = bt.from_arrow(pa.table(table)).distinct(subset=subset, keep="last", order_by=order_by)
    assert ds.collect().num_rows == expected
    assert ds.count() == expected, "count() answered from metadata and ignored the rank limit"


def test_a_plain_window_is_still_row_preserving() -> None:
    """The fix must not cost the un-limited window its (correct) row-preserving estimate."""
    ds = bt.from_arrow(pa.table({"g": [1, 1, 2], "s": [1, 2, 3]}))
    windowed = ds.with_columns(total=col("s").sum().over("g"))
    assert windowed.count() == 3 == windowed.collect().num_rows


def test_is_empty_is_not_fooled_either() -> None:
    """`is_empty()` reads the same EXACT-provenance channel `count()` does."""
    ds = bt.from_arrow(pa.table({"id": [1, 1], "seq": [1, 2]}))
    deduped = ds.distinct(subset=["id"], keep="last", order_by="seq")
    assert not deduped.is_empty()
    assert deduped.count() == 1
