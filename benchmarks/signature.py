"""A small, order-independent fingerprint of a query result.

`harness.results_match` is the full comparison and stays the correctness gate. This is the
cheap one: a fixed-size value that can be computed once from the oracle and then asserted
by every client on every request, in a loop that runs thousands of times.

That distinction is why it exists. The concurrency harness has to check *each* request,
not just the first — a wrong answer that only appears at 16-way concurrency is exactly the
failure a QPS number would otherwise hide — and it cannot afford a full multiset
comparison per request. The fingerprint keeps the row count plus the first and last five
sorted rows, which catches a truncated, duplicated, reordered-into-wrongness, or silently
empty result at constant cost.

It is deliberately *not* a hash: when it mismatches, the two values print as readable rows,
so the failure says what went wrong rather than that something did.

Moved here from ``iso/worker.py`` when the concurrency and resilience harnesses needed the
same fingerprint; the logic is unchanged.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["SIGNATURE_EDGE_ROWS", "result_signature", "signatures_match"]

#: How many rows from each end of the sorted result the fingerprint keeps.
SIGNATURE_EDGE_ROWS = 5

#: Decimal places floats are rounded to before comparison. Matches `harness.ROUND_DECIMALS`
#: in spirit but is coarser: this fingerprint gates a *repeat* of a query the full
#: comparison already accepted, so its only job is to notice a different answer, not to
#: adjudicate a borderline one.
ROUND_DECIMALS = 4


def _scalar(value: Any) -> Any:
    """Normalize one cell to something orderable, comparable, and JSON-friendly."""
    if isinstance(value, Decimal):
        value = float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return "nan" if math.isnan(value) else round(value, ROUND_DECIMALS)
    return value


def result_signature(table: pa.Table) -> list:
    """A compact order-independent fingerprint of `table`.

    Rows are canonicalized, sorted by their repr, and only the row count plus the two
    edges are kept — so the fingerprint is bounded no matter how large the result is.

    Args:
        table: One engine's result.

    Returns:
        ``[row_count, first_rows, last_rows]``, where the row lists are empty or short.
    """
    cols = sorted(table.column_names)
    data = table.select(cols).to_pydict()
    n = table.num_rows
    rows = [tuple(_scalar(data[c][i]) for c in cols) for i in range(n)]
    rows.sort(key=lambda row: tuple(repr(x) for x in row))
    head = rows[:SIGNATURE_EDGE_ROWS]
    tail = rows[-SIGNATURE_EDGE_ROWS:] if n > SIGNATURE_EDGE_ROWS else []
    return [n, head, tail]


def signatures_match(expected: list, actual: list) -> tuple[bool, str]:
    """Compare two fingerprints. Returns ``(ok, message)``.

    Args:
        expected: The fingerprint taken from the oracle before the run.
        actual: The fingerprint of a result produced during the run.

    Returns:
        ``(True, "ok")`` when they agree, else ``(False, why)`` naming the first
        difference in a form a reader can act on.
    """
    if expected == actual:
        return True, "ok"
    if expected[0] != actual[0]:
        return False, f"row count: expected {expected[0]}, got {actual[0]}"
    for label, index in (("first", 1), ("last", 2)):
        if expected[index] != actual[index]:
            return False, (
                f"{label} rows differ: expected {expected[index]!r}, got {actual[index]!r}"
            )
    return False, f"expected {expected!r}, got {actual!r}"
