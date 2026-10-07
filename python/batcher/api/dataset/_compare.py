"""Tolerant result comparison for `Dataset.equals(rtol=, atol=, check_dtypes=)`.

`equals` compares *results* exactly by default, which is the right answer for a regression
check and the wrong one for a migration: a ported query whose float aggregate was summed in a
different order differs in the last bits, and so does one run single-node and one run
distributed (`.claude/rules/python-control-plane.md` records reassociation as an accepted
divergence). This module is the opt-in comparison for that case. It works on the two already
materialized Arrow tables and compares whole columns with vectorized kernels, never row by
row.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["tolerant_equals"]


def tolerant_equals(
    left: pa.Table,
    right: pa.Table,
    *,
    ordered: bool,
    rtol: float,
    atol: float,
    check_dtypes: bool,
) -> bool:
    """Whether two results match, with float columns compared to a tolerance.

    Args:
        left: This side's result; its schema is the reference.
        right: The other side's result, with the same column names.
        ordered: Compare in emitted order rather than as multisets.
        rtol: Relative tolerance for float columns, as in ``numpy.isclose``.
        atol: Absolute tolerance for float columns, as in ``numpy.isclose``.
        check_dtypes: Require identical column types. When false, `right` is cast to
            `left`'s types first, and a value that does not survive the cast is a mismatch.

    Returns:
        True when the two results match under the tolerance.
    """
    if left.num_rows != right.num_rows:
        return False
    if left.schema != right.schema:
        if check_dtypes:
            return False
        try:
            right = right.cast(left.schema)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return False
    if not ordered:
        keys = _sort_keys(left.schema)
        left, right = left.sort_by(keys), right.sort_by(keys)
    return all(
        _columns_match(left.column(i), right.column(i), rtol, atol) for i in range(left.num_columns)
    )


def _sort_keys(schema: pa.Schema) -> list[tuple[str, str]]:
    """Sort keys that line two multisets up row for row: exact columns first, floats last.

    A float that differs in the last bits can sort on the other side of its neighbour, so
    the exact columns lead and decide the order wherever they can. Nested and
    dictionary-encoded columns cannot be sort keys at all and are left out, which only
    narrows what the sort can line up.
    """
    exact, floats = [], []
    for field in schema:
        kind = field.type
        if pa.types.is_nested(kind) or pa.types.is_dictionary(kind):
            continue
        (floats if pa.types.is_floating(kind) else exact).append((field.name, "ascending"))
    return exact + floats


def _columns_match(left: pa.ChunkedArray, right: pa.ChunkedArray, rtol: float, atol: float) -> bool:
    """One column pair: within tolerance for floats, exactly equal for everything else.

    Floats need the null positions to agree before the values are compared, and treat two
    NaNs as equal, as the default exact comparison of `equals` does.
    """
    if not pa.types.is_floating(left.type):
        return left.equals(right)
    import numpy as np

    if not left.is_null().equals(right.is_null()):
        return False
    zero = pa.scalar(0.0, left.type)
    lhs = left.fill_null(zero).to_numpy()
    rhs = right.fill_null(zero).to_numpy()
    return bool(np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=True).all())
