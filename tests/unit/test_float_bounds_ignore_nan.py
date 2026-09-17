"""A recorded float maximum is not a bound on a column that may hold NaN.

The engine ranks NaN above every number, and Parquet, Delta and Iceberg all keep NaN out of a
float column's recorded min/max. Each layer that reads those bounds must therefore refuse to
prove `x > v`, `x >= v` or `x != v` from the max, while still using the min and the max's other
proofs where no NaN can change the answer. These tests pin that rule at each layer separately:
the statistics bridge Kyber reasons over, the manifest file-skipping pass, and the pyarrow filter
translation. The end-to-end agreement is `tests/differential/test_diff_parquet_nan_pruning.py`.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.dataset as pads
import pytest

from batcher.io.predicate import to_pyarrow_expression
from batcher.io.stats.file_skipping import surviving_files
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

pytestmark = pytest.mark.unit


def _cmp(op: str, column: str, value: dict) -> dict:
    return {
        "e": "binary",
        "op": op,
        "left": {"e": "col", "name": column},
        "right": {"e": "lit", "value": value},
    }


# --- the statistics bridge ----------------------------------------------------------------


def _stats(*, bounds_include_nan: bool) -> SourceStatistics:
    return SourceStatistics(
        row_count=10,
        columns={
            "f": ColumnStat(min=0.1, max=0.5, null_count=0, provenance=Provenance.EXACT),
            "i": ColumnStat(min=1, max=5, null_count=0, provenance=Provenance.EXACT),
        },
        bounds_include_nan=bounds_include_nan,
    )


def test_a_footer_float_max_reaches_the_optimizer_as_unknown():
    columns = _stats(bounds_include_nan=False).to_relstats(default_rows=1.0).columns
    assert columns["f"].max is None
    assert columns["f"].min == 0.1  # no NaN is below the minimum
    assert columns["i"].max == 5  # an integer column has no NaN to hide


def test_bounds_computed_in_the_engine_order_keep_their_max():
    columns = _stats(bounds_include_nan=True).to_relstats(default_rows=1.0).columns
    assert columns["f"].max == 0.5


# --- manifest file skipping ---------------------------------------------------------------


def _manifest() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"path": "small", "num_records": 10, "min.f": 0.1, "max.f": 0.5, "null_count.f": 0},
            {"path": "large", "num_records": 10, "min.f": 2.0, "max.f": 3.0, "null_count.f": 0},
        ]
    )


@pytest.mark.parametrize("op", ["gt", "ge", "ne"])
def test_a_float_file_is_kept_for_what_a_nan_satisfies(op):
    kept = surviving_files(_cmp(op, "f", {"float": 0.9}), _manifest())
    assert kept is None or "small" in kept


def test_a_float_file_is_still_skipped_for_what_no_nan_satisfies():
    """The positive control: the rule above must not switch float skipping off entirely."""
    assert surviving_files(_cmp("lt", "f", {"float": 1.0}), _manifest()) == ["small"]


# --- the pyarrow filter translation -------------------------------------------------------


def _rows(expr) -> int:
    table = pa.table({"f": [0.1, float("nan"), 2.0], "i": [1, 2, 3]})
    return pads.dataset(table).to_table(filter=expr).num_rows


def test_a_float_greater_than_keeps_nan_rows():
    schema = pa.schema([("f", pa.float64()), ("i", pa.int64())])
    assert _rows(to_pyarrow_expression(_cmp("gt", "f", {"float": 1.0}), schema)) == 2
    assert _rows(to_pyarrow_expression(_cmp("ge", "f", {"float": 1.0}), schema)) == 2


def test_a_float_less_than_still_excludes_nan_rows():
    schema = pa.schema([("f", pa.float64()), ("i", pa.int64())])
    assert _rows(to_pyarrow_expression(_cmp("lt", "f", {"float": 1.0}), schema)) == 1


def test_the_translation_is_exact_under_negation():
    """`NOT (f > 1)` must drop the NaN row, as the engine does, not keep it."""
    schema = pa.schema([("f", pa.float64()), ("i", pa.int64())])
    negated = {"e": "not", "input": _cmp("gt", "f", {"float": 1.0})}
    assert _rows(to_pyarrow_expression(negated, schema)) == 1


def test_an_integer_comparison_is_unchanged():
    schema = pa.schema([("f", pa.float64()), ("i", pa.int64())])
    expr = to_pyarrow_expression(_cmp("gt", "i", {"int": 1}), schema)
    assert "is_nan" not in str(expr)
