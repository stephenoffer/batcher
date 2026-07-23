"""A measured distinct count must inform the optimizer without answering a query.

A Parquet footer gives EXACT min/max and null counts but **never a distinct count**, so the
only `ndv` a columnar source can have is a measured (HLL) one. `ColumnStat` carried a single
provenance for the whole bundle, so attaching that ndv to an EXACT column would have tagged it
EXACT and let it answer `count_distinct` — and so it was refused outright. The price was that
every Parquet column reached the optimizer with **no ndv at all**: join cardinality fell back
to `max(|L|, |R|)`, every join in a query looked the same size, and join ordering went blind.
TPC-H q9 applied its 5%-selective `part` filter *last* and ran 5.8x slower than DuckDB; with
the ndv restored it applies it first and runs 6.8x faster.

`ndv_provenance` is the fix: the distinct count carries its own tag. These pin both halves —
that the optimizer *uses* it, and that an exact answer still *refuses* it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.kyber.stats.columns import scan_columns
from batcher.plan.stats import ColumnStat, Provenance

pytestmark = pytest.mark.unit


def _footer_column() -> ColumnStat:
    """What a Parquet footer actually supplies: exact bounds, and no distinct count."""
    return ColumnStat(min=1, max=200_000, null_count=0, ndv=None, provenance=Provenance.EXACT)


def _measured(ndv: float) -> ColumnStat:
    """What the metadata loop measures: an HLL distinct count."""
    return ColumnStat(ndv=ndv, provenance=Provenance.SKETCH)


def test_a_measured_ndv_reaches_an_exact_footer_column():
    """The regression: this used to be dropped, leaving the join estimator with no ndv."""
    cols = scan_columns({"k": _footer_column()}, {"k": _measured(201_152)})
    assert cols["k"].ndv == 201_152


def test_the_footer_bounds_stay_exact():
    """The ndv rides alongside; it must not downgrade what the footer proved."""
    cols = scan_columns({"k": _footer_column()}, {"k": _measured(201_152)})
    assert cols["k"].provenance is Provenance.EXACT
    assert (cols["k"].min, cols["k"].max) == (1, 200_000)


def test_a_measured_ndv_is_not_exact_and_cannot_answer_count_distinct():
    """The whole safety argument: a sketch count may inform cost, never answer a query."""
    cols = scan_columns({"k": _footer_column()}, {"k": _measured(201_152)})
    assert cols["k"].ndv_provenance is Provenance.SKETCH
    assert cols["k"].ndv_is_exact is False


def test_a_source_declared_exact_ndv_stays_exact():
    """A source that really does record a true distinct count may still answer exactly."""
    declared = ColumnStat(min=1, max=3, ndv=3, provenance=Provenance.EXACT)
    cols = scan_columns({"k": declared}, {})
    assert cols["k"].ndv == 3
    assert cols["k"].ndv_is_exact is True


def test_a_measured_ndv_never_overwrites_a_declared_one():
    declared = ColumnStat(min=1, max=3, ndv=3, provenance=Provenance.EXACT)
    cols = scan_columns({"k": declared}, {"k": _measured(999)})
    assert cols["k"].ndv == 3
    assert cols["k"].ndv_is_exact is True


def test_downgrade_weakens_the_ndv_tag_too():
    """A row-shrinking operator cannot make an ndv *more* trustworthy."""
    exact = ColumnStat(min=1, max=3, ndv=3, provenance=Provenance.EXACT)
    assert exact.downgrade(Provenance.DEFAULT).ndv_is_exact is False


def test_count_distinct_refuses_a_sketch_ndv_end_to_end(tmp_path):
    """`n_unique()` must execute, not answer from the HLL count the optimizer now holds."""
    import batcher as bt

    path = str(tmp_path / "t.parquet")
    pa.parquet = pytest.importorskip("pyarrow.parquet")
    pa.parquet.write_table(pa.table({"k": list(range(1000)) * 3}), path)

    ds = bt.read.parquet(path)
    ds.collect()  # a first run measures the column, seeding a SKETCH ndv
    assert ds.select("k").n_unique("k") == 1000  # exact, so it must be a real execution
