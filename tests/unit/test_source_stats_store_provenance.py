"""Bug-hunt regression: per-field provenance sub-tags must survive persistence.

`ColumnStat` carries a bundle `provenance` plus two per-field sub-tags — `ndv_provenance`
and `null_count_provenance` — that let a *sketch* distinct count ride beside *exact* bounds
(and an *exact* null count ride beside byte-truncated bounds). `ndv_is_exact` /
`null_count_is_exact` fall back to the bundle tag only when the sub-tag is ``None``.

The defect: ``source_stats_store`` persisted only the bundle `provenance`, silently dropping
both sub-tags. On reload they came back ``None``, so the trust gate fell back to the bundle
tag — **promoting a SKETCH distinct count to EXACT** (a wrong `count_distinct` answered from
an approximate value) and, in the other direction, **demoting an exact null count** to a
needless rescan. The fix round-trips both sub-tags (and the `mean` field, previously dropped).
"""

from __future__ import annotations

import pytest

from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.metadata.source_stats_store import load_source_stats, save_source_stats
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

pytestmark = pytest.mark.unit


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def test_sketch_ndv_beside_exact_bounds_stays_inexact() -> None:
    # Exact bounds/null count, but the distinct count is a sketch: `count_distinct` must
    # NOT be answerable from it after a reload.
    hub = _hub()
    col = ColumnStat(
        min=0,
        max=99,
        null_count=0,
        ndv=50,
        provenance=Provenance.EXACT,
        ndv_provenance=Provenance.SKETCH,
    )
    assert col.ndv_is_exact is False  # baseline before persistence
    save_source_stats(hub, "src://a", SourceStatistics(row_count=100, columns={"c": col}))
    got = load_source_stats(hub, "src://a")
    assert got is not None
    rc = got.columns["c"]
    assert rc.ndv_provenance is Provenance.SKETCH
    assert rc.ndv_is_exact is False  # the corruption was a reload flipping this to True


def test_exact_null_count_beside_weak_bounds_stays_exact() -> None:
    # Byte-truncated (DEFAULT) bounds, but the null count is exact: `null_count()` must stay
    # answerable from it after a reload (dropping the sub-tag would demote it to a rescan).
    hub = _hub()
    col = ColumnStat(
        null_count=7,
        provenance=Provenance.DEFAULT,
        null_count_provenance=Provenance.EXACT,
    )
    assert col.null_count_is_exact is True
    save_source_stats(hub, "src://b", SourceStatistics(row_count=42, columns={"c": col}))
    got = load_source_stats(hub, "src://b")
    assert got is not None
    rc = got.columns["c"]
    assert rc.null_count_provenance is Provenance.EXACT
    assert rc.null_count_is_exact is True


def test_mean_and_bundle_provenance_roundtrip() -> None:
    hub = _hub()
    col = ColumnStat(min=1, max=9, ndv=5, mean=4.5, provenance=Provenance.EXACT)
    save_source_stats(hub, "src://c", SourceStatistics(row_count=9, columns={"c": col}))
    rc = load_source_stats(hub, "src://c").columns["c"]
    assert rc.mean == 4.5
    assert rc.provenance is Provenance.EXACT
    # A bundle-only provenance (no per-field override) still falls back correctly.
    assert rc.ndv_provenance is None
    assert rc.ndv_is_exact is True


def test_exact_moments_beside_boundless_stats_stay_exact() -> None:
    """The third sub-tag, in the same direction as the null count's.

    An immutable in-memory relation computes its own `sum`/`mean` on demand, and the
    conductor collects the surrounding bundle only for the columns a `MIN`/`MAX` reads — so
    a `SUM`'s bundle holds no bounds and is tagged `DEFAULT` while the total inside it is
    exact. Dropping `moments_provenance` on reload sends `SELECT sum(x) FROM t` back to a
    full scan of a relation whose total is already on file.
    """
    hub = _hub()
    col = ColumnStat(
        null_count=0,
        total_sum=4950.0,
        mean=49.5,
        provenance=Provenance.DEFAULT,
        moments_provenance=Provenance.EXACT,
    )
    assert col.moments_are_exact is True  # baseline before persistence
    save_source_stats(hub, "src://moments", SourceStatistics(row_count=100, columns={"c": col}))
    got = load_source_stats(hub, "src://moments")
    assert got is not None
    rc = got.columns["c"]
    assert rc.moments_provenance is Provenance.EXACT
    assert rc.moments_are_exact is True
    assert rc.total_sum == 4950.0
    assert rc.mean == 49.5


def test_moments_without_a_subtag_follow_the_bundle() -> None:
    """No sub-tag means "the bundle speaks for it" — the pre-existing contract, unchanged.

    A catalog-recorded total on an EXACT bundle stays answerable, and one on a DEFAULT
    bundle stays refused, exactly as before the sub-tag existed.
    """
    for bundle, expected in ((Provenance.EXACT, True), (Provenance.DEFAULT, False)):
        hub = _hub()
        col = ColumnStat(total_sum=1.0, mean=1.0, provenance=bundle)
        save_source_stats(hub, "src://plain", SourceStatistics(row_count=1, columns={"c": col}))
        got = load_source_stats(hub, "src://plain")
        assert got is not None
        assert got.columns["c"].moments_provenance is None
        assert got.columns["c"].moments_are_exact is expected
