"""A skewed *high-cardinality* key must have its skew measured.

Most-common-values exist to correct the uniform `1/ndv` equality estimate on exactly the
columns where uniformity is wrong. Gating the measurement on `ndv <= 1/min_frac` assumed
uniformity to decide whether to check for non-uniformity, so a high-cardinality key with
one dominant value — a sentinel, a default account, one whale customer, i.e. every join key
that ever needs salting — had its skew discarded and was estimated ~500x low.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

from batcher import core, kyber
from batcher.api.terminal._metadata import learn_column_stats
from batcher.io.source.inmemory import InMemorySource
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical import Filter, Scan
from batcher.plan.schema import SchemaRef
from batcher.plan.source_stats import source_stats_key

pytest.importorskip("batcher._native", reason="native engine not built")
pytestmark = pytest.mark.unit

_HOT = 7
_ROWS = 40_000


def _skewed_source() -> InMemorySource:
    """~1000 distinct keys, but `_HOT` is half of all rows."""
    rng = random.Random(7)
    vals = [_HOT] * (_ROWS // 2) + [rng.randrange(1000) for _ in range(_ROWS // 2)]
    rng.shuffle(vals)
    return InMemorySource(pa.table({"cust_id": pa.array(vals, type=pa.int64())}).to_batches())


def test_skew_on_a_high_cardinality_key_is_measured() -> None:
    hub = MetadataHub(InProcessBackend())
    src = _skewed_source()
    learn_column_stats(hub, [src.read()], [src])

    learned = kyber.load_learned_stats(hub)
    key = source_stats_key(src)
    ndv = kyber.columns_for(learned, kyber.NDV_KEY, key)["cust_id"]
    mcv = kyber.columns_for(learned, kyber.MCV_KEY, key)

    assert ndv > 20, "the point of the test: this key is well past the old gate"
    assert str(_HOT) in mcv.get("cust_id", {}), "the dominant key's skew was discarded"
    assert mcv["cust_id"][str(_HOT)] == pytest.approx(0.5, abs=0.05)


def test_the_measured_skew_sharpens_the_equality_estimate() -> None:
    """`cust_id = 7` must estimate ~half the table, not `1/ndv` of it."""
    hub = MetadataHub(InProcessBackend())
    src = _skewed_source()
    learn_column_stats(hub, [src.read()], [src])

    est = StatsEstimator([src], kyber.load_learned_stats(hub))
    scan = Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema()))
    hot = est.estimate(Filter(scan, Col("cust_id") == Lit(_HOT)))

    # Truth is ~50% of the rows. The uniform 1/ndv estimate would be ~40 rows.
    assert hot.rows == pytest.approx(_ROWS * 0.5, rel=0.15)


def test_a_cold_key_still_uses_the_uniform_estimate() -> None:
    """Only the *measured* hot values get a frequency; everything else stays 1/ndv."""
    hub = MetadataHub(InProcessBackend())
    src = _skewed_source()
    learn_column_stats(hub, [src.read()], [src])

    est = StatsEstimator([src], kyber.load_learned_stats(hub))
    scan = Scan(source_id=0, schema=SchemaRef.from_arrow(src.schema()))
    cold = est.estimate(Filter(scan, Col("cust_id") == Lit(999_999)))

    assert cold.rows < _ROWS * 0.1, "a non-hot value must not inherit the hot value's mass"


def test_measurement_uses_the_native_heavy_hitters(_=None) -> None:
    """Guard the primitive itself: Misra-Gries finds a hot value at high cardinality."""
    src = _skewed_source()
    hits = core.heavy_hitters(src.read(), ["cust_id"], 0.05)
    assert hits["cust_id"], "the sketch must find the dominant value"
    assert str(hits["cust_id"][0][0]) == str(_HOT)
