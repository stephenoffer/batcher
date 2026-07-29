"""Cardinality for nested and semi-structured shapes, where the *type* already knows.

An `Unnest` was estimated at a 1x fan-out on every cold run, on the stated grounds that
average list length is a property of the data. That is true of a variable-length list and
false of a `fixed_size_list`, whose length is in the type — the embedding and fixed-shape
vector column of every AI pipeline. Estimating a `fixed_size_list<float32, 768>` explode at
1x under-sizes every stage below it by 768x on the run that has nothing learned yet.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


def _embeddings(rows: int = 5, dim: int = 768):
    values = pa.array(np.zeros(rows * dim, dtype="float32"))
    return bt.from_arrow(pa.table({"e": pa.FixedSizeListArray.from_arrays(values, dim)}))


def _tensors(rows: int = 4, shape=(8, 8, 3)):
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((rows, *shape), dtype="uint8"))
    return bt.from_arrow(pa.table({"t": arr}))


def test_a_fixed_size_list_explode_has_an_exact_fanout():
    ds = _embeddings(rows=5, dim=768)
    est = CardinalityEstimator(ds._sources)
    exploded = ds.explode("e")._plan
    assert est.estimate(exploded).rows == 5 * 768


def test_the_exact_fanout_is_as_trusted_as_its_input():
    # A fan-out read off the type is a proof, not a guess, so it must not downgrade the
    # estimate's provenance to DEFAULT — which is the marker admission reads to decide
    # whether it is looking at an estimate or a placeholder.
    ds = _embeddings()
    est = CardinalityEstimator(ds._sources)
    inp = est.estimate(ds._plan)
    out = est.estimate(ds.explode("e")._plan)
    assert out.provenance == inp.provenance
    assert out.provenance != Provenance.DEFAULT


def test_a_variable_length_list_is_left_to_the_learning_loop():
    # The case the 1x default was written for and still the right answer: no structural
    # rule can know the average length, so the measured fan-out corrects it later.
    ds = bt.from_arrow(pa.table({"l": pa.array([[1, 2, 3], [4], [5, 6]])}))
    est = CardinalityEstimator(ds._sources)
    exploded = ds.explode("l")._plan
    assert est.estimate(exploded).rows == est.estimate(exploded.input).rows
    assert est.estimate(exploded).provenance == Provenance.DEFAULT


def test_an_extension_tensor_column_explodes_by_its_storage_length():
    # Same unwrap `column_bytes` needs: no `pa.types.is_*` predicate sees through an
    # extension label, and every decoded image/audio/video column in Batcher wears one.
    # A fixed-shape tensor's storage is the *flattened* fixed-size list, so a 8x8x3 frame
    # explodes to 192 elements — which is exactly what the engine produces.
    ds = _tensors(rows=4, shape=(8, 8, 3))
    est = CardinalityEstimator(ds._sources)
    assert est.estimate(ds.explode("t")._plan).rows == 4 * 8 * 8 * 3


@pytest.mark.parametrize(("rows", "dim"), [(5, 768), (3, 16), (1, 1536)])
def test_the_estimate_matches_what_the_engine_actually_emits(rows, dim):
    # The strongest form of the claim: not "bigger than before" but *equal to the truth*.
    ds = _embeddings(rows=rows, dim=dim)
    est = CardinalityEstimator(ds._sources)
    exploded = ds.explode("e")
    assert est.estimate(exploded._plan).rows == exploded.collect().num_rows


def test_the_fanout_scales_with_the_declared_dimension():
    small = _embeddings(rows=10, dim=16)
    large = _embeddings(rows=10, dim=1536)
    rows_small = CardinalityEstimator(small._sources).estimate(small.explode("e")._plan).rows
    rows_large = CardinalityEstimator(large._sources).estimate(large.explode("e")._plan).rows
    assert rows_large == 96 * rows_small


def test_an_explode_is_budgeted_at_its_fanout_not_at_one_morsel():
    """An `Unnest` emits its whole fan-out for a morsel in one batch, and nothing re-cuts it.

    Measured: exploding 4,000 rows of a `fixed_size_list<float32, 768>` yields a *single*
    3,072,000-row batch — 12 MB against a 1 MiB morsel budget. Budgeting the operator at one
    morsel under-counted it by the fan-out, in the direction that lets admission accept a
    query the node cannot run.
    """
    from batcher.config import active_config
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cost import CostModel

    cfg = active_config()
    ds = _embeddings(rows=64, dim=768)
    est = CardinalityEstimator(ds._sources)
    ops = annotate_ops(ds.explode("e")._plan, est, cfg, CostModel(est))
    unnest = next(o for o in ops if o.kind == "Unnest")
    scan = next(o for o in ops if o.kind == "Scan")
    # The scan is byte-capped at one morsel; the explode is not, and holds its fan-out.
    assert scan.bounds.m_max_bytes <= cfg.execution.morsel_bytes
    assert unnest.bounds.m_max_bytes > 50 * cfg.execution.morsel_bytes


def test_a_one_to_one_explode_is_still_one_morsel():
    """The safety property: a fan-out of 1 must leave the budget exactly where it was."""
    import pyarrow as pa_

    from batcher.config import active_config
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cost import CostModel

    cfg = active_config()
    values = pa_.array([1.0] * 8)
    ds = bt.from_arrow(pa_.table({"e": pa_.FixedSizeListArray.from_arrays(values, 1)}))
    est = CardinalityEstimator(ds._sources)
    ops = annotate_ops(ds.explode("e")._plan, est, cfg, CostModel(est))
    unnest = next(o for o in ops if o.kind == "Unnest")
    assert unnest.bounds.m_max_bytes <= cfg.execution.morsel_bytes


def test_the_explode_output_really_is_one_unbounded_batch():
    """The measurement the budget above rests on, pinned so it cannot silently change.

    If the engine ever starts re-morselizing an explode, this fails and the budget should
    go back to the byte-capped form — the point is that the estimate tracks the engine.
    """
    ds = _embeddings(rows=200, dim=768)
    sizes = [b.num_rows for b in ds.explode("e").iter_batches()]
    assert max(sizes) > 16_384


def test_an_unpivot_is_budgeted_at_its_column_count():
    """The same shape through the other expander, and its fan-out is exact.

    Measured: an `unpivot` of 4,000 rows over 20 columns emits one 80,000-row batch. The
    rule is stated as a property of the *fan-out* rather than of a node type, so it covers
    both expanders without a list to keep in sync.
    """
    import pyarrow as pa_

    from batcher.config import active_config
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cost import CostModel

    cfg = active_config()
    n = 64
    table = pa_.table(
        {"id": pa_.array(range(n)), **{f"c{i}": pa_.array([1.0] * n) for i in range(20)}}
    )
    ds = bt.from_arrow(table).unpivot(index=["id"], on=[f"c{i}" for i in range(20)])
    est = CardinalityEstimator(ds._sources)
    ops = annotate_ops(ds._plan, est, cfg, CostModel(est))
    up = next(o for o in ops if o.kind == "Unpivot")
    assert up.bounds.m_max_bytes > cfg.execution.morsel_bytes


def test_a_row_shrinking_operator_is_still_byte_capped():
    """A `Filter` cannot expand, so it must keep the morsel byte cap exactly."""
    from batcher.config import active_config
    from batcher.kyber.annotate import annotate_ops
    from batcher.kyber.cost import CostModel

    cfg = active_config()
    ds = bt.from_pydict({"x": list(range(1000))})
    est = CardinalityEstimator(ds._sources)
    ops = annotate_ops(ds.filter(bt.col("x") > 5)._plan, est, cfg, CostModel(est))
    for op in ops:
        assert op.bounds.m_max_bytes <= cfg.execution.morsel_bytes
