"""The ML/genAI surface must cooperate with the rest of the engine, not sit beside it.

A feature that only works single-node, or that the optimizer cannot see through, is not
a feature at PB scale. These pin the three places where the ML additions meet the
architecture, each of which was broken when the feature was first written:

* **the split is a row-wise filter.** Built on `with_random` it depended on `RowId`,
  which has no distributed implementation and is not in the streaming allow-list — so
  `ds.ml.train_test_split` silently pinned a pipeline to one node. Hashing the row's
  own content instead makes each part a `Filter`, the shape both the distributed
  executor and the streaming engine already handle.
* **explode distributes.** The RAG ingest shape (scan → chunk → explode → embed) has an
  `Unnest` in the middle; while `Unnest` was absent from the linear-map set the whole
  chain fell to the single-node fallback.
* **Kyber can learn the explode fan-out.** The rows an explode emits is the average list
  length — a property of the data, not the plan. No structural rule can know it, so the
  estimator is wrong by exactly that factor until Core measures it. `Unnest` must
  therefore be a *correctable* operator, or a 20:1 chunker under-sizes every stage below.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.dist.executors.plan_analysis import _has_breaker, _is_linear_map_pipeline
from batcher.kyber.cost import CostModel
from batcher.kyber.learning import CARDINALITY_CORRECTION_KEY, load_learned_stats
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.metadata import MetadataHub
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.logical import Filter
from batcher.plan.logical.transforms import is_streamable

pytestmark = pytest.mark.unit


def _identity(batch):
    return batch


@pytest.fixture
def docs():
    return bt.from_pydict({"id": [1, 2], "doc": ["a" * 40, "b" * 30]})


# --- the split cooperates with distribution and streaming ---------------------


def test_train_test_split_parts_are_plain_filters():
    """No RowId, no extra column, no shuffle — just a predicate over the input."""
    train, test = bt.range(0, 100).ml.train_test_split(0.2, seed=1)
    assert isinstance(train._plan, Filter)
    assert isinstance(test._plan, Filter)
    assert train.columns == test.columns == ["value"]


def test_split_parts_are_streamable_and_distributable():
    train, test = bt.range(0, 100).ml.train_test_split(0.2, seed=1)
    for part in (train, test):
        assert is_streamable(part._plan), "a split must survive on an unbounded source"
        assert _is_linear_map_pipeline(part._plan), "a split must distribute"


def test_the_old_row_index_based_split_would_not_have_distributed():
    """Guards the reason for the rewrite: `with_random` pulls in a non-distributable
    `RowId`, so a split built on it is single-node whatever the cluster size."""
    ds = bt.range(0, 100)
    row_index_based = ds.with_random("u", seed=1).filter(bt.col("u") < 0.2).drop("u")
    assert not _is_linear_map_pipeline(row_index_based._plan)
    assert not is_streamable(row_index_based._plan)


def test_split_is_partition_independent():
    """The same row lands in the same part however the data is chunked."""
    rows = {"id": list(range(500))}
    one_batch = bt.from_pydict(rows)
    many_batches = bt.from_arrow(one_batch.to_arrow().to_batches(max_chunksize=17))
    a, _ = one_batch.ml.train_test_split(0.3, seed=9)
    b, _ = many_batches.ml.train_test_split(0.3, seed=9)
    assert set(a.to_pydict()["id"]) == set(b.to_pydict()["id"])


def test_key_keeps_the_split_stable_across_a_schema_change():
    """Recomputing a feature must not move rows between train and test."""
    base = bt.from_pydict({"id": list(range(400)), "f": [i % 7 for i in range(400)]})
    grown = base.with_columns(new_feature=bt.col("f") * 3.5)
    with_key, _ = base.ml.train_test_split(0.3, seed=5, key="id")
    with_key_grown, _ = grown.ml.train_test_split(0.3, seed=5, key="id")
    assert set(with_key.to_pydict()["id"]) == set(with_key_grown.to_pydict()["id"])


def test_without_a_key_a_new_column_reshuffles_the_split():
    """The default hashes every column, so the split tracks the whole row — correct,
    but the reason `key=` exists."""
    base = bt.from_pydict({"id": list(range(400)), "f": [i % 7 for i in range(400)]})
    grown = base.with_columns(new_feature=bt.col("f") * 3.5)
    a, _ = base.ml.train_test_split(0.3, seed=5)
    b, _ = grown.ml.train_test_split(0.3, seed=5)
    assert set(a.to_pydict()["id"]) != set(b.to_pydict()["id"])


def test_split_rejects_an_unknown_key_column():
    with pytest.raises(PlanError, match="unknown key column"):
        bt.range(0, 10).ml.train_test_split(0.2, key="nope")


def test_a_single_fraction_returns_the_dataset_unchanged():
    ds = bt.range(0, 50)
    (only,) = ds.ml.random_split([1.0])
    assert only.count() == 50


# --- the RAG pipeline distributes and streams ---------------------------------


def test_chunk_explode_embed_distributes(docs):
    """The RAG ingest shape must be embarrassingly parallel, not single-node."""
    rag = (
        docs.with_columns(chunk=bt.col("doc").str.chunk(10, overlap=2))
        .explode("chunk")
        .ml.map_batches(_identity)
    )
    assert _is_linear_map_pipeline(rag._plan)
    assert is_streamable(rag._plan)


def test_explode_is_not_a_pipeline_breaker(docs):
    """It multiplies rows but holds no state and materializes nothing."""
    exploded = docs.with_columns(c=bt.col("doc").str.chunk(10)).explode("c")
    assert not _has_breaker(exploded._plan)


def test_explode_is_costed_by_its_output_rows():
    """A zero-cost explode would let Kyber float it above an inference stage, where it
    multiplies the rows that stage pays for."""
    from batcher.plan.logical import Unnest

    ds = bt.from_pydict({"c": [[1, 2, 3]] * 10})
    plan = ds.explode("c")._plan
    assert isinstance(plan, Unnest)
    model = CostModel(StatsEstimator(ds._sources))
    assert model.op_cost(plan).cpu > 0.0


# --- Kyber learns the explode fan-out -----------------------------------------


def _feed(hub, signature, *, est, actual, times=8):
    for _ in range(times):
        hub.record(
            OperatorFeedback(
                op_id=OpId(0),
                kind="unnest",
                n_actual=actual,
                t_op_ms=1.0,
                m_peak_bytes=0,
                selectivity=1.0,
                batch_size=1024,
                signature=signature,
                n_estimated=est,
            )
        )


def test_kyber_learns_the_explode_fanout():
    """Structurally the estimator must guess 1x; after Core measures it, 10x."""
    ds = bt.from_pydict({"id": list(range(100)), "doc": ["x" * 100] * 100})
    rag = ds.with_columns(c=bt.col("doc").str.chunk(10)).explode("c")
    plan = rag._plan
    true_rows = rag.count()

    cold = StatsEstimator(ds._sources)
    structural = cold.estimate(plan).rows
    signature = cold.signature_of(plan)
    assert structural == pytest.approx(100.0), "the structural rule cannot know list length"
    assert true_rows == 1000

    # An explode teaches the loop something, so it must be reported as a sample.
    assert cold.reportable_estimate(plan) == pytest.approx(structural)

    hub = MetadataHub(InProcessBackend())
    _feed(hub, signature, est=structural, actual=true_rows)
    corrections = load_learned_stats(hub).get(CARDINALITY_CORRECTION_KEY, {})
    assert corrections.get(signature) == pytest.approx(10.0)

    warm = StatsEstimator(ds._sources, learned=load_learned_stats(hub))
    assert warm.estimate(plan).rows == pytest.approx(float(true_rows))
