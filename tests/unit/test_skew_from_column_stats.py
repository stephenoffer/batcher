"""Kyber knows a join key is skewed before the join has ever run.

Skew is a property of the *column*, not of the query. If `cust_id = 7` holds 47% of the
rows, that is true of every join on `cust_id` — so the distributed join should not have to
discover it with a Misra-Gries pre-pass over both sides, nor wait until the identical query
shape has run once. `kyber.hot_join_values` answers it from the column statistics the
metadata loop already measured, at no cost and on the first run.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

from batcher import kyber
from batcher.api.terminal._metadata import learn_column_stats
from batcher.io.source.inmemory import InMemorySource
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.hub import MetadataHub
from batcher.plan.logical import Join, JoinOutputCol, Scan
from batcher.plan.schema import SchemaRef

pytest.importorskip("batcher._native", reason="native engine not built")
pytestmark = pytest.mark.unit

_HOT = 7
_ROWS = 40_000
_FRACTION = 0.05


def _skewed() -> InMemorySource:
    """~1000 distinct keys; `_HOT` is half the rows."""
    rng = random.Random(11)
    vals = [_HOT] * (_ROWS // 2) + [rng.randrange(1000) for _ in range(_ROWS // 2)]
    rng.shuffle(vals)
    return InMemorySource(pa.table({"k": pa.array(vals, type=pa.int64())}).to_batches())


def _uniform() -> InMemorySource:
    rng = random.Random(12)
    vals = [rng.randrange(1000) for _ in range(_ROWS)]
    return InMemorySource(pa.table({"k": pa.array(vals, type=pa.int64())}).to_batches())


def _dim() -> InMemorySource:
    return InMemorySource(
        pa.table({"k": pa.array(list(range(1000)), type=pa.int64())}).to_batches()
    )


def _join(left: InMemorySource, right: InMemorySource) -> Join:
    return Join(
        left=Scan(source_id=0, schema=SchemaRef.from_arrow(left.schema())),
        right=Scan(source_id=1, schema=SchemaRef.from_arrow(right.schema())),
        left_keys=("k",),
        right_keys=("k",),
        join_type="inner",
        output=(
            JoinOutputCol(side="left", name="k", alias="k"),
            JoinOutputCol(side="right", name="k", alias="k_r"),
        ),
    )


def _learned_hub(*sources: InMemorySource) -> MetadataHub:
    hub = MetadataHub(InProcessBackend())
    for src in sources:
        learn_column_stats(hub, [src.read()], [src])
    return hub


def test_a_skewed_key_is_known_hot_from_column_statistics() -> None:
    fact, dim = _skewed(), _dim()
    hub = _learned_hub(fact, dim)
    hot = kyber.hot_join_values(_join(fact, dim), [fact, dim], hub, _FRACTION)
    assert hot == [str(_HOT)]


def test_a_uniform_key_is_not_reported_hot() -> None:
    """No value clears the fraction, so salting must stay off — it is not free."""
    fact, dim = _uniform(), _dim()
    hub = _learned_hub(fact, dim)
    assert kyber.hot_join_values(_join(fact, dim), [fact, dim], hub, _FRACTION) == []


def test_nothing_is_claimed_before_anything_is_measured() -> None:
    """A cold hub knows no skew; the caller keeps its existing behavior."""
    fact, dim = _skewed(), _dim()
    cold = MetadataHub(InProcessBackend())
    assert kyber.hot_join_values(_join(fact, dim), [fact, dim], cold, _FRACTION) == []


def test_a_multi_key_join_is_declined() -> None:
    """Salting is defined for a single key; a composite key must not be guessed at."""
    fact, dim = _skewed(), _dim()
    hub = _learned_hub(fact, dim)
    join = _join(fact, dim)
    multi = Join(
        left=join.left,
        right=join.right,
        left_keys=("k", "k"),
        right_keys=("k", "k"),
        join_type="inner",
        output=join.output,
    )
    assert kyber.hot_join_values(multi, [fact, dim], hub, _FRACTION) == []


def test_the_hot_value_is_reported_as_the_string_the_partitioner_keys_on() -> None:
    """`salted_partition_batches` keys on the rendered value, so the form must match."""
    fact, dim = _skewed(), _dim()
    hub = _learned_hub(fact, dim)
    hot = kyber.hot_join_values(_join(fact, dim), [fact, dim], hub, _FRACTION)
    assert all(isinstance(v, str) for v in hot)
