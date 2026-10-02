"""The join-skew pre-pass counts a bounded, streamed sample, never a whole split at once.

`dist.skew.sample_heavy_hitters` replaced a pre-pass that read its whole split and ran the
join side over all of it in memory -- which, on 3 x m5.4xlarge at TPC-H SF1000, held enough
of `orders` to lose q13 and q10 to the out-of-memory killer. It must still find the hot key,
stop reading once the sample is taken, and never hand the engine more than a chunk at once.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest

from batcher._internal.native import engine
from batcher.dist import skew
from batcher.dist.executors.ray_runtime import engine_config_json

pytestmark = pytest.mark.unit

_SCAN = json.dumps({"op": "scan", "source_id": 0})
_BATCH = 10_000


def _batches(n_batches: int, consumed: list[int]):
    """Batches whose key 7 holds 30% of the rows; counts how many were read."""
    for b in range(n_batches):
        consumed.append(b)
        keys = [7 if (b * _BATCH + i) % 10 < 3 else b * _BATCH + i for i in range(_BATCH)]
        yield pa.record_batch({"k": pa.array(keys, pa.int64())})


def test_the_hot_key_is_found_and_reading_stops_at_the_sample(monkeypatch):
    monkeypatch.setattr(skew, "_SAMPLE_ROWS", 100_000)
    monkeypatch.setattr(skew, "_SAMPLE_CHUNK_ROWS", 20_000)
    largest_chunk: list[int] = []
    nat = engine()

    class Recording:
        """The engine, recording how many rows each sub-plan run is handed."""

        def execute_plan(self, ir, sources, cfg):
            largest_chunk.append(sum(b.num_rows for b in sources[0]))
            return nat.execute_plan(ir, sources, cfg)

        def heavy_hitters(self, *args):
            return nat.heavy_hitters(*args)

    consumed: list[int] = []
    pairs, seen = skew.sample_heavy_hitters(
        Recording(), _SCAN, "k", _batches(200, consumed), 0.1, engine_config_json()
    )
    counts = dict(pairs)
    hot = {v for v, c in counts.items() if c >= 0.1 * seen}
    assert {int(v) for v in hot} == {7}
    # Misra-Gries counts are lower bounds; the 30% key reads below 30%, never above it.
    assert 0.1 < counts[next(iter(hot))] / seen <= 0.3
    # It stopped well short of the 2,000,000 rows on offer, at about the sample size.
    assert 100_000 <= seen <= 140_000
    assert len(consumed) < 20
    assert max(largest_chunk) <= 20_000 + _BATCH


def test_an_empty_split_counts_nothing():
    pairs, seen = skew.sample_heavy_hitters(
        engine(), _SCAN, "k", iter([]), 0.1, engine_config_json()
    )
    assert (pairs, seen) == ([], 0)
