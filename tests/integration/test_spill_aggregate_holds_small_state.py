"""A *reducing* out-of-core aggregate holds its partials instead of bucketing them to disk.

The out-of-core aggregate is reached on the size of the **input**, but what it has to hold is
its **state**. For a reducing group-by those differ by orders of magnitude, and the partition
phase was paying the difference on every chunk: hash-partition the partial, open a bucket
file, write it, and read every bucket back in a reduce pass — for a partial that is four rows.

It is reached on a bad number, too. `GROUP BY l_returnflag, l_linestatus` over TPC-H
`lineitem` has **four** groups; with no column statistics Kyber reads the group count as at
least one morsel's worth, so `kyber.annotate._aggregate_resident_bytes` sizes the operator's
envelope at the whole input — 17.6 GB at sf100 — which is what both routes the query
out-of-core and asks for 132 buckets.

So the partition phase now holds partials until they actually exceed
`memory.spill_bucket_max_bytes`, and only then starts bucketing. Measured on that query,
sf100 on local NVMe: **40.4 s -> 28.9 s (1.45x)**, the per-chunk `partition_batches` going
from 9.8 s to zero and the reduce pass from 5.6 s to 1.8 s.

These tests pin the two halves that could break: a reducing aggregate must produce the same
rows while writing nothing, and one whose state genuinely outgrows the budget must still
switch to buckets mid-stream and produce the same rows across the switch.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import Config, config_context
from batcher.dist.spill import aggregate as spill_agg

pytestmark = pytest.mark.integration


def _rows(t: pa.Table):
    return sorted(
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    )


def _batches(n_batches: int, groups: int, per: int = 2_000):
    rng = np.random.default_rng(11)
    return [
        pa.record_batch(
            {
                "k": rng.integers(0, groups, per).astype("int64"),
                "v": rng.integers(0, 100, per).astype("int64"),
            }
        )
        for _ in range(n_batches)
    ]


def _spilled_and_memory(batches, budget_bytes: int, hold_bytes: int | None = None):
    """The same grouped aggregate run out-of-core under `budget_bytes`, and in memory.

    `hold_bytes` caps the partials the partition phase may keep before it starts bucketing.
    Both live on one `Config`: nesting a second `config_context` inside this one would
    replace it wholesale, which is how the first version of this test silently kept the
    128 MiB default and never spilled.

    `collect(spill=True)` rather than a small `max_memory_bytes`, because the memory route
    could not be relied on to fire: `from_batches` reports no row count, so
    `projected_input_bytes` reads 0 and the size-based spill gate never triggers. The tests
    then passed while measuring an ordinary in-memory run — which is what the `entries`
    positive control in `_spy` exists to catch, and did.
    """
    schema = batches[0].schema
    table = pa.Table.from_batches(batches)

    def q(ds):
        return ds.group_by("k").agg(s=col("v").sum(), n=count(), mx=col("v").max())

    base = Config()
    mem = dataclasses.replace(base.memory, max_memory_bytes=budget_bytes)
    if hold_bytes is not None:
        mem = dataclasses.replace(mem, spill_bucket_max_bytes=hold_bytes)
    with config_context(base.replace(memory=mem)):
        spilled = q(bt.from_batches((lambda: iter(batches)), schema)).collect(spill=True)
    return spilled, q(bt.from_arrow(table)).collect()


def _spy(monkeypatch) -> tuple[list[int], list[int]]:
    """Record (bucket writes, out-of-core entries).

    The second list is the positive control, and it is not optional: `writes == []` is also
    what a query that never went out-of-core at all produces, so without it the "nothing was
    spilled" assertion below would hold just as well against a plain in-memory run.
    """
    writes: list[int] = []
    entries: list[int] = []
    real_write = spill_agg._spill_partial
    real_exec = spill_agg.execute_spilling_aggregate

    def write_spy(writers, nat, partial, key_idx, n_buckets):
        writes.append(n_buckets)
        return real_write(writers, nat, partial, key_idx, n_buckets)

    def exec_spy(*a, **k):
        entries.append(1)
        return real_exec(*a, **k)

    monkeypatch.setattr(spill_agg, "_spill_partial", write_spy)
    monkeypatch.setattr(spill_agg, "execute_spilling_aggregate", exec_spy)
    monkeypatch.setattr("batcher.dist.spill.execute_spilling_aggregate", exec_spy, raising=False)
    return writes, entries


def test_a_reducing_aggregate_writes_no_buckets_at_all(monkeypatch):
    """Eight groups over 240,000 rows: the state is bytes, so nothing should reach disk."""
    writes, entries = _spy(monkeypatch)
    batches = _batches(n_batches=120, groups=8)
    spilled, in_memory = _spilled_and_memory(batches, budget_bytes=1 << 20)

    assert _rows(spilled) == _rows(in_memory)
    assert spilled.num_rows == 8
    assert entries, "the query must actually go out-of-core, or the next assertion is vacuous"
    assert writes == [], "a reducing aggregate's partials fit in memory; none should be spilled"


def test_a_non_reducing_aggregate_still_switches_to_buckets(monkeypatch):
    """A near-unique key outgrows the hold budget, so the partition phase must take over.

    The switch happens *mid-stream*, so this is also the test that the partials held before
    it and the chunks bucketed after it still meet: `combine` is associative and commutative,
    so a group's rows reach the same bucket whichever side of the switch they arrived on.
    """
    writes, entries = _spy(monkeypatch)
    # `hold_bytes` down to the floor `_held_partial_budget` enforces (1 MiB); 1M near-unique
    # keys make ~16 MB of partials, so the switch is forced well before the stream ends.
    batches = _batches(n_batches=200, groups=10_000_000, per=5_000)
    spilled, in_memory = _spilled_and_memory(batches, budget_bytes=1 << 20, hold_bytes=1 << 20)

    assert _rows(spilled) == _rows(in_memory)
    assert entries, "the query must actually go out-of-core"
    assert writes, "a state larger than the hold budget must fall back to bucketed spilling"


def test_an_empty_input_still_returns_the_global_identity_row():
    """The hold path must not swallow the one row a keyless aggregate owes an empty input.

    `count()` over nothing is `0`, not "no rows" — both the in-memory engine and DuckDB say
    so — and the held-partials branch returns nothing at all, so it has to fall through to
    the empty-input handling rather than answer from an empty hold.
    """
    empty = pa.record_batch({"k": pa.array([], pa.int64()), "v": pa.array([], pa.int64())})
    base = Config()
    tiny = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=1 << 20))
    with config_context(tiny):
        got = (
            bt.from_batches((lambda: iter([empty])), empty.schema)
            .agg(n=count())
            .collect(spill=True)
        )
    assert got.to_pydict() == {"n": [0]}
