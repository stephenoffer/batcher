"""The bucket mechanics every out-of-core breaker shares (`dist.spill.buckets`).

The aggregate, the join, the partitioned window, the sort and the global window each used
to state these for themselves, and they had already drifted: the join derived its
re-partition salt by a second formula and named its own recursion constants, and one of the
five wrote the store's publish/cleanup pair in the other order. What is pinned here is that
there is now one answer and that the breakers hold to it — a divergence in any of these is
invisible in a result (a wrong salt makes a grace split move no rows and the query still
returns the right answer, just out of memory), which is why it needs a test rather than a
convention.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher.dist.spill.buckets as buckets_mod
from batcher.carbonite.spill import TieredSpillStore
from batcher.config import Config, MemoryConfig, config_context
from batcher.dist.spill.buckets import (
    GRACE_DEPTH,
    GRACE_SUB_BUCKETS,
    BucketWriters,
    over_envelope,
    read_reserved_bucket,
    resident_bytes,
    spill_scratch,
    split_salt,
)

pytestmark = pytest.mark.unit


def _batch(n: int, base: int = 0) -> pa.RecordBatch:
    return pa.record_batch({"k": list(range(base, base + n)), "v": [base] * n})


class TestTheBreakersAgreeOnOneGraceRecursion:
    """Every breaker's named constants resolve to the shared ones, not to copies."""

    def test_every_breaker_grace_splits_to_the_same_depth_and_width(self) -> None:
        from batcher.dist.spill import aggregate as agg
        from batcher.dist.spill_breakers import join, window

        depths = {
            agg._MAX_SPILL_RECURSION,
            join._MAX_JOIN_SPLIT_DEPTH,
            window._MAX_WINDOW_SPLIT_DEPTH,
        }
        widths = {agg._SUB_BUCKETS, join._JOIN_SUB_BUCKETS, window._WINDOW_SUB_BUCKETS}
        assert depths == {GRACE_DEPTH}, "a breaker re-split to its own depth"
        assert widths == {GRACE_SUB_BUCKETS}, "a breaker re-split to its own width"

    def test_the_join_reducers_sub_bucket_salt_is_the_shared_depth_zero_salt(self) -> None:
        """It was `0x9E3779B97F4A7C15 | 1`, a second formula that happens to agree.

        Agreement by coincidence is what nothing was checking: the two expressions differ
        (`(K * (d+1)) | 1` against `(K | 1)`) and coincide only because K is odd and d is 0.
        """
        from batcher.dist.spill_breakers.join import _SUBBUCKET_SALT

        assert split_salt(0) == _SUBBUCKET_SALT


class TestTheSalt:
    def test_it_is_odd_so_a_power_of_two_re_split_actually_moves_rows(self) -> None:
        """Zero is the unsalted cluster-wide assignment; an even salt is nearly as bad.

        Bucket assignment reads the low hash bits at a power-of-two count, so a re-partition
        under a salt sharing those bits sends every row to one sub-bucket — the recursion
        rewrites and re-reads the whole bucket and changes nothing.
        """
        for depth in range(GRACE_DEPTH + 2):
            assert split_salt(depth) % 2 == 1
            assert split_salt(depth) != 0

    def test_it_differs_per_level_so_colliding_keys_spread(self) -> None:
        salts = [split_salt(d) for d in range(GRACE_DEPTH + 1)]
        assert len(set(salts)) == len(salts)

    def test_it_stays_inside_64_bits(self) -> None:
        assert all(0 < split_salt(d) < (1 << 64) for d in range(16))


class TestHowBigABucketIs:
    def test_it_is_the_uncompressed_size_not_the_size_on_disk(self, tmp_path) -> None:
        """Budgeting against `nbytes` lets a compressible bucket through the guard.

        This is the measure `combine_finalize` (or the window kernel, or the join) actually
        pays when the bucket is read back, and it is the trap the three breakers each used
        to restate in a comment.
        """
        store = TieredSpillStore(str(tmp_path / "spill"))
        # A low-entropy column: many repeated values, so the codec buys a lot.
        handle = store.spill([pa.record_batch({"k": ["hot"] * 5000, "v": [1] * 5000})])
        assert resident_bytes(handle) == handle.logical_nbytes
        assert resident_bytes(handle) >= handle.nbytes

    def test_an_unwritten_bucket_costs_nothing(self) -> None:
        assert resident_bytes(None) == 0

    def test_an_unbounded_envelope_never_asks_for_a_split(self, tmp_path, monkeypatch) -> None:
        """`spill_bucket_max_bytes <= 0` is "unbounded", and a bare `resident > envelope`
        comparison reads that as "everything is over budget" — an infinite split of every
        bucket down to the depth floor.

        Config validation currently refuses a non-positive value, so this is reached by
        patching the reader rather than the config. The guard stays because `sort.py`'s own
        bucket sizing takes the same precaution against the same value, and the two must not
        disagree about what "unbounded" means.
        """
        store = TieredSpillStore(str(tmp_path / "spill"))
        handle = store.spill([_batch(500)])
        monkeypatch.setattr(buckets_mod, "bucket_envelope", lambda: 0)
        assert not over_envelope(handle, 0)

    def test_the_depth_floor_stops_the_recursion(self, tmp_path) -> None:
        store = TieredSpillStore(str(tmp_path / "spill"))
        handle = store.spill([_batch(500)])
        with config_context(Config(memory=MemoryConfig(spill_bucket_max_bytes=1))):
            assert over_envelope(handle, 0)
            assert not over_envelope(handle, GRACE_DEPTH)


class TestBucketWriters:
    def test_a_bucket_that_gets_no_rows_opens_no_file(self, tmp_path) -> None:
        """The partition phase holds every open writer at once, so opening one per bucket
        rather than per non-empty bucket puts the file-descriptor cap under pressure the
        data never justified."""
        store = TieredSpillStore(str(tmp_path / "spill"))
        writers = BucketWriters(store, "bucket")
        writers.add([[_batch(10)], [], [_batch(5)]])
        handles = writers.close()
        assert set(handles) == {0, 2}

    def test_an_empty_batch_is_dropped_rather_than_written(self, tmp_path) -> None:
        store = TieredSpillStore(str(tmp_path / "spill"))
        writers = BucketWriters(store, "bucket")
        writers.add([[_batch(0)], [_batch(3)]])
        assert set(writers.close()) == {1}

    def test_the_dense_form_holds_a_place_for_every_bucket(self, tmp_path) -> None:
        """A range partition emits its buckets in key order and a co-partitioned join pairs
        bucket `b` with bucket `b`, so both index positionally and both need the gap."""
        store = TieredSpillStore(str(tmp_path / "spill"))
        writers = BucketWriters(store, "bucket")
        writers.add([[], [_batch(4)], []])
        handles = writers.close_dense(4)
        assert len(handles) == 4
        assert [h is None for h in handles] == [True, False, True, True]

    def test_the_rows_survive_the_round_trip(self, tmp_path) -> None:
        store = TieredSpillStore(str(tmp_path / "spill"))
        writers = BucketWriters(store, "bucket")
        writers.add([[_batch(10), _batch(10, 10)]])
        handle = writers.close()[0]
        read = read_reserved_bucket(store, handle)
        assert sum(b.num_rows for b in read) == 20
        assert read_reserved_bucket(store, None) is None


class TestTheScratchLifecycle:
    def test_a_directory_it_allocated_is_removed(self, tmp_path) -> None:
        import os

        with config_context(Config(memory=MemoryConfig(spill_dir=str(tmp_path)))):
            with spill_scratch("test_spill_", None) as store:
                store.spill([_batch(10)])
            entries = os.listdir(tmp_path)
        assert entries == [], f"scratch left behind: {entries}"

    def test_a_caller_owned_directory_is_left_alone(self, tmp_path) -> None:
        """An explicit `spill_dir` is the caller's — removing it would delete a directory the
        caller may be striping several queries across."""
        owned = tmp_path / "mine"
        owned.mkdir()
        with spill_scratch("test_spill_", str(owned)) as store:
            store.spill([_batch(10)])
        assert owned.exists()

    def test_the_store_is_cleaned_up_even_when_the_breaker_raises(self, tmp_path) -> None:
        """A partition phase abandoned by an exception leaves writers open and files on both
        tiers; `cleanup` is what aborts and removes them, so it cannot be conditional."""
        owned = tmp_path / "mine"
        owned.mkdir()
        captured = {}
        with (
            pytest.raises(RuntimeError, match="breaker failed"),
            spill_scratch("test_spill_", str(owned)) as store,
        ):
            captured["handle"] = store.spill([_batch(10)])
            captured["store"] = store
            raise RuntimeError("breaker failed")
        assert captured["store"].total_bytes == 0
        assert list(owned.iterdir()) == []


class TestTheOrderedBucketEnvelopeTracksTheMemoryBudget:
    """`spill_bucket_max_bytes` is a *ceiling*, never a licence to exceed the budget.

    It was the sole input to `_buckets_for_staged`, so the ordered split was sized against a
    fixed 128 MiB whatever the envelope actually was. Lowering `max_memory_bytes` to 1 MiB
    therefore still asked for 128 MiB buckets -- one bucket for a 7 MiB input -- which the
    reader then pulls back whole. The out-of-core path was not bounded by the number it
    exists to respect, and the failure surfaces as the engine refusing the bucket rather
    than as anything a result can show.

    `MemoryConfig.streaming_state_budget_bytes` resolves the same tension the same way, for
    the reason it states: a cap has to scale with the configured envelope rather than be a
    fixed magic number.
    """

    @staticmethod
    def _envelope(**memory):
        from batcher.dist.spill.buckets import bucket_envelope

        base = Config()
        scoped = base.replace(memory=dataclasses.replace(base.memory, **memory))
        with config_context(scoped):
            return bucket_envelope()

    def test_a_small_budget_shrinks_the_bucket_below_the_configured_ceiling(self) -> None:
        envelope = self._envelope(max_memory_bytes=1 << 20)
        assert envelope <= (1 << 20), "a bucket may not exceed the whole memory budget"
        assert envelope < Config().memory.spill_bucket_max_bytes

    def test_a_large_budget_leaves_the_configured_ceiling_in_charge(self) -> None:
        """The clamp is a `min`, so it never *raises* the bucket size."""
        assert self._envelope(max_memory_bytes=64 << 30) == Config().memory.spill_bucket_max_bytes

    def test_an_explicit_smaller_ceiling_still_wins(self) -> None:
        envelope = self._envelope(max_memory_bytes=64 << 30, spill_bucket_max_bytes=4 << 20)
        assert envelope == (4 << 20)

    def test_opting_out_of_any_memory_bound_keeps_the_configured_ceiling(self) -> None:
        """`unbounded_memory` means the user declined a budget; it must not zero the bucket."""
        assert self._envelope(unbounded_memory=True) == Config().memory.spill_bucket_max_bytes


class TestTheOrderedPathsRespectTheBucketEnvelope:
    """A bucket is read back *whole*, so no path may read one larger than the envelope.

    The aggregate, the join and the partitioned window each re-split an over-envelope bucket
    before reading it. The two **ordered** paths -- the out-of-core sort and the global window
    -- did not: `stage_and_partition` sizes the bucket count from the *total* staged bytes, so
    it bounds the average bucket and not the largest, and the sampled boundaries leave buckets
    uneven. Measured on 60k fat rows with a distinct key under a 1 MiB envelope, **33 of 65
    buckets exceeded it**, the worst by 2x -- so the path quietly used several times the budget
    it was given. The sort never raised, which is why nothing caught it: only the window kernel
    refuses an oversized bucket, and the sort just used the memory.

    Asserted on the bucket *sizes* rather than on the result, because the result was always
    correct -- it is the bound that was broken.
    """

    @staticmethod
    def _buckets_read(descending: bool, hot: bool) -> tuple[list[int], int]:
        """Sizes of every bucket the spilled sort reads, and the memory budget in force.

        Measured against the *budget*, not `bucket_envelope()`: the envelope is the thing
        under test, so asserting a bucket fits it is circular -- at HEAD the envelope was a
        fixed 128 MiB and every bucket "fitted" by definition. The budget is the number the
        user actually set, and it is the same on both sides of the fix.
        """
        import numpy as np

        import batcher as bt
        import batcher.dist.spill.buckets as bucket_mod

        rng = np.random.default_rng(3)
        n = 60_000
        keys = np.where(rng.random(n) < 0.7, 42.0, rng.random(n)) if hot else rng.random(n)
        table = pa.table(
            {
                "k": pa.array(keys.tolist(), pa.float64()),
                "s": pa.array(["z" * (i % 2000) for i in range(n)], pa.large_string()),
            }
        )
        seen: list[int] = []
        original = bucket_mod.read_reserved_bucket

        def watched(store, handle):
            if handle is not None:
                seen.append(bucket_mod.resident_bytes(handle))
            return original(store, handle)

        import batcher.dist.spill_breakers.sort as sort_mod

        bucket_mod.read_reserved_bucket = watched
        sort_mod.read_reserved_bucket = watched
        base = Config()
        tiny = base.replace(memory=dataclasses.replace(base.memory, max_memory_bytes=1 << 20))
        try:
            with config_context(tiny):
                budget = tiny.spill_budget_bytes()
                bt.from_arrow(table).sort("k", descending=descending).collect()
        finally:
            bucket_mod.read_reserved_bucket = original
            sort_mod.read_reserved_bucket = original
        return seen, budget

    @pytest.mark.parametrize("descending", [False, True])
    def test_a_splittable_key_never_leaves_a_bucket_over_the_envelope(self, descending) -> None:
        seen, budget = self._buckets_read(descending, hot=False)
        assert seen, "no bucket was read, so this proves nothing about the bound"
        over = [b for b in seen if b > budget]
        assert not over, (
            f"{len(over)} of {len(seen)} buckets exceeded the {budget:,}-byte memory budget "
            f"(worst {max(over):,}) — an ordered path must re-split before reading"
        )

    def test_one_hot_key_is_the_documented_floor_and_not_a_leak(self) -> None:
        """A single dominant key cannot be range-split — equal keys must be ordered together.

        Pinned so the recursion floor stays a *known* ceiling rather than becoming the excuse
        for the general case: at most one bucket may exceed, and it is the hot one.
        """
        seen, budget = self._buckets_read(descending=False, hot=True)
        over = [b for b in seen if b > budget]
        assert len(over) <= 1, f"{len(over)} buckets over the envelope; only the hot key may be"
