"""The reduce side's critical path is logarithmic in the mapper count, not linear.

`dist/reduction/tree.py` is the arithmetic every shuffle's reduce shares: how many levels a
bounded-fan-in fold needs and which partials form each chunk. It is worth pinning
separately from any operator because the claim it encodes is asymptotic — a reduce that
folds its inputs in a line does Θ(W) sequential combines, so the reduce phase *grows* as
workers are added and the query stops scaling — and an asymptotic claim is exactly the kind
a single end-to-end test cannot see.

The disk aggregate's combiner tree (`executors/aggregate._tree_combine_buckets`) is checked
here too, with a stubbed native combine, so the orchestration is verified without a cluster:
every mapper partial reaches exactly one chunk, no task reads more than `fan_in` of them,
and the merged value equals the flat fold's.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from batcher.dist.reduction import chunks, reduce_levels, tree_reduce

pytestmark = pytest.mark.unit


def test_chunks_partitions_exactly_once():
    items = list(range(17))
    got = chunks(items, 5)
    assert [len(c) for c in got] == [5, 5, 5, 2]
    assert [x for c in got for x in c] == items


def test_chunks_of_empty_is_empty():
    assert chunks([], 4) == []


@pytest.mark.parametrize("size", [0, -3])
def test_chunks_never_loops_forever_on_a_nonpositive_size(size):
    # A caller that computes its fan-out can hand this a 0; a step of 0 would spin.
    assert [list(c) for c in chunks([1, 2], size)] == [[1], [2]]


@pytest.mark.parametrize(
    ("n", "fan_in", "expected"),
    [
        (0, 4, 0),
        (1, 4, 0),
        (4, 4, 0),  # already one combine's worth
        (5, 4, 1),
        (16, 4, 1),  # 16 -> 4, then the caller's finalize
        (17, 4, 2),  # 17 -> 5 -> 2
        (1000, 4, 4),
        (1000, 8, 3),
    ],
)
def test_reduce_levels_matches_the_closed_form(n, fan_in, expected):
    assert reduce_levels(n, fan_in) == expected


@pytest.mark.parametrize("n", [2, 3, 8, 9, 64, 65, 4096, 4097])
@pytest.mark.parametrize("fan_in", [2, 4, 8, 32])
def test_reduce_levels_agrees_with_the_loop_that_runs(n, fan_in):
    """The count is computed by division precisely so it cannot disagree with the frontier
    loop; a `log` would be off by one at exact powers of the base."""
    calls: list[int] = []
    tree_reduce(
        list(range(n)),
        lambda chunk, level, _i: (calls.append(level), chunk[0])[1],
        fan_in,
    )
    observed = 1 + max(calls) if calls else 0
    assert observed == reduce_levels(n, fan_in)


@pytest.mark.parametrize("fan_in", [2, 3, 8])
def test_reduce_levels_is_logarithmic_not_linear(fan_in):
    """The whole point: doubling the cluster adds a bounded number of levels, it does not
    double the reduce's critical path."""
    for n in (128, 256, 512, 1024, 2048):
        assert reduce_levels(2 * n, fan_in) - reduce_levels(n, fan_in) <= 1
    assert reduce_levels(10**6, fan_in) <= math.ceil(math.log(10**6, fan_in))


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 100, 257])
@pytest.mark.parametrize("fan_in", [2, 4, 8])
def test_tree_reduce_leaves_a_frontier_one_combine_can_finish(n, fan_in):
    frontier = tree_reduce(list(range(n)), lambda chunk, _l, _i: sum(chunk), fan_in)
    assert 1 <= len(frontier) <= fan_in


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 100, 257, 1000])
@pytest.mark.parametrize("fan_in", [2, 3, 8, 32])
def test_tree_reduce_preserves_the_total_under_any_bracketing(n, fan_in):
    """Sum stands in for `combine`: associative and commutative, so every bracketing of the
    same partials is the same state. This is why `fan_in` can never trade correctness."""
    frontier = tree_reduce(list(range(n)), lambda chunk, _l, _i: sum(chunk), fan_in)
    assert sum(frontier) == n * (n - 1) // 2


@pytest.mark.parametrize("fan_in", [2, 4, 8])
def test_tree_reduce_never_hands_a_node_more_than_fan_in(fan_in):
    widths: list[int] = []
    tree_reduce(list(range(500)), lambda c, _l, _i: (widths.append(len(c)), sum(c))[1], fan_in)
    assert widths and max(widths) <= fan_in


def test_tree_reduce_is_a_no_op_below_the_threshold():
    """A small shuffle pays nothing for the tree existing — no combine is invoked at all."""
    called = False

    def combine(chunk, _l, _i):
        nonlocal called
        called = True
        return sum(chunk)

    assert tree_reduce([1, 2, 3], combine, 8) == [1, 2, 3]
    assert not called


def test_tree_reduce_passes_a_lone_straggler_through_uncombined():
    """At `n = f^k + 1` every level has exactly one chunk of one. Combining it with nothing
    would be a full read and write to reproduce bytes that already exist."""
    widths: list[int] = []
    tree_reduce(list(range(9)), lambda c, _l, _i: (widths.append(len(c)), sum(c))[1], 4)
    assert widths == [4, 4]  # the 9th element rode along untouched


def test_fan_in_is_floored_at_two_so_the_frontier_always_shrinks():
    assert tree_reduce([1, 2, 3, 4], lambda c, _l, _i: sum(c), 1) == [3, 7]


# --- the disk aggregate's combiner tree ---------------------------------------------


class _FakeRef:
    """Stands in for a Ray ObjectRef: the barrier stub reads `.value` instead of blocking."""

    def __init__(self, value):
        self.value = value


class _FakeCombineTask:
    """Records every chunk it was handed and writes a merged 'file' whose name encodes the
    sum of its inputs, so the driver's bookkeeping can be checked without a cluster."""

    def __init__(self, files: dict[str, int]) -> None:
        self.files = files
        self.chunks: list[list[str]] = []

    def remote(self, _gk, _aj, input_paths, _work_dir, out_name, _cfg=""):
        self.chunks.append(list(input_paths))
        total = sum(self.files[p] for p in input_paths)
        if total == 0:
            return _FakeRef([])
        self.files[out_name] = total
        return _FakeRef([out_name])


@pytest.fixture
def wired_tree(monkeypatch):
    from batcher.carbonite import resilience
    from batcher.dist.executors import aggregate

    monkeypatch.setattr(
        resilience, "gather_with_backups", lambda refs, _s, _p, stage="": [r.value for r in refs]
    )
    return aggregate


def _run(aggregate, n_mappers, n_reducers, fan_in, values=None):
    """Drive `_tree_combine_buckets` over `n_mappers` mappers, each publishing one file per
    bucket holding `values[m]` (default 1) units of state."""
    import dataclasses

    from batcher.config import active_config, config_context

    files = {}
    shuffle_paths = []
    for m in range(n_mappers):
        row = []
        for r in range(n_reducers):
            name = f"m{m}_r{r}.arrow"
            files[name] = 1 if values is None else values[m]
            row.append(name)
        shuffle_paths.append(row)
    task = _FakeCombineTask(files)
    base = active_config()
    cfg = base.replace(flow_control=dataclasses.replace(base.flow_control, shuffle_fan_in=fan_in))
    with config_context(cfg):
        import unittest.mock as mock

        with mock.patch.object(aggregate, "_combine_task", task):
            out = aggregate._tree_combine_buckets(
                shuffle_paths, n_reducers, "gk", "aj", "/tmp/wd", "", None
            )
    return out, task, files


def test_disk_tree_is_skipped_when_the_mappers_already_fit(wired_tree):
    out, task, _files = _run(wired_tree, n_mappers=8, n_reducers=3, fan_in=8)
    assert task.chunks == []
    assert [len(b) for b in out] == [8, 8, 8]


def test_disk_tree_waits_until_a_level_earns_its_tasks(wired_tree):
    """Just past the fan-in the tree would spend a task, a write and a read per chunk to save
    about as many combines. It engages at `_TREE_MIN_MAPPERS x fan_in`, where the saving is
    several times the cost, and leaves the flat fold alone below that."""
    from batcher.dist.executors.aggregate import _TREE_MIN_MAPPERS

    fan_in = 4
    below = _TREE_MIN_MAPPERS * fan_in - 1
    out, task, _files = _run(wired_tree, n_mappers=below, n_reducers=2, fan_in=fan_in)
    assert task.chunks == []
    assert [len(b) for b in out] == [below, below]

    out, task, _files = _run(wired_tree, n_mappers=below + 1, n_reducers=2, fan_in=fan_in)
    assert task.chunks and all(len(b) <= fan_in for b in out)


def test_disk_tree_bounds_every_reducer_at_the_fan_in(wired_tree):
    out, _task, _files = _run(wired_tree, n_mappers=200, n_reducers=4, fan_in=8)
    assert all(len(b) <= 8 for b in out)


def test_disk_tree_never_reads_more_than_fan_in_per_task(wired_tree):
    _out, task, _files = _run(wired_tree, n_mappers=200, n_reducers=4, fan_in=8)
    assert task.chunks and max(len(c) for c in task.chunks) <= 8


def test_disk_tree_conserves_every_mapper_partial(wired_tree):
    """Each mapper contributes one unit per bucket; the surviving frontier must still add
    up to exactly that. A dropped or duplicated chunk is a wrong aggregate."""
    out, _task, files = _run(wired_tree, n_mappers=100, n_reducers=3, fan_in=4)
    for bucket in out:
        assert sum(files[p] for p in bucket) == 100


def test_disk_tree_drops_a_chunk_whose_inputs_were_all_empty(wired_tree):
    """An all-empty chunk writes no file, so the reducer is handed one fewer input rather
    than a path that does not exist."""
    values = [0] * 40 + [1] * 60
    out, _task, files = _run(wired_tree, n_mappers=100, n_reducers=1, fan_in=4, values=values)
    assert all(files[p] > 0 for p in out[0])
    assert sum(files[p] for p in out[0]) == 60


def test_disk_tree_handles_the_keyless_single_reducer(wired_tree):
    """The global aggregate has one bucket, so the flat fold was a `W`-long line on one node
    with the rest of the cluster idle — the shape the tree helps most."""
    out, task, files = _run(wired_tree, n_mappers=512, n_reducers=1, fan_in=8)
    assert len(out) == 1 and len(out[0]) <= 8
    assert sum(files[p] for p in out[0]) == 512
    # 512 -> 64 -> 8: two interior levels, matching the closed form.
    assert reduce_levels(512, 8) == 2
    assert max(len(c) for c in task.chunks) <= 8


# --- range-partition sample sizing --------------------------------------------------


def test_sample_grid_holds_the_samples_per_bucket_constant():
    """The quantity imbalance tracks is pooled samples per bucket (`g x S / P`), not the grid
    size — so the grid has to scale with the bucket-to-sampler ratio to hold it."""
    from batcher.dist.executors.partition_io import sample_probs

    for ratio in (1, 2, 4, 8):
        g = len(sample_probs(8 * ratio, 8)) - 1
        assert g * 8 / (8 * ratio) == 64


def test_sample_grid_never_drops_below_the_floor():
    """Fewer buckets than samplers already has samples to spare; it must not sample more
    coarsely than the historical constant did."""
    from batcher.dist.executors.partition_io import SAMPLE_PROBS, sample_probs

    assert sample_probs(1, 64) == SAMPLE_PROBS
    assert sample_probs(4, 64) == SAMPLE_PROBS


def test_sample_grid_is_capped_so_the_driver_merge_stays_bounded():
    """`samplers x probes` is a driver-side sort; an uncapped grid would make a wide shuffle
    pay for its own boundaries."""
    from batcher.dist.executors.partition_io import sample_probs

    assert len(sample_probs(10**6, 1)) - 1 == 1024


def test_sample_grid_is_ascending_and_spans_the_unit_interval():
    from batcher.dist.executors.partition_io import sample_probs

    for buckets, samplers in ((8, 8), (64, 8), (5, 3), (10**6, 1)):
        probs = sample_probs(buckets, samplers)
        assert probs[0] == 0.0 and probs[-1] == 1.0
        assert all(b > a for a, b in itertools.pairwise(probs))


@pytest.mark.parametrize(("samplers", "buckets"), [(8, 32), (16, 128), (32, 256)])
def test_the_derived_grid_actually_levels_the_buckets(samplers, buckets):
    """The claim, measured rather than asserted. A fixed 33-probe grid leaves the busiest
    bucket carrying 2-8x the mean once the cut is wider than the sampler count, and every
    other reducer waits on it; the derived grid holds it inside a tenth.

    Models the real pipeline exactly — per-sampler quantiles, `merge_boundaries`, then
    `searchsorted(side="right")`, which is what `bucketize` does — over a heavy-tailed key,
    because a uniform one hides the failure entirely.
    """
    import numpy as np

    from batcher.dist.executors.partition_io import merge_boundaries, sample_probs

    def busiest(probs):
        rng = np.random.default_rng(0)
        vals = rng.lognormal(0, 1.4, 400_000)
        grids = [(list(np.quantile(p, probs)), len(p)) for p in np.array_split(vals, samplers)]
        edges = np.asarray(merge_boundaries(grids, buckets))
        counts = np.bincount(np.searchsorted(edges, vals, side="right"), minlength=buckets)
        return counts.max() / (len(vals) / buckets)

    fixed = busiest([i / 32 for i in range(33)])
    derived = busiest(sample_probs(buckets, samplers))
    assert fixed > 1.5, "the fixed grid was supposed to be the broken arm"
    assert derived < 1.15
    assert derived < fixed / 1.5


def test_disk_tree_stops_when_a_level_shrinks_nothing(wired_tree, monkeypatch):
    """Every chunk declining on memory leaves the frontier exactly as wide, and a further
    level would decline identically — so the loop must stop rather than spin. The bucket
    reaches the reducer at full width, which is the out-of-core fold's input."""

    class _Decline:
        def remote(self, _gk, _aj, input_paths, _wd, _name, _cfg=""):
            return _FakeRef(list(input_paths))

    import unittest.mock as mock

    from batcher.config import active_config, config_context

    files = {f"m{m}_r0.arrow": 1 for m in range(64)}
    shuffle_paths = [[f"m{m}_r0.arrow"] for m in range(64)]
    base = active_config()
    cfg = base.replace(flow_control=dataclasses.replace(base.flow_control, shuffle_fan_in=4))
    with config_context(cfg), mock.patch.object(wired_tree, "_combine_task", _Decline()):
        out = wired_tree._tree_combine_buckets(shuffle_paths, 1, "gk", "aj", "/tmp/wd", "", None)
    assert len(out[0]) == 64
    assert sum(files[p] for p in out[0]) == 64


# --- staging the aggregate over a distributed dedup ----------------------------------


class _Intermediate:
    """A partitioned shuffle output that knows exactly how many rows it wrote."""

    def __init__(self, rows):
        self._rows = rows

    def row_count(self):
        return self._rows


def test_a_small_dedup_result_is_not_restaged():
    """`COUNT(DISTINCT status)` over six values must not pay a second map/shuffle/reduce to
    count six rows. The dedup already measured the intermediate, so this is a lookup."""
    from batcher.dist.executor import _STAGED_DISTINCT_ROWS, _too_small_to_restage

    assert _too_small_to_restage(_Intermediate(6))
    assert _too_small_to_restage(_Intermediate(_STAGED_DISTINCT_ROWS - 1))


def test_a_large_dedup_result_is_restaged():
    """The case the staging exists for: a 15M-key `COUNT(DISTINCT)` whose driver-side fold
    is Θ(cardinality) and gets no faster however many workers the query is given."""
    from batcher.dist.executor import _STAGED_DISTINCT_ROWS, _too_small_to_restage

    assert not _too_small_to_restage(_Intermediate(_STAGED_DISTINCT_ROWS))
    assert not _too_small_to_restage(_Intermediate(15_000_000))


def test_an_intermediate_that_cannot_count_is_treated_as_large():
    """Restaging something small wastes one stage; centralizing something large is the serial
    fraction. With no measurement, take the bounded mistake."""
    from batcher.dist.executor import _too_small_to_restage

    assert not _too_small_to_restage(_Intermediate(None))
    assert not _too_small_to_restage(object())


# --- the scaling claim, stated mechanically ------------------------------------------


@pytest.mark.parametrize("mappers", [32, 64, 128, 512, 2048, 8192])
def test_a_reducers_fold_length_does_not_grow_with_the_cluster(wired_tree, mappers):
    """This is the property the whole rebracketing exists for, and the one a correctness
    test cannot see: however many mappers there are, a reducer folds at most `fan_in` of
    them. Before the tree that number *was* the mapper count, so the reduce phase grew as
    the cluster did while the map phase it followed shrank."""
    out, _task, _files = _run(wired_tree, n_mappers=mappers, n_reducers=2, fan_in=8)
    assert all(len(b) <= 8 for b in out)


@pytest.mark.parametrize("mappers", [32, 64, 128, 512, 2048, 8192])
def test_the_disk_tree_spends_a_logarithmic_number_of_levels(wired_tree, mappers):
    """Doubling the fleet adds at most one level, never doubles the work in front of the
    reducer. Counted from the interior file names, which encode their level."""
    _out, _task, files = _run(wired_tree, n_mappers=mappers, n_reducers=1, fan_in=8)
    levels = {name.split("_")[0] for name in files if name.startswith("c")}
    assert len(levels) == reduce_levels(mappers, 8)
    assert len(levels) <= math.ceil(math.log(mappers, 8))


@pytest.mark.parametrize("mappers", [32, 64, 128, 512, 2048])
def test_every_mapper_partial_survives_at_every_cluster_size(wired_tree, mappers):
    """Scaling is worthless if the rebracketing loses a partial on the way. Each mapper
    contributes one unit per bucket and the surviving frontier must still add to exactly
    that."""
    out, _task, files = _run(wired_tree, n_mappers=mappers, n_reducers=3, fan_in=8)
    for bucket in out:
        assert sum(files[p] for p in bucket) == mappers
