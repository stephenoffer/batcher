"""Learned + data-adaptive sizing for the stage-overlapped streaming UDF path.

Three scheduling knobs, none of which can change a result (a chunk only shards a morsel, a
prefetch depth only reorders when a chunk is read):

* the GPU sub-batch row cap, seeded from a model's learned VRAM-safe size (`_learned_gpu_cap`);
* a CPU stage's byte-adaptive chunk, which shrinks on wide post-decode rows (`_cpu_batch_rows`);
* the source-read prefetch depth, deepened for a source measured as slow (`_learned_read_depth`).

The invariance gate runs a two-stage chain (a CPU stage feeding a GPU-tagged stage — the tag makes
it stream-eligible; the `fn` itself runs on the CPU here) with the learned GPU cap seeded tiny vs
cold, and asserts byte-identical output.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher.config import Config, config_context
from batcher.core import default_hub
from batcher.core.udf import stream

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pinned_config():
    """Run each sizing test under a fresh default `Config`, hermetic against any config a
    prior (differential-suite) test leaked — these tests assert *result-invariance* under
    a known execution config, so an external morsel/execution knob must not perturb them."""
    with config_context(Config()):
        yield


def _stage1(b: pa.RecordBatch) -> pa.RecordBatch:
    return b.append_column("y", pc.add(b.column("x"), 1))


def _stage2(b: pa.RecordBatch) -> pa.RecordBatch:
    return b.append_column("z", pc.multiply(b.column("x"), 2))


# --- _gpu_batch_rows: data-width adaptive, capped -----------------------------------------


def test_gpu_batch_rows_narrow_fills_to_cap():
    narrow = pa.record_batch({"x": pa.array(list(range(1000)), pa.int64())})
    assert stream._gpu_batch_rows(narrow, row_cap=256) == 256  # narrow rows fill the cap


def test_gpu_batch_rows_wide_shrinks_below_cap():
    # A wide per-row payload (~1 MB/row) must batch far fewer than the cap to stay in budget.
    wide = pa.record_batch({"x": pa.array([b"\0" * (1 << 20)] * 200)})
    got = stream._gpu_batch_rows(wide, row_cap=256)
    assert 1 <= got < 256


def test_gpu_batch_rows_respects_learned_cap():
    narrow = pa.record_batch({"x": pa.array(list(range(1000)), pa.int64())})
    assert stream._gpu_batch_rows(narrow, row_cap=32) == 32  # learned cap caps the fill


# --- _learned_gpu_cap: cold default vs seeded --------------------------------------------


def test_learned_gpu_cap_cold_is_config_default():
    op = bt.from_arrow(pa.table({"x": [1]})).ml.map_batches(_stage2, num_gpus=1)._plan
    assert stream._learned_gpu_cap(op) == stream._GPU_STREAM_BATCH_ROWS


def test_learned_gpu_cap_seeded_caps_down():
    op = bt.from_arrow(pa.table({"x": [1]})).ml.map_batches(_stage2, num_gpus=1)._plan
    sig = stream._stage_sig(op)
    default_hub().put_keyed_param(stream._GPU_BATCH_NS, sig, {"ema": 40.0})
    assert stream._learned_gpu_cap(op) == 40  # min(config cap, learned)


# --- _cpu_batch_rows: byte-adaptive shrink on wide rows ----------------------------------


def test_cpu_batch_rows_narrow_keeps_morsel():
    narrow = pa.record_batch({"x": pa.array(list(range(1000)), pa.int64())})
    assert stream._cpu_batch_rows(narrow, morsel=16_384) == 16_384


def test_cpu_batch_rows_wide_shrinks():
    wide = pa.record_batch({"x": pa.array([b"\0" * (4 << 20)] * 100)})  # ~4 MB/row
    assert stream._cpu_batch_rows(wide, morsel=16_384) < 16_384


# --- _learned_read_depth: slow source deepens the prefetch -------------------------------


def test_read_depth_cold_is_base():
    src = bt.from_arrow(pa.table({"x": [1, 2, 3]}))._sources[0]
    assert stream._learned_read_depth(src) == stream._STREAM_PREFETCH_DEPTH


def test_read_depth_deepens_for_a_slow_source():
    src = bt.from_arrow(pa.table({"x": [1, 2, 3]}))._sources[0]
    default_hub().put_keyed_param(stream._SCAN_TPUT_NS, src.identity(), {"ema": 1000.0})  # slow
    assert stream._learned_read_depth(src) > stream._STREAM_PREFETCH_DEPTH


def test_ema_round_trip():
    stream._fold_ema(stream._SCAN_TPUT_NS, "k", 100.0)
    assert stream._read_ema(stream._SCAN_TPUT_NS, "k") == 100.0
    assert stream._read_ema(stream._SCAN_TPUT_NS, None) is None


# --- result-invariance: the streaming GPU chunk size never changes the output ------------


def _run_chain(t: pa.Table):
    return bt.from_arrow(t).ml.map_batches(_stage1).ml.map_batches(_stage2, num_gpus=1).to_pydict()


def test_streaming_gpu_batch_size_is_result_invariant():
    t = pa.table({"x": list(range(5000))})
    sig = stream._stage_sig(bt.from_arrow(t).ml.map_batches(_stage2, num_gpus=1)._plan)

    cold = _run_chain(t)  # config-default GPU chunk
    default_hub().put_keyed_param(stream._GPU_BATCH_NS, sig, {"ema": 8.0})  # tiny learned chunk
    warm = _run_chain(t)

    assert cold == warm  # chunk size only shards — byte-identical result
    assert cold["z"] == [v * 2 for v in range(5000)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_the_learned_gpu_cap_can_recover_and_is_not_a_one_way_ratchet():
    """The folded size must measure the data, not the cap it just produced.

    The applied GPU size is `min(learned_cap, by_bytes)`, which is `<= learned_cap` by
    construction. Folding *that* made the EMA its own input, so every run could only ever
    contribute a value at or below the prior EMA — a monotonically non-increasing ratchet
    with `int()` truncation drifting it further down and no path back up. One run over wide
    rows (large frames) therefore shrank a model's GPU batch permanently, on every later
    run, even over narrow rows, sliding toward `_GPU_STREAM_BATCH_MIN`.

    Sizing the folded observation against the *config* cap keeps it a property of the data,
    so a narrow-row run recovers. Batch size never changes what the model computes.
    """
    import pyarrow as pa

    cap = stream._GPU_STREAM_BATCH_ROWS
    sig = "test.ratchet_model"

    def morsel(per_row_bytes: int, rows: int) -> pa.RecordBatch:
        payload = b"\0" * per_row_bytes
        return pa.RecordBatch.from_pydict({"blob": [payload] * rows})

    # Wide enough that the byte budget, not the row cap, binds (3 MB frames); few rows so
    # the fixture stays small. Narrow rows let the row cap bind instead.
    wide, narrow = morsel(3_000_000, 4), morsel(64, 512)
    # A wide morsel really does size down, and a narrow one really does size up — otherwise
    # this test would pass on a broken implementation that ignored width entirely.
    assert stream._gpu_batch_rows(wide, cap) < stream._gpu_batch_rows(narrow, cap)

    # Run 1 is all wide rows: the learned cap settles small.
    default_hub().put_keyed_param(stream._GPU_BATCH_NS, sig, {})
    stream._fold_ema(stream._GPU_BATCH_NS, sig, float(stream._gpu_batch_rows(wide, cap)))
    after_wide = stream._read_ema(stream._GPU_BATCH_NS, sig)
    assert after_wide is not None

    # Run 2 is all narrow rows. The observation is sized against the CONFIG cap, so it
    # reports what the narrow data permits and the EMA climbs back.
    learned_cap = min(cap, int(after_wide))
    stream._fold_ema(stream._GPU_BATCH_NS, sig, float(stream._gpu_batch_rows(narrow, cap)))
    assert stream._read_ema(stream._GPU_BATCH_NS, sig) > after_wide  # recovered

    # Sizing against the previously *learned* cap — what the old code folded — could not
    # have exceeded it, which is precisely why the EMA could only ever fall.
    assert stream._gpu_batch_rows(narrow, learned_cap) <= learned_cap
