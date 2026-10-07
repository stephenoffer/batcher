"""A broadcast join's output bound is the node's to set, not a spill estimate's.

The worker's grant is its *spill threshold* -- an estimate of the plan's peak divided across
tasks -- and as the hard bound on a probe task's joined output it declined broadcasts that fit
easily: TPC-H q9 at SF1000 gave up at 0.1 GiB per node and ran a 28-minute single-worker
shuffle. The bound is now at least the node share, and what stops a broadcast from taking a
node down is the engine's live headroom reading.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist import flight_broadcast as fb

pytestmark = pytest.mark.unit

_GB = 1 << 30


class _Engine:
    def __init__(self, reading):
        self.reading = reading

    def memory_headroom(self):
        return self.reading


def test_a_small_grant_does_not_become_the_bound(monkeypatch):
    import psutil

    total = psutil.virtual_memory().total
    share = int(total * fb._OUTPUT_BUDGET_FRACTION)
    assert fb._output_budget(100 << 20) == max(100 << 20, share)
    assert fb._output_budget(total) == total  # a large grant still stands


def test_output_is_refused_past_the_bound_and_when_the_node_runs_short(monkeypatch):
    batch = pa.record_batch({"x": pa.array(range(1000), pa.int64())})
    monkeypatch.setattr(fb, "engine", lambda: _Engine((40 * _GB, 4 * _GB)))
    held = fb._charge(0, 1 << 40, [batch])  # positive control: roomy node, large bound
    assert held > 0
    with pytest.raises(fb.BroadcastOutputTooLarge):
        fb._charge(0, 1, [batch])  # over the bound
    monkeypatch.setattr(fb, "engine", lambda: _Engine((7 * _GB, 4 * _GB)))
    with pytest.raises(fb.BroadcastOutputTooLarge):
        fb._charge(0, 1 << 40, [batch])  # within twice the floor
    monkeypatch.setattr(fb, "engine", lambda: _Engine(None))
    assert fb._charge(0, 1 << 40, [batch]) > 0  # unreadable: the bound alone decides


def test_the_chunked_probe_join_matches_the_per_chunk_loop_and_meters_every_chunk(monkeypatch):
    """One engine call with the build prepared once returns what one call per chunk returned.

    The metrics must sum the chunks: one chunk's counts as the operator's would teach the
    learning loop a fraction of the truth. Forcing the not-chunkable answer is the control that
    the per-chunk loop is still there and agrees.
    """
    import json

    from batcher._internal.native import engine

    nat = engine()
    rows = 300_000
    probe = [
        pa.record_batch({"k": pa.array([(i * 7 + c) % 50_000 for i in range(rows)], pa.int64())})
        for c in range(3)
    ]
    build = [pa.record_batch({"bk": pa.array(range(0, 50_000, 2), pa.int64())})]
    probe_ir = json.dumps({"op": "scan", "source_id": 0})
    join_ir = json.dumps(
        {
            "op": "hash_join",
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
            "left_keys": ["k"],
            "right_keys": ["bk"],
            "join_type": "inner",
            "output": [{"side": "left", "name": "k", "alias": "k"}],
        }
    )
    monkeypatch.setattr(fb, "_PROBE_CHUNK_BYTES", 1 << 20)  # several chunks
    docs: list[str] = []
    got = fb.stream_probe_join(
        nat, probe_ir, join_ir, iter(probe), build, '{"parallelism": 4}', None, None, docs.append
    )
    want_rows = sum(1 for b in probe for k in b.column(0).to_pylist() if k % 2 == 0)
    assert sum(b.num_rows for b in got) == want_rows
    assert len(docs) == 1  # one metered call, not one per chunk
    ops = json.loads(docs[0])["ops"]
    (join,) = [op for op in ops if op["kind"] == "hash_join"]
    assert join["rows_out"] == want_rows
    # Every chunk counted: the probe scan read all three, though a runtime key filter keeps
    # some of its rows from reaching the join's own `rows_in`.
    assert max(op["rows_out"] for op in ops if op["kind"] == "scan") == 3 * rows

    class _NotChunkable:
        def __getattr__(self, name):
            return getattr(nat, name)

        def plan_chunkable(self, *_a):
            return False

    per_chunk = fb.stream_probe_join(
        _NotChunkable(), probe_ir, join_ir, iter(probe), build, '{"parallelism": 4}', None, None
    )
    key = lambda bs: sorted(k for b in bs for k in b.column(0).to_pylist())  # noqa: E731
    assert key(per_chunk) == key(got)


def test_a_small_spill_threshold_in_the_config_does_not_decline_the_chunked_join(monkeypatch):
    """The output bound, not the shipped spill threshold, decides when the chunked join gives up.

    The chunked engine call holds the tighter of its explicit budget and the config's
    `memory_budget_bytes`. TPC-H q10 at SF1000 shipped 16 MiB there and every probe task
    declined after 28 MB of output. The control: the same join under a bound below its output
    still declines, so the bound is live rather than switched off.
    """
    import json

    from batcher._internal.native import engine

    nat = engine()
    probe = [pa.record_batch({"k": pa.array(range(c, 400_000, 2), pa.int64())}) for c in range(2)]
    build = [pa.record_batch({"bk": pa.array(range(400_000), pa.int64())})]
    probe_ir = json.dumps({"op": "scan", "source_id": 0})
    join_ir = json.dumps(
        {
            "op": "hash_join",
            "left": {"op": "scan", "source_id": 0},
            "right": {"op": "scan", "source_id": 1},
            "left_keys": ["k"],
            "right_keys": ["bk"],
            "join_type": "inner",
            "output": [{"side": "left", "name": "k", "alias": "k"}],
        }
    )
    tiny = json.dumps({"parallelism": 2, "memory_budget_bytes": 64 << 10})
    got = fb.stream_probe_join(
        nat, probe_ir, join_ir, iter(probe), build, tiny, None, None, output_budget=1 << 30
    )
    assert sum(b.num_rows for b in got) == 400_000
    monkeypatch.setattr(fb, "_OUTPUT_BUDGET_FRACTION", 0.0)  # no node-share floor under it
    with pytest.raises(fb.BroadcastOutputTooLarge):
        fb.stream_probe_join(
            nat, probe_ir, join_ir, iter(probe), build, tiny, None, None, output_budget=64 << 10
        )


def test_the_holding_config_only_ever_raises_the_threshold():
    import json

    cfg = '{"memory_budget_bytes": 100, "parallelism": 3}'
    assert json.loads(fb._holding_config(cfg, 1000))["memory_budget_bytes"] == 1000
    assert fb._holding_config(cfg, 50) == cfg  # a tighter output bound leaves the config alone
    unbounded = '{"memory_budget_bytes": 0}'
    assert fb._holding_config(unbounded, 1000) == unbounded  # unbounded stays unbounded
    assert fb._holding_config(cfg, 0) == cfg
