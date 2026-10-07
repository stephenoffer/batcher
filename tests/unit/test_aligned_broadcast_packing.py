"""Held broadcasts travel compressed past a size, and come back exactly as they left."""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.dist.executors.aligned import run as aligned_run
from batcher.dist.executors.aligned import transfer

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_decode_cache(monkeypatch):
    """Each test's runs start with nothing decoded and no orphaned copies, as a new run's do."""
    monkeypatch.setattr(transfer, "_DECODED", {})
    monkeypatch.setattr(transfer, "_ORPHANS", [])


def test_large_broadcasts_round_trip_compressed(monkeypatch):
    monkeypatch.setattr(transfer, "PACK_BYTES", 1)
    table = pa.table({"k": pa.array(range(100_000), pa.int64()), "s": ["abc", None] * 50_000})
    empty = pa.RecordBatch.from_pylist([], schema=pa.schema([("a", pa.float64())]))
    held = {3: table.to_batches(max_chunksize=4096), 4: [empty]}
    packed = transfer.pack_held(held, "run")
    assert all(isinstance(v, transfer.Packed) for v in packed.values())
    # Cut into several streams (compressed in parallel), and reassembled in order.
    assert len(packed[3].parts) > 1
    back = transfer.unpack_held(packed)
    assert pa.Table.from_batches(back[3]).equals(table)
    assert pa.Table.from_batches(back[4], schema=empty.schema).num_rows == 0


def test_one_large_batch_is_compressed_in_several_parts(monkeypatch):
    """A hoisted broadcast is often one batch (a finalized aggregate): sliced, not one stream."""
    monkeypatch.setattr(transfer, "PACK_BYTES", 1)
    table = pa.table({"k": pa.array(range(200_000), pa.int64())}).combine_chunks()
    packed = transfer.pack_held({0: table.to_batches()}, "run")
    assert len(table.to_batches()) == 1 and len(packed[0].parts) > 1
    assert pa.Table.from_batches(transfer.unpack_held(packed)[0]).equals(table)


def test_small_broadcasts_are_left_as_they_are():
    batch = pa.RecordBatch.from_pylist([{"k": 1}])
    held = {0: [batch]}
    assert transfer.pack_held(held, "run") is held
    assert transfer.unpack_held(held) == held


def test_units_never_ask_for_more_cores_than_a_node_has(monkeypatch):
    """A unit asking for more CPUs than any node holds is never placed, and the gather waits
    forever: a test cluster of 2-CPU nodes hung on the default 8-CPU unit."""
    import sys
    import types

    nodes = [{"Alive": True, "Resources": {"CPU": 2.0}} for _ in range(4)]
    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(nodes=lambda: nodes))
    assert aligned_run._unit_slots(4) == (2, 4)
    nodes[:] = [
        {"Alive": True, "Resources": {"CPU": 16.0}},
        {"Alive": True, "Resources": {"CPU": 8.0}},
    ]
    assert aligned_run._unit_slots(2) == (8, 3)
    # A wide task (large held broadcasts) takes a whole node; a node narrower than it runs none.
    assert aligned_run._unit_slots(2, 16) == (16, 1)


def test_a_node_read_broadcast_is_evaluated_once_per_process(monkeypatch):
    """Every unit stream on a node joins one result, and the next run's recipe replaces it."""
    from batcher.dist.executors.aligned import local

    calls = []

    def evaluate(recipe, tables):
        calls.append(recipe.key)
        return [pa.record_batch({"k": [len(calls)]})]

    monkeypatch.setattr(local, "_evaluate", evaluate)
    monkeypatch.setattr(local, "_RESULTS", {})

    def recipe(key: str) -> local.LocalBroadcast:
        return local.LocalBroadcast(key, "{}", {}, {}, [], "{}")

    table = [pa.record_batch({"x": [1]})]
    first = local.resolve_local({0: table, 1: recipe("run1:1")})
    again = local.resolve_local({0: table, 1: recipe("run1:1")})
    assert calls == ["run1:1"] and first[1] is again[1] and first[0] is table
    local.resolve_local({1: recipe("run2:1")})
    assert calls == ["run1:1", "run2:1"] and list(local._RESULTS) == ["run2:1"]


def test_a_recipe_travels_beside_packed_rows(monkeypatch):
    from batcher.dist.executors.aligned.local import LocalBroadcast

    monkeypatch.setattr(transfer, "PACK_BYTES", 1)
    recipe = LocalBroadcast("run:5", "{}", {}, {}, [], "{}")
    rows = [pa.record_batch({"k": list(range(1000))})]
    packed = transfer.pack_held({4: rows, 5: recipe}, "run")
    assert isinstance(packed[4], transfer.Packed) and packed[5] is recipe
    back = transfer.unpack_held(packed)
    assert back[5] is recipe and pa.Table.from_batches(back[4]).equals(pa.Table.from_batches(rows))


def test_a_packed_broadcast_is_decoded_once_per_process(monkeypatch):
    monkeypatch.setattr(transfer, "PACK_BYTES", 1)
    rows = [pa.record_batch({"k": list(range(1000))})]
    first = transfer.pack_held({4: rows}, "run1")
    assert transfer.unpack_held(first)[4] is transfer.unpack_held(first)[4]
    transfer.unpack_held(transfer.pack_held({4: rows}, "run2"))
    assert list(transfer._DECODED) == ["run2:4"]


def test_pulled_units_land_in_order_and_a_slow_worker_takes_fewer(monkeypatch):
    """Three workers, one four times slower: it is handed fewer units, and every unit's
    result lands at its own index."""
    import ray

    cost = {0: 1.0, 1: 1.0, 2: 4.0}
    clock, finish, taken = [0.0], {}, {0: 0, 1: 0, 2: 0}

    def submit(worker, units):
        ref = object()
        started = max([clock[0], *(t for (w, t) in finish.values() if w == worker)])
        finish[ref] = (worker, started + cost[worker] * len(units))
        taken[worker] += len(units)
        return ref, units

    refs = {}

    def fake_submit(worker, units):
        ref, units = submit(worker, units)
        refs[ref] = [([], u, (0.0, 0.0, 0, 0.0, 0.0)) for u in units]
        return ref

    def wait(pending, num_returns, timeout=None):
        ref = min(pending, key=lambda r: finish[r][1])
        clock[0] = finish[ref][1]
        return [ref], [r for r in pending if r is not ref]

    monkeypatch.setattr(ray, "wait", wait)
    monkeypatch.setattr(ray, "get", lambda ref: refs[ref])
    calls = list(range(24))
    out = transfer.pull_units(calls, 3, 1, fake_submit)
    assert [unit for _rows, unit, _t in out] == calls
    assert taken[2] < taken[0] and taken[2] < taken[1]


def test_a_straggling_call_is_copied_and_the_first_result_kept(monkeypatch):
    """Every unit handed out, a free worker runs a copy of the call far past the median; the
    copy lands first, the straggler is dropped, and each unit's result is in its place."""
    import types

    import ray

    clock, finish, runs = [0.0], {}, {}
    slow_unit = 5

    def submit(worker, units):
        ref = object()
        busy = [t for (w, t) in finish.values() if w == worker and t > clock[0]]
        cost = 50.0 if slow_unit in units and not runs.get(slow_unit) else 1.0
        for u in units:
            runs[u] = runs.get(u, 0) + 1
        finish[ref] = (worker, max([clock[0], *busy]) + cost * len(units))
        results[ref] = [([], u, (0.0, 0.0, 0, 0.0, 0.0)) for u in units]
        return ref

    results = {}

    def wait(pending, num_returns, timeout=None):
        ref = min(pending, key=lambda r: finish[r][1])
        if timeout is not None and finish[ref][1] > clock[0] + timeout:
            clock[0] += timeout
            return [], list(pending)
        clock[0] = finish[ref][1]
        return [ref], [r for r in pending if r is not ref]

    monkeypatch.setattr(ray, "wait", wait)
    monkeypatch.setattr(ray, "get", lambda ref: results[ref])
    monkeypatch.setattr(
        transfer, "time", types.SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0])
    )
    calls = list(range(12))
    out = transfer.pull_units(calls, 4, 1, submit)
    assert [unit for _rows, unit, _t in out] == calls
    # Positive control: the slow unit ran twice, and the query did not wait out its 50 s.
    assert runs[slow_unit] == 2 and clock[0] < 50.0


def test_a_worker_still_running_an_orphaned_copy_is_given_fewer_calls(monkeypatch):
    """An orphan from an earlier run holds one of worker 0's slots, so it starts with one call
    where the others start with two."""
    import ray

    orphan, given = object(), {0: 0, 1: 0}
    results = {}

    def submit(worker, units):
        ref = object()
        given[worker] += 1
        results[ref] = [([], u, (0.0, 0.0, 0, 0.0, 0.0)) for u in units]
        return ref

    def wait(refs, num_returns, timeout=None):
        live = [r for r in refs if r is not orphan]
        if timeout == 0:  # the orphan check: the orphan is still running
            return live, [r for r in refs if r is orphan]
        return live[:1], live[1:]

    monkeypatch.setattr(ray, "wait", wait)
    monkeypatch.setattr(ray, "get", lambda ref: results[ref])
    transfer._ORPHANS.append(("a", orphan))
    calls = list(range(3))
    # Positive control on the dealing: before any landing, worker 0 holds one call.
    first = {}
    real_drain = transfer._drain

    def drain(*args, **kwargs):
        first.update(given)
        return real_drain(*args, **kwargs)

    monkeypatch.setattr(transfer, "_drain", drain)
    out = transfer.pull_units(calls, 2, 2, submit, ["a", "b"])
    assert [unit for _rows, unit, _t in out] == calls
    assert first == {0: 1, 1: 2}


def test_a_subtree_the_plan_shares_is_hoisted_once(monkeypatch):
    """TPC-H q21 at SF100 raised `KeyError: 5`: one `supplier JOIN nation` object sat under
    both EXISTS arms, was hoisted twice, and the identity-keyed swap left id 5 unread."""
    from batcher.dist.executors.aligned.analysis import AlignedCut, KeyClass
    from batcher.plan.logical import Join, Scan
    from batcher.plan.logical.join import JoinOutputCol
    from batcher.plan.schema import SchemaRef

    schema = SchemaRef.from_arrow(pa.schema([("k", pa.int64())]))
    keep_left = (JoinOutputCol("left", "k", "k"),)

    def semi(left, right):
        return Join(left, right, ("k",), ("k",), "semi", keep_left)

    shared = semi(Scan(1, schema), Scan(2, schema))
    body = semi(semi(Scan(0, schema), shared), shared)
    cut = AlignedCut(body, None, KeyClass.by_file(0), frozenset({0}), frozenset({1, 2}))
    evaluated = []

    def evaluate(node, scanned, sources, workers):
        evaluated.append(node)
        return pa.table({"k": pa.array([1], pa.int64())})

    from batcher.dist.executors.aligned import hoist

    monkeypatch.setattr(hoist, "evaluate_broadcast", evaluate)
    monkeypatch.setattr(hoist, "on_node_schema", lambda *a: None)
    held: dict = {}
    local: dict = {}
    out = hoist.hoist_broadcasts(body, cut, [None, None, None], held, 2, local)
    assert len(evaluated) == 1
    assert list(held) == [3]
    placeholders = {n.right.source_id for n in (out, out.left)}
    assert placeholders == {3}


def test_results_past_the_driver_budget_decline_rather_than_raise(monkeypatch):
    """The decline cancels the calls still running; they are actor calls, which Ray refuses to
    force-cancel, and TPC-H q9/q10 at SF1000 failed with that `ValueError` instead of falling
    back to another executor."""
    import sys
    import types

    batch = pa.RecordBatch.from_pylist([{"k": 1}])
    cancelled = []

    def cancel(ref, force=False):
        if force:
            raise ValueError("force=True is not supported for actor tasks.")
        cancelled.append(ref)

    fake = types.SimpleNamespace(
        wait=lambda refs, num_returns, timeout: ([refs[0]], refs[1:]),
        get=lambda ref: [([batch], None, (0.0, 0.0, 0.0, 0.0))],
        cancel=cancel,
    )
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.setattr(transfer, "RESULT_BYTES", 0)
    pending = {"a": (0, [0]), "b": (1, [1])}
    out = transfer._drain([(), ()], pending, lambda worker, ref: None, 0.0)
    assert out is None
    assert cancelled == ["b"]


def test_the_driver_gathers_no_more_than_its_machine_holds(monkeypatch):
    """`result_budget`: the constant ceiling, lowered to what the driver can actually hold.

    TPC-H q22 at SF1000 was killed on a 32 GB head node gathering 18 GB under a 12 GiB
    constant beside the object store. A roomy driver keeps the ceiling (the control).
    """
    from batcher.dist.executors.aligned import transfer

    gib = 1 << 30

    class _Engine:
        reading: tuple[int, int] | None = (200 * gib, 4 * gib)

        def memory_headroom(self):
            return self.reading

    fake = _Engine()
    monkeypatch.setattr("batcher._internal.native.engine", lambda: fake)
    monkeypatch.setattr(transfer, "RESULT_BYTES", 12 * gib)
    assert transfer.result_budget() == 12 * gib
    fake.reading = (20 * gib, 2 * gib)
    assert transfer.result_budget() == 8 * gib  # (20 - 2x2) / 2
    fake.reading = None
    assert transfer.result_budget() == 12 * gib
    monkeypatch.setattr(transfer, "RESULT_BYTES", 0)
    fake.reading = (200 * gib, 4 * gib)
    assert transfer.result_budget() == 0  # the configured ceiling always wins
