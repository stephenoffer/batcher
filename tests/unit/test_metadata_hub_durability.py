"""The MetadataHub's guarantees about *not* losing what has been measured.

Every test here pins a failure that is silent by construction: the learning loop is
best-effort in both directions, so when it stops working nothing raises and no result is
wrong — plans simply stop improving, which is invisible until someone benchmarks. These
assert the store keeps answering under the conditions that used to switch it off.
"""

from __future__ import annotations

import json
import math
import threading

import pytest

from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.sqlite import SQLiteBackend
from batcher.metadata.hardware_scope import scoped
from batcher.metadata.hub import _OP_STATS_MAX, MetadataHub
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar
from batcher.plan.feedback import OperatorFeedback

pytestmark = pytest.mark.unit


def _row(**over: object) -> bytes:
    from batcher._internal.hardware import fingerprint

    row = {
        "op_id": 1,
        "kind": "filter",
        "n_actual": 10,
        "t_op_ms": 1.0,
        "signature": "sig",
        "hw_fingerprint": fingerprint(),
    }
    row.update(over)
    return json.dumps(row).encode()


def _feedback(**over: object) -> OperatorFeedback:
    kwargs: dict = {
        "op_id": 1,
        "kind": "filter",
        "n_actual": 10,
        "t_op_ms": 1.0,
        "m_peak_bytes": 0,
        "selectivity": 1.0,
        "batch_size": 1024,
        "signature": "sig",
    }
    kwargs.update(over)
    return OperatorFeedback(**kwargs)  # type: ignore[arg-type]


class TestCorruptRowsAreIsolated:
    """One unreadable row must cost one row, not the whole measured history."""

    def test_undecodable_row_does_not_empty_the_views(self) -> None:
        backend = InProcessBackend()
        backend.put("op_stats", (1, 1), _row(kind="filter"))
        backend.put("op_stats", (1, 2), b"{not json at all")
        backend.put("op_stats", (1, 3), _row(kind="hash_join"))
        hub = MetadataHub(backend)

        by_kind = hub.op_stats_by_kind()

        # Before per-row isolation the bad row aborted the loop, so `hash_join` — written
        # after it — vanished and cost calibration lost that whole operator family.
        assert sorted(by_kind) == ["filter", "hash_join"]
        assert len(hub.op_stats_with_signature()) == 2

    def test_non_object_row_is_skipped_not_crashed(self) -> None:
        backend = InProcessBackend()
        backend.put("op_stats", (1, 1), b"null")  # valid JSON, not a row
        backend.put("op_stats", (1, 2), b"7")
        backend.put("op_stats", (1, 3), _row())
        hub = MetadataHub(backend)

        assert list(hub.op_stats_by_kind()) == ["filter"]

    def test_a_corrupt_row_late_in_the_scan_keeps_the_earlier_ones(self) -> None:
        backend = InProcessBackend()
        for seq in range(1, 21):
            backend.put("op_stats", (1, seq), _row(signature=f"s{seq}"))
        backend.put("op_stats", (1, 21), b"\xff\xfe truncated")
        hub = MetadataHub(backend)

        assert len(hub.op_stats_with_signature()) == 20


class TestViewsLoadOnce:
    """Both derived views come from one scan, and a concurrent record is not lost."""

    def test_a_single_scan_builds_both_views(self) -> None:
        backend = InProcessBackend()
        backend.put("op_stats", (1, 1), _row())
        scans = 0
        inner = backend.scan

        def counting_scan(table: str, prefix: tuple = ()):  # type: ignore[type-arg]
            nonlocal scans
            if table == "op_stats":
                scans += 1
            return inner(table, prefix)

        backend.scan = counting_scan  # type: ignore[method-assign]
        hub = MetadataHub(backend)

        hub.op_stats_by_kind()
        hub.op_stats_with_signature()
        hub.op_stats_by_kind()

        assert scans == 1

    def test_recording_while_the_view_loads_keeps_the_row(self) -> None:
        # The load holds the hub's lock, so a record that lands during it either precedes
        # the scan (and is found) or follows the assignment (and is folded in). Neither
        # ordering may drop it.
        backend = InProcessBackend()
        hub = MetadataHub(backend)
        done = threading.Event()

        def recorder() -> None:
            for i in range(200):
                hub.record(_feedback(op_id=i, signature=f"s{i}"))
            done.set()

        thread = threading.Thread(target=recorder)
        thread.start()
        signed = hub.op_stats_with_signature()
        thread.join()
        done.wait(timeout=5)

        assert len(signed) == 200
        assert hub.version == 200


class TestNamespaceCachesAreIndependent:
    """A write to one learned-parameter namespace must not invalidate the others."""

    def test_saving_one_namespace_keeps_another_cached(self) -> None:
        backend = InProcessBackend()
        hub = MetadataHub(backend)
        hub.put_keyed_param("kyber.calibration", "row_ns", 1.5)
        assert hub.get_keyed_param("kyber.calibration", "row_ns") == 1.5

        scans = 0
        inner = backend.scan

        def counting_scan(table: str, prefix: tuple = ()):  # type: ignore[type-arg]
            nonlocal scans
            if prefix[:1] == ("kyber.calibration",):
                scans += 1
            return inner(table, prefix)

        backend.scan = counting_scan  # type: ignore[method-assign]

        # A source's statistics are persisted under a namespace of their own, once per
        # written dataset. Under a single shared generation counter this threw away every
        # other namespace's parsed view.
        hub.save_params("io.source_stats:/data/a.parquet", {"row_count": 10})
        hub.save_params("io.source_stats:/data/b.parquet", {"row_count": 20})

        assert hub.get_keyed_param("kyber.calibration", "row_ns") == 1.5
        assert scans == 0

    def test_saving_a_namespace_still_invalidates_its_own_keyed_view(self) -> None:
        backend = InProcessBackend()
        hub = MetadataHub(backend)
        assert hub.load_keyed_params("ns") == {}
        hub.save_params("ns", {"a": 1})
        assert hub.load_keyed_params("ns") == {"a": 1}

    def test_load_params_is_served_from_the_parsed_view(self) -> None:
        backend = InProcessBackend()
        hub = MetadataHub(backend)
        hub.save_params("ns", {"a": 1})
        gets = 0
        inner = backend.get

        def counting_get(table: str, key: tuple):  # type: ignore[type-arg]
            nonlocal gets
            gets += 1
            return inner(table, key)

        backend.get = counting_get  # type: ignore[method-assign]
        assert hub.load_params("ns") == {"a": 1}
        assert hub.load_params("ns") == {"a": 1}
        assert gets == 0

    def test_the_resident_namespace_set_is_bounded(self) -> None:
        hub = MetadataHub(InProcessBackend())
        for i in range(2000):
            hub.save_params(f"io.source_stats:/data/{i}.parquet", {"row_count": i})
        # One namespace per source path would otherwise retain every decoded blob forever.
        assert len(hub._params._generations) <= 256
        assert len(hub._params._blob_cache) <= 256
        # Eviction must not lose data — the store still answers, it just re-reads.
        assert hub.load_params("io.source_stats:/data/0.parquet") == {"row_count": 0}


class TestSmoothedScalarsCannotBePoisoned:
    """A single non-finite observation must not corrupt a learned scalar permanently."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_observation_is_dropped(self, bad: float) -> None:
        hub = MetadataHub(InProcessBackend())
        record_smoothed_scalar(hub, "ns", "k", 100.0)
        record_smoothed_scalar(hub, "ns", "k", bad)

        value = load_scalar(hub, "ns", "k")
        assert value == 100.0

    def test_a_poisoned_stored_value_reads_as_absent(self) -> None:
        hub = MetadataHub(InProcessBackend())
        hub.put_keyed_param("ns", "k", {"value": float("nan"), "n": 5.0})
        assert load_scalar(hub, "ns", "k") is None

        # ...and the next observation restarts the estimate rather than blending into NaN.
        record_smoothed_scalar(hub, "ns", "k", 4.0)
        assert load_scalar(hub, "ns", "k") == 4.0

    def test_a_nonsense_observation_count_cannot_invert_the_step(self) -> None:
        hub = MetadataHub(InProcessBackend())
        hub.put_keyed_param("ns", "k", {"value": 10.0, "n": -5.0})
        record_smoothed_scalar(hub, "ns", "k", 20.0)

        value = load_scalar(hub, "ns", "k")
        assert value is not None and math.isfinite(value)
        # A step of the wrong sign would move the estimate away from the observation.
        assert 10.0 < value <= 20.0

    def test_a_legacy_bare_float_still_reads_and_migrates(self) -> None:
        hub = MetadataHub(InProcessBackend())
        hub.save_params("ns", {"k": 8.0})  # the pre-per-key whole-blob shape
        assert load_scalar(hub, "ns", "k") == 8.0
        record_smoothed_scalar(hub, "ns", "k", 12.0)
        value = load_scalar(hub, "ns", "k")
        assert value is not None and 8.0 < value < 12.0


class TestLearnedGpuFiguresUseTheSharedHelper:
    """The GPU learned figures moved off their whole-blob read-modify-write."""

    def test_utilization_round_trips_and_smooths(self) -> None:
        from batcher.ml.gpu import load_gpu_utilization, record_gpu_utilization

        hub = MetadataHub(InProcessBackend())
        assert load_gpu_utilization(hub, "pipe") is None
        record_gpu_utilization(hub, "pipe", 0.8)
        assert load_gpu_utilization(hub, "pipe") == 0.8
        record_gpu_utilization(hub, "pipe", 0.4)
        blended = load_gpu_utilization(hub, "pipe")
        assert blended is not None and 0.4 < blended < 0.8

    def test_a_store_written_by_the_old_whole_blob_shape_keeps_answering(self) -> None:
        from batcher.ml.gpu import _NAMESPACE, load_gpu_utilization

        hub = MetadataHub(InProcessBackend())
        hub.save_params(scoped(_NAMESPACE), {"pipe": 0.55})
        assert load_gpu_utilization(hub, "pipe") == 0.55

    def test_two_pipelines_recording_do_not_clobber_each_other(self) -> None:
        from batcher.ml.gpu import record_gpu_utilization

        hub = MetadataHub(InProcessBackend())
        record_gpu_utilization(hub, "a", 0.1)
        record_gpu_utilization(hub, "b", 0.9)
        from batcher.ml.gpu import load_gpu_utilization

        assert load_gpu_utilization(hub, "a") == 0.1
        assert load_gpu_utilization(hub, "b") == 0.9


class TestInProcessBackendScan:
    """`scan` is a generator over a live table, so it must snapshot."""

    def test_writing_during_a_scan_does_not_raise(self) -> None:
        backend = InProcessBackend()
        for i in range(50):
            backend.put("op_stats", (1, i), b"{}")
        seen = 0
        for _key, _value in backend.scan("op_stats"):
            backend.put("op_stats", (2, seen), b"{}")  # the concurrent writer's effect
            seen += 1
        assert seen == 50


class TestSqliteBackend:
    """The durable default: thread-safe, prefix-pushed, and prunable."""

    def test_prefix_scan_matches_the_tuple_semantics(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        keys = [
            ("ns",),
            ("ns", "k1"),
            ("ns", "k2"),
            ("nsX",),
            ("n",),
            ("ns2", "q"),
            (5,),
            (5, 1),
            (50,),
        ]
        for i, key in enumerate(keys):
            backend.put("t", key, str(i).encode())

        for prefix in [(), ("ns",), ("nsX",), ("n",), (5,), (50,), ("ns2",)]:
            got = sorted((repr(k) for k, _ in backend.scan("t", prefix)))
            want = sorted(repr(k) for k in keys if k[: len(prefix)] == prefix)
            assert got == want, prefix

    def test_a_prefix_scan_does_not_read_the_whole_table(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        for i in range(500):
            backend.put("learned_params", (f"io.source_stats:/f{i}.parquet",), b"{}")
        backend.put("learned_params", ("kyber.calibration", "row_ns"), b"1.0")

        rows = list(backend.scan("learned_params", ("kyber.calibration",)))
        assert [k for k, _ in rows] == [("kyber.calibration", "row_ns")]

    def test_it_is_usable_from_another_thread(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        hub = MetadataHub(backend)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                hub.record(_feedback(op_id=i, signature=f"s{i}"))
                hub.load_params("kyber.calibration")
            except BaseException as exc:  # the assertion is that there is none
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        # `record` swallows, so the real assertion is that the rows actually landed.
        assert len(list(backend.scan("op_stats"))) == 8

    def test_it_uses_wal_and_relaxed_sync(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        assert backend._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert backend._conn.execute("PRAGMA synchronous").fetchone()[0] == 1

    def test_delete_drops_only_the_named_keys(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        backend.put("t", ("a",), b"1")
        backend.put("t", ("b",), b"2")
        backend.delete("t", [("a",), ("absent",)])
        assert [k for k, _ in backend.scan("t")] == [("b",)]

    def test_an_oversized_store_is_pruned_at_the_first_view_load(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        backend.batch_put(
            "op_stats",
            [((i % 7, i), _row(op_id=i % 7)) for i in range(_OP_STATS_MAX + 300)],
        )
        # A served workload writes a handful of rows per process, so the every-N-records
        # prune inside `record` is never reached and nobody ever trims the store.
        MetadataHub(backend).op_stats_by_kind()
        assert len(list(backend.scan("op_stats"))) == _OP_STATS_MAX

    def test_pruning_keeps_the_newest_rows(self, tmp_path) -> None:
        backend = SQLiteBackend(str(tmp_path / "m.db"))
        backend.batch_put("op_stats", [((0, i), _row()) for i in range(_OP_STATS_MAX + 10)])
        MetadataHub(backend).op_stats_by_kind()
        remaining = sorted(k[1] for k, _ in backend.scan("op_stats"))
        assert remaining[0] == 10
        assert remaining[-1] == _OP_STATS_MAX + 9
