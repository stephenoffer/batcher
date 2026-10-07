"""The shuffle worker runs under the driver's `Config`, not the worker process's defaults.

`_FlightWorker` is a Ray actor: its process sees neither the driver's `config_context` nor its
profile, so any tunable it reads through `active_config()` silently answers with the worker's
own defaults. The gather's stream count and byte bound, the shuffle store cap, the AIMD gains
and the pressure limits all used to be read that way. These tests construct the actor's class
body directly (no Ray cluster) with a config that differs from the ambient one, and check the
shipped values are the ones installed.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("ray", reason="the fleet actor lives behind the optional ray extra")

import batcher.carbonite.transfer as transfer_mod
import batcher.dist.flight_worker as fw
from batcher.config import active_config

pytestmark = pytest.mark.unit

_CLASS = fw._FlightWorker.__ray_metadata__.modified_class


class _FakeNative:
    def __init__(self) -> None:
        self.transport: tuple | None = None

    def set_flight_transport_config(self, *args) -> None:
        self.transport = args


class _FakeSession:
    def __init__(self, credits, **kwargs) -> None:
        self.credits = credits
        self.kwargs = kwargs


def _driver_config():
    base = active_config()
    fc = dataclasses.replace(
        base.flow_control,
        gather_streams=base.flow_control.gather_streams + 7,
        gather_inflight_bytes=base.flow_control.gather_inflight_bytes + 12345,
        aimd_alpha=base.flow_control.aimd_alpha + 3,
    )
    return dataclasses.replace(base, flow_control=fc)


@pytest.fixture
def construct(monkeypatch):
    nat = _FakeNative()
    monkeypatch.setattr(fw, "engine", lambda: nat)
    monkeypatch.setattr(transfer_mod, "ShuffleSession", _FakeSession)
    monkeypatch.setattr(fw.ray.util, "get_node_ip_address", lambda: "127.0.0.1")

    def _build(config, *, adaptive: bool):
        worker = _CLASS.__new__(_CLASS)
        _CLASS.__init__(worker, 0, 4, "", adaptive, config=config)
        return worker, nat

    return _build


def test_the_gather_and_store_settings_come_from_the_shipped_config(construct) -> None:
    from batcher.carbonite.policies import shuffle_store_cap

    cfg = _driver_config()
    assert cfg.flow_control.gather_streams != active_config().flow_control.gather_streams
    _worker, nat = construct(cfg, adaptive=False)
    assert nat.transport is not None
    *_, store_cap, streams, inflight, _spill_root = nat.transport
    assert streams == cfg.flow_control.gather_streams
    assert inflight == cfg.flow_control.gather_inflight_bytes
    assert store_cap == shuffle_store_cap(cfg)


def test_the_shuffle_store_spills_under_the_shipped_spill_dir(construct, tmp_path) -> None:
    """The store's spill root is the configured `memory.spill_dir`, not the OS tempdir.

    It used to receive no root at all and spilled under `temp_dir()` whatever the operator
    configured — on a container whose `/tmp` is a small tmpfs, into the RAM it was meant to
    free (BT-030).
    """
    base = _driver_config()
    cfg = dataclasses.replace(
        base, memory=dataclasses.replace(base.memory, spill_dir=str(tmp_path))
    )
    _worker, nat = construct(cfg, adaptive=False)
    assert nat.transport is not None
    assert nat.transport[-1] == str(tmp_path)


def test_with_no_spill_dir_the_store_uses_the_measured_local_scratch(
    construct, monkeypatch
) -> None:
    import batcher._internal.site as site

    base = _driver_config()
    cfg = dataclasses.replace(base, memory=dataclasses.replace(base.memory, spill_dir=None))
    monkeypatch.setattr(site, "local_scratch_root", lambda: "/mnt/nvme0")
    _worker, nat = construct(cfg, adaptive=False)
    assert nat.transport is not None
    assert nat.transport[-1] == "/mnt/nvme0"


def test_the_aimd_controller_uses_the_shipped_gains(construct) -> None:
    cfg = _driver_config()
    worker, _nat = construct(cfg, adaptive=True)
    controller = worker.session.kwargs["flow_control"]
    assert controller._alpha == cfg.flow_control.aimd_alpha
    assert controller._alpha != max(1, active_config().flow_control.aimd_alpha)


def test_spawn_ships_the_drivers_active_config(monkeypatch) -> None:
    import batcher.dist.executors.ray_runtime as rt
    from batcher.config import config_context

    calls: list[dict] = []

    class _Handle:
        def options(self, **_opts):
            return self

        def remote(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return object()

    monkeypatch.setattr(fw, "_FlightWorker", _Handle())
    monkeypatch.setattr(rt, "current_envelope", lambda: None)
    monkeypatch.setattr(rt, "create_worker_placement", lambda *_a: None)
    monkeypatch.setattr(rt, "fleet_actor_options", lambda _pg, n: [{}] * n)
    monkeypatch.setattr(fw, "_slot_engine_configs", lambda _env, n, cfg, _pg: [cfg] * n)

    cfg = _driver_config()
    with config_context(cfg):
        fw.spawn_flight_workers(2, 4, "{}", plan_id=7)
    assert len(calls) == 2
    assert all(c["args"][-1] == cfg for c in calls)
