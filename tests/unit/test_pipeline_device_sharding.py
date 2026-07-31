"""A transformers pipeline shards across devices only when it has to.

`device_map="auto"` is accelerate's *balanced* map, not a fill-first one: it spreads the layers
evenly over every visible device whether or not they need spreading. A model that fits one card
then runs as a naive pipeline with no micro-batching, so one device computes while the rest
wait and a transfer crosses the bus at every stage boundary. That is slower than the
single-device pin it replaced, on more hardware, and it is the default a local multi-GPU run
lands on.

The asymmetry that decides every case here: sharding a model that turns out not to fit is
recoverable, and pinning one that does not fit is an out-of-memory error at load. So an
unreadable footprint keeps the old sharding behaviour rather than the better-looking one.
"""

from __future__ import annotations

import pytest

from batcher.ml.inference import pipelines

pytestmark = pytest.mark.unit

GIB = 1 << 30


@pytest.fixture
def _multi_gpu(monkeypatch):
    """A process that sees two 80 GiB CUDA devices with accelerate installed."""
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("batcher.ml.llm.engines.footprint.device_total_bytes", lambda: 80 * GIB)

    class _Cuda:
        @staticmethod
        def device_count() -> int:
            return 2

    monkeypatch.setitem(__import__("sys").modules, "torch", type("torch", (), {"cuda": _Cuda})())


def _weights(monkeypatch, value: int | None) -> None:
    monkeypatch.setattr("batcher.ml.llm.engines.footprint.model_weight_bytes", lambda model: value)


def test_a_model_that_fits_one_device_is_pinned(monkeypatch, _multi_gpu) -> None:
    _weights(monkeypatch, 14 * GIB)
    assert not pipelines._should_shard_across_devices("cuda", "org/small")


def test_a_model_too_large_for_one_device_is_sharded(monkeypatch, _multi_gpu) -> None:
    _weights(monkeypatch, 140 * GIB)
    assert pipelines._should_shard_across_devices("cuda", "org/large")


def test_a_model_that_only_fits_without_headroom_is_still_sharded(monkeypatch, _multi_gpu) -> None:
    # 72 GiB of weights on an 80 GiB card leaves nothing for activations or the CUDA context,
    # and "fits" has to mean fits in practice.
    _weights(monkeypatch, 72 * GIB)
    assert pipelines._should_shard_across_devices("cuda", "org/snug")


def test_an_unreadable_footprint_keeps_the_previous_behaviour(monkeypatch, _multi_gpu) -> None:
    _weights(monkeypatch, None)
    assert pipelines._should_shard_across_devices("cuda", "org/unknown")


def test_a_model_named_by_nobody_keeps_the_previous_behaviour(monkeypatch, _multi_gpu) -> None:
    # The caller may not have a model id at all (an already-built pipeline object).
    _weights(monkeypatch, 14 * GIB)
    assert pipelines._should_shard_across_devices("cuda", "")


def test_a_single_device_process_never_shards(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())

    class _Cuda:
        @staticmethod
        def device_count() -> int:
            return 1

    monkeypatch.setitem(__import__("sys").modules, "torch", type("torch", (), {"cuda": _Cuda})())
    assert not pipelines._should_shard_across_devices("cuda", "org/large")


def test_without_accelerate_the_pin_is_kept(monkeypatch, _multi_gpu) -> None:
    # `device_map` would raise where the pin at least loads.
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    _weights(monkeypatch, 140 * GIB)
    assert not pipelines._should_shard_across_devices("cuda", "org/large")


@pytest.mark.parametrize("device", ["cpu", "xpu", "mps", "xla"])
def test_only_cuda_uses_the_auto_map(device: str, _multi_gpu) -> None:
    assert not pipelines._should_shard_across_devices(device, "org/large")


def test_an_unreadable_device_size_keeps_the_previous_behaviour(monkeypatch) -> None:
    monkeypatch.setattr("batcher.ml.llm.engines.footprint.device_total_bytes", lambda: None)
    _weights(monkeypatch, 14 * GIB)
    assert not pipelines._fits_one_device("org/small")
