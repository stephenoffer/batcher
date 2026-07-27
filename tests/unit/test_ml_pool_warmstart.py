"""InferencePool threads the learned batch-size warm-start into its throughput controller.

Before, the pool constructed a `ThroughputController` with no `hub`/`signature`, so every
run cold-climbed from `target_batch_rows` and the autobatch warm-start machinery was dead
from the pool. These assert the plateau is now read back and recorded. Pure fake hub.
"""

from __future__ import annotations

from batcher.metadata.hardware_scope import scoped
from batcher.ml.autobatch import _LEARN_NS
from batcher.ml.inference import InferencePool


class _FakeHub:
    def __init__(self, stored: dict | None = None) -> None:
        self._s = dict(stored or {})

    def get_keyed_param(self, ns: str, sig: str):
        return self._s.get((ns, sig))

    def put_keyed_param(self, ns: str, sig: str, val: dict) -> None:
        self._s[(ns, sig)] = val


def test_pool_warm_starts_the_batch_size_from_the_hub():
    # Seeded under the hardware-scoped namespace the warm-start reads: a learned batch
    # size is measured against a specific GPU, so it is stored per machine class.
    hub = _FakeHub({(scoped(_LEARN_NS), "embed-job"): {"size": 999}})
    pool = InferencePool(
        lambda: lambda b: b,
        objective="throughput",
        target_batch_rows=128,  # cold default...
        max_batch_rows=4096,
        learned_hub=hub,
        learned_signature="embed-job",
    )
    # ...but the controller starts from the learned 999, not 128.
    assert pool._throughput_ctl.current() == 999


def test_pool_without_a_hub_is_the_pure_hill_climb():
    pool = InferencePool(lambda: lambda b: b, objective="throughput", target_batch_rows=128)
    assert pool._throughput_ctl.current() == 128
