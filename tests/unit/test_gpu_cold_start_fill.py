"""A first, unmeasured run must fill its devices, not wait for a second run to do it.

The GPU packing loop is a measure-then-adapt loop, so run 0 of a new pipeline runs whatever
the cold default is — and one actor per device is the wrong cold default for an inference
stage. A `map_batches` UDF spends part of every batch on the host (decode, normalize, the
H2D copy) with the device idle: measured on four T4s, one actor per device held ResNet-50 at
78% and two held it at 92%. A first run that cannot pack pays that gap in full, and a
single-shot job never gets a second run.

Two things have to be true for run 0 to pack, and each was false on its own:

1. The *reservation* must be packable. Ray fixes an actor's `num_gpus` at creation, so a
   pool that reserved a whole GPU each can never add a second actor however much room the
   model leaves — the reservation, not the model, would be the reason it stayed unpacked.
2. The *decision* must be measured, not guessed. The model's footprint is read after it
   loads and before any batch runs, so a device is only ever packed once it has shown room.
"""

from __future__ import annotations

import pytest

from batcher.ml.gpu import (
    cold_start_actors_per_device,
    cold_start_gpu_fraction,
    recommend_num_gpus,
)

pytestmark = pytest.mark.unit


def test_an_unmeasured_whole_gpu_request_reserves_a_packable_fraction() -> None:
    assert cold_start_gpu_fraction(1.0) == 0.5


def test_a_caller_that_sized_its_own_packing_is_left_alone() -> None:
    """An explicit sub-whole request already expresses a density; don't second-guess it."""
    assert cold_start_gpu_fraction(0.25) == 0.25
    assert cold_start_gpu_fraction(0.5) == 0.5


def test_the_reservation_never_grows_past_what_was_asked_for() -> None:
    assert cold_start_gpu_fraction(0.1) == 0.1


def test_a_small_model_packs_two_to_a_device() -> None:
    assert cold_start_actors_per_device(0.12) == 2


def test_a_model_that_fills_the_device_stays_alone_on_it() -> None:
    assert cold_start_actors_per_device(0.5) == 1
    assert cold_start_actors_per_device(0.95) == 1


def test_an_unmeasurable_device_is_never_packed_on_a_guess() -> None:
    """No NVML, no CUDA, a CPU stage: fall back to the previous unpacked behaviour."""
    assert cold_start_actors_per_device(None) == 1


def test_the_cold_fill_never_exceeds_two_however_small_the_model() -> None:
    """Two closes most of the 78%->92% gap and only doubles VRAM demand; the measured loop
    refines past it on the next run, where it has utilization to justify going further."""
    assert cold_start_actors_per_device(0.001) == 2


def test_the_measured_loop_takes_over_once_anything_has_been_recorded() -> None:
    """The cold start is a first-run default, not a competing policy: from the first recorded
    utilization onward the density comes from `recommend_num_gpus`."""
    # Two actors per device measured at 45% -> room for four.
    assert recommend_num_gpus(0.45, 1.0, 2) == 0.25
    # ... and the packed configuration is a fixed point once the target is reached.
    assert recommend_num_gpus(0.92, 1.0, 2) == 0.5


def test_the_cold_fill_is_skipped_for_an_explicit_concurrency() -> None:
    """`concurrency=N` is the caller sizing their own pool; the engine must not resize it."""
    from batcher.dist.executors.map import _cold_start_devices

    assert _cold_start_devices(None, None, 4, 1.0) == 0
    assert _cold_start_devices(None, None, (2, 8), 1.0) == 0


def test_the_cold_fill_is_skipped_for_a_cpu_stage() -> None:
    from batcher.dist.executors.map import _cold_start_devices

    assert _cold_start_devices(None, None, None, 0.0) == 0


def test_the_vram_cap_is_read_per_actor_not_per_device() -> None:
    """A device holding two actors at 12% each reads 24%; that is not a 24% actor.

    Read as one actor's footprint, the cap falls as the pool grows — so the recommendation
    falls with it, the pool is rebuilt (a full model reload on every device), and the loop
    never settles. Measured: the density walked 2 -> 3 -> 2 across three consecutive runs,
    each one paying a rebuild.
    """
    from batcher.ml.gpu import actors_per_gpu_from_learned_vram as fits

    alone = fits(0.12)
    assert fits(0.24, actors_per_device=2) == alone
    assert fits(0.36, actors_per_device=3) == alone


def test_the_loop_settles_instead_of_walking_the_density() -> None:
    """With a per-actor cap the recommendation is a fixed point, which is the whole point:
    a density that changes every run rebuilds the pool every run."""
    from batcher.ml.gpu import actors_per_gpu_from_learned_vram as fits

    # Three actors per device, saturated: the loop must ask for three again.
    cap = fits(0.36, actors_per_device=3)
    assert recommend_num_gpus(0.95, 1.0, 3, cap) == pytest.approx(0.33)
    # Two actors already above the satisfied band: hold, do not pay a rebuild for the tail.
    cap2 = fits(0.24, actors_per_device=2)
    assert recommend_num_gpus(0.855, 1.0, 2, cap2) == pytest.approx(0.5)


def test_a_device_at_the_goal_is_left_alone() -> None:
    """80% is the bar; chasing the last points costs a pool rebuild on every device and
    measured slower and less evenly spread than the density it left."""
    assert recommend_num_gpus(0.83, 1.0, 2, 6) == pytest.approx(0.5)
    assert recommend_num_gpus(0.95, 1.0, 2, 6) == pytest.approx(0.5)


def test_a_starved_device_still_packs() -> None:
    assert recommend_num_gpus(0.45, 1.0, 2, 6) == pytest.approx(0.25)
    assert recommend_num_gpus(0.78, 1.0, 1, 6) == pytest.approx(0.5)
