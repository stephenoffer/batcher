"""The within-run batch-size loop, closed. Does it find the plateau, and never OOM there?

`ThroughputController`'s unit tests drive one `update` at a time, which proves the arithmetic
and not the loop. This runs it against a simulated device — a throughput curve and a memory
curve, both functions of batch size, with a hard limit that raises — and asserts the two
properties the loop exists for: it converges near the throughput optimum, and it does not
walk into an out-of-memory getting there.

No GPU: the device is two functions of batch size, which is exactly the interface the
controller sees.
"""

from __future__ import annotations

import pytest

from batcher.ml.autobatch import ThroughputController

pytestmark = pytest.mark.unit


class Device:
    """A simulated accelerator with a throughput curve, a memory curve, and a hard limit.

    Throughput saturates: doubling a batch stops paying once the device is busy, which is what
    makes a plateau exist to find. Memory grows linearly in the batch, and asking past
    `oom_rows` raises the way a real allocator does.
    """

    def __init__(self, knee: int, oom_rows: int, vram_per_row: float = 1 / 4096) -> None:
        self.knee = knee
        self.oom_rows = oom_rows
        self.vram_per_row = vram_per_row
        self.ooms = 0
        self.sizes: list[int] = []

    def run(self, rows: int) -> tuple[float, float]:
        """`(throughput, vram_fraction)` for a batch, or raise if it does not fit."""
        self.sizes.append(rows)
        if rows > self.oom_rows:
            self.ooms += 1
            raise RuntimeError(f"CUDA out of memory. Tried to allocate {rows} rows")
        # Rows/sec rises steeply below the knee and flattens above it.
        throughput = 1000.0 * rows / (rows + self.knee)
        return throughput, min(1.0, rows * self.vram_per_row)


def drive(device: Device, *, rounds: int = 60, **kwargs) -> ThroughputController:
    """Run the controller against the device, halving into the OOM retry the pool would."""
    controller = ThroughputController(min_rows=1, max_rows=65_536, initial=32, **kwargs)
    for _ in range(rounds):
        rows = controller.current()
        try:
            throughput, vram = device.run(rows)
        except RuntimeError:
            controller.note_oom(rows=rows)  # what `InferencePool._note_oom` does
            continue
        controller.update(throughput, vram)
    return controller


def test_the_climb_finds_the_plateau_when_memory_is_not_the_constraint():
    device = Device(knee=512, oom_rows=1_000_000, vram_per_row=1 / 1_000_000)
    controller = drive(device)
    # Past the knee the curve is flat, so "found it" means comfortably into the flat region
    # rather than an exact size — there is no exact size to find.
    assert controller.current() >= device.knee


def test_the_predictive_cap_keeps_the_climb_under_the_memory_ceiling():
    # The guard exists so the climb is out-of-memory-safe *by construction* rather than by
    # catching the failure after the fact.
    device = Device(knee=256, oom_rows=8192, vram_per_row=1 / 8192)
    drive(device)
    assert device.ooms == 0, (
        f"climbed into {device.ooms} OOMs at sizes {sorted(set(device.sizes))[-3:]}"
    )


@pytest.mark.parametrize("oom_rows", [64, 128, 512, 1024, 4096])
def test_an_unpredictable_oom_is_absorbed_and_not_re_entered(oom_rows):
    # A co-tenant that grows, a long-sequence shard, allocator fragmentation: the guard sees
    # this process's memory and cannot see those, so the ceiling has to be learned from the
    # failure. What must not happen is climbing back into it over and over.
    device = Device(knee=128, oom_rows=oom_rows, vram_per_row=1 / 1_000_000)  # memory invisible
    drive(device, rounds=80)
    assert device.ooms <= 3, f"re-entered the OOM {device.ooms} times"
    assert max(device.sizes[-20:]) <= oom_rows, "still running sizes that have failed"


def test_the_learned_ceiling_holds_for_the_rest_of_the_run():
    device = Device(knee=128, oom_rows=500, vram_per_row=1 / 1_000_000)
    controller = drive(device, rounds=80)
    assert controller.current() <= 500
    # One batch fitting is not evidence that a size which already failed became safe.
    assert all(size <= 500 for size in device.sizes[-30:])


def test_repeated_failures_converge_downward_rather_than_hovering():
    # A device that keeps failing must ratchet down, not sit just under a ceiling that is
    # itself too high.
    device = Device(knee=64, oom_rows=100, vram_per_row=1 / 1_000_000)
    drive(device, rounds=40)
    first_half = device.ooms
    drive(device, rounds=40)
    assert device.ooms - first_half <= first_half, "failure rate did not fall"


def test_the_size_settles_rather_than_drifting():
    device = Device(knee=512, oom_rows=100_000, vram_per_row=1 / 100_000)
    controller = drive(device, rounds=80)
    settled = controller.current()
    for _ in range(20):
        throughput, vram = device.run(controller.current())
        controller.update(throughput, vram)
    # A hill-climb on a flat curve must not wander: the plateau is where it stops.
    assert abs(controller.current() - settled) <= max(1, settled // 8)


def test_a_degenerate_measurement_cannot_freeze_the_controller():
    """An infinite or NaN reading must not leave the climb permanently unable to improve.

    `best_throughput = inf` makes every later observation non-improving, so without the
    staleness escape the size would settle back to whatever it was when the bad reading landed
    and stay there for the whole run. Recovery is driven with a *rising* throughput curve,
    because that is what distinguishes "recovered" from "cycling": on a flat curve there is no
    information that a bigger batch is better and holding is the correct answer.
    """
    device = Device(knee=2048, oom_rows=1_000_000, vram_per_row=1 / 1_000_000)
    controller = ThroughputController(min_rows=1, max_rows=65_536, initial=32)
    controller.update(float("inf"), 0.1)
    controller.update(float("nan"), 0.1)
    poisoned = controller.current()
    for _ in range(60):
        throughput, vram = device.run(controller.current())
        controller.update(throughput, vram)
    assert controller.current() > poisoned * 4, (
        f"still pinned near the poisoned size {poisoned}: {controller.current()}"
    )


def test_a_flat_throughput_curve_re_explores_rather_than_drifting_away():
    """On a plateau the controller alternates between the best size and one growth step.

    That is the staleness escape working as designed rather than a settled size: a durable
    regression would otherwise make the recorded optimum unreachable forever. What matters is
    that the excursion is *bounded* — one growth step above the best, not a drift — because the
    excursion is the direction that costs memory.
    """
    device = Device(knee=64, oom_rows=1_000_000, vram_per_row=1 / 1_000_000)
    controller = drive(device, rounds=40)
    best = controller.best_size()
    sizes = []
    for _ in range(40):
        throughput, vram = device.run(controller.current())
        controller.update(throughput, vram)
        sizes.append(controller.current())
    assert max(sizes) <= best * 2, f"excursion above the plateau is unbounded: {sorted(set(sizes))}"
    assert min(sizes) >= best // 2, f"drifted below the plateau: {sorted(set(sizes))}"


def test_the_vram_cap_shrinks_a_batch_that_is_already_over_it():
    controller = ThroughputController(min_rows=1, max_rows=65_536, initial=4096)
    before = controller.current()
    controller.update(1000.0, vram_fraction=0.99)
    assert controller.current() < before
