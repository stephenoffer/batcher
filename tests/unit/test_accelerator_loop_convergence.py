"""The packing loop, closed. Does it actually land on a fed device, and never on an OOM?

Every existing test drives `recommend_num_gpus` for one step. That proves the arithmetic and
not the *loop*, and a control loop's failure modes are all in the loop: it oscillates, it
never reaches the target, or it walks past what memory allows. This runs the loop the way
`api.executors` composes it — measure, record density and peak VRAM, recommend, repeat — over
a simulated device, and asserts the three properties that matter.

No GPU: the device is a pair of functions from actor count to utilization and to memory. That
is exactly the interface the loop sees, so what is being tested is the controller rather than
a mock of it.
"""

from __future__ import annotations

import pytest

from batcher.ml.gpu import (
    _MAX_ACTORS_PER_DEVICE,
    _PACK_SATISFIED,
    actors_per_gpu_from_learned_vram,
    recommend_num_gpus,
)

pytestmark = pytest.mark.unit


class Device:
    """A simulated accelerator: `actors` in, utilization and peak-memory fraction out.

    `util_per_actor` is what one actor alone drives the device to, and utilization saturates
    at 1.0 rather than summing past it — a device cannot be more than busy. `vram_per_actor`
    is one actor's share of memory, which adds without a ceiling: a device *can* be asked for
    more memory than it has, and that is the failure the loop must never provoke.
    """

    def __init__(self, util_per_actor: float, vram_per_actor: float) -> None:
        self.util_per_actor = util_per_actor
        self.vram_per_actor = vram_per_actor

    def utilization(self, actors: int) -> float:
        return min(1.0, actors * self.util_per_actor)

    def peak_vram(self, actors: int) -> float:
        return actors * self.vram_per_actor


def run_loop(device: Device, *, rounds: int = 12, start: int = 1) -> list[int]:
    """The densities the loop visits, exactly as `api.executors` sequences the pieces."""
    density = start
    history = [density]
    for _ in range(rounds):
        # What the run measured, and the density that produced it.
        util = device.utilization(density)
        peak = device.peak_vram(density)
        # What the next run asks for.
        cap = actors_per_gpu_from_learned_vram(peak, actors_per_device=density)
        fraction = recommend_num_gpus(util, 1.0, density, cap)
        density = max(1, round(1.0 / fraction)) if fraction > 0 else 1
        history.append(density)
    return history


def settled(history: list[int]) -> int | None:
    """The density the loop settled on, or `None` if it never stopped moving."""
    tail = history[-4:]
    return tail[0] if len(set(tail)) == 1 else None


@pytest.mark.parametrize("util_per_actor", [0.10, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 0.95])
def test_the_loop_settles_on_a_fed_device_when_memory_allows(util_per_actor):
    # Memory is deliberately not the binding constraint here, so this isolates the utilization
    # half: whatever one actor drives the device to, the loop must end with it fed.
    device = Device(util_per_actor=util_per_actor, vram_per_actor=0.02)
    history = run_loop(device)
    final = settled(history)
    assert final is not None, f"never settled: {history}"
    reached = device.utilization(final)
    ceiling = device.utilization(_MAX_ACTORS_PER_DEVICE)
    assert reached >= min(_PACK_SATISFIED, ceiling), f"settled at {reached:.2f}: {history}"


@pytest.mark.parametrize("util_per_actor", [0.10, 0.2, 0.3, 0.5])
@pytest.mark.parametrize("vram_per_actor", [0.05, 0.1, 0.2, 0.35, 0.5])
def test_the_loop_never_asks_for_more_memory_than_the_device_has(util_per_actor, vram_per_actor):
    # The property that matters most: utilization must never win over memory. A loop that
    # packs for the last few points of a busy device walks a whole fleet into an OOM.
    device = Device(util_per_actor=util_per_actor, vram_per_actor=vram_per_actor)
    for density in run_loop(device):
        assert device.peak_vram(density) <= 1.0, f"asked for {device.peak_vram(density):.2f}"


@pytest.mark.parametrize("util_per_actor", [0.1, 0.2, 0.3, 0.45, 0.7])
def test_the_loop_stops_moving_rather_than_oscillating(util_per_actor):
    # Every change is a pool rebuild — a model reload on every device — so a loop that keeps
    # stepping between two densities costs more than the utilization it is chasing.
    device = Device(util_per_actor=util_per_actor, vram_per_actor=0.05)
    history = run_loop(device, rounds=20)
    assert settled(history) is not None, f"oscillated: {history}"


def test_a_memory_bound_model_settles_below_the_utilization_target():
    # A big model that leaves the device half idle must still stop at what fits, and must not
    # keep asking for the actor that would not.
    device = Device(util_per_actor=0.25, vram_per_actor=0.4)  # 2 actors fit; 2 gives 50% busy
    history = run_loop(device)
    final = settled(history)
    assert final is not None, history
    assert device.peak_vram(final) <= 1.0
    assert final <= 2, f"packed {final} actors of 40% VRAM each"


def test_an_already_fed_device_is_left_exactly_where_it_is():
    # The fixed point. A device measured at or above the target recomputes the same density.
    for density in (1, 2, 4):
        device = Device(util_per_actor=_PACK_SATISFIED / density, vram_per_actor=0.05)
        assert run_loop(device, start=density)[1:] == [density] * 12


def test_the_loop_converges_within_a_few_runs():
    # A learning loop that needs twenty runs to settle never settles in practice, because the
    # pipeline changes first.
    device = Device(util_per_actor=0.12, vram_per_actor=0.05)
    history = run_loop(device, rounds=12)
    first_settled = next(i for i in range(len(history) - 3) if len(set(history[i : i + 4])) == 1)
    assert first_settled <= 3, f"took {first_settled} rounds: {history}"


def test_a_starved_device_is_never_left_at_one_actor():
    # The regression that motivates the whole loop: one actor holding a device at 20% is
    # leaving four fifths of an accelerator idle, and the bill does not.
    device = Device(util_per_actor=0.2, vram_per_actor=0.05)
    assert settled(run_loop(device)) >= 4


def test_the_density_never_exceeds_the_hard_ceiling():
    # A very cheap model would otherwise pack until per-actor contexts cost more than the
    # compute they win.
    device = Device(util_per_actor=0.01, vram_per_actor=0.001)
    assert max(run_loop(device)) <= _MAX_ACTORS_PER_DEVICE
