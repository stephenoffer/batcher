"""Measuring what a stage drew — Core's half of the energy loop.

The lanes here are the ones the architecture rule states: `plan.energy` defines the quantities,
Kyber decides against them, Carbonite protects a budget, and **Core measures**. This module is
that measurement: it brackets a stage, reads the device draw at both ends, and records one
`StageEnergy` into the run's ledger. It makes no decision and rewrites no plan.

**Three sources, in falling order of trust, and the record says which one it got.**

1. *Integrated.* On Volta and later the driver keeps a hardware joule counter per board, and the
   difference between two readings of it is the energy a stage actually consumed — every
   transient included. That is a measurement rather than an inference, and it is what a
   chargeback or a carbon figure needs. Recorded as `integrated=True`.
2. *Sampled.* Where the counter is absent, the draw is read at each end and multiplied by the
   duration. Real readings of a real board, and an approximation: it assumes the draw between
   the samples was the mean of them, which a stage alternating between a 60 W staged transfer
   and a 700 W kernel violates badly. Recorded as `measured=True, integrated=False`.
3. *Modelled.* Where NVML answers nothing, the datasheet model at the measured utilization.
   Recorded as `measured=False`.

A cost figure that cannot be told apart from an estimate is worth less than either, which is
why the distinction travels on every record rather than being decided once for the run.

**Sampling is at the ends, not on a timer.** A background thread per stage would cost more than
it measures on a short stage and would need shutting down on every failure path. That is why
source 2 degrades on a stage whose draw swings wildly — and why source 1 exists, since a counter
the hardware integrates continuously has no such limitation at any stage length that spans a
sampling interval.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections.abc import Iterator
from dataclasses import dataclass

from batcher.plan.energy import EnergyLedger, StageEnergy

__all__ = [
    "StageMeter",
    "active_ledger",
    "energy_scope",
    "measure_stage",
    "reset_energy_sampling",
]

_LEDGER: contextvars.ContextVar[EnergyLedger | None] = contextvars.ContextVar(
    "batcher_energy_ledger", default=None
)


def active_ledger() -> EnergyLedger | None:
    """The ledger stages are recording into, or `None` outside an `energy_scope`.

    Returns:
        The active ledger, or `None` when nothing is collecting energy.
    """
    return _LEDGER.get()


@contextlib.contextmanager
def energy_scope(ledger: EnergyLedger | None = None) -> Iterator[EnergyLedger]:
    """Collect the energy of every stage inside the block into one ledger.

    A scope opened inside another folds into it when it closes, so bracketing a sub-pipeline
    does not hide its stages from the outer run's total.

    The scope is a `ContextVar`, which bounds what it can see: a stage running in a thread the
    block did not start, or in a Ray worker, has its own (empty) context and records nothing
    here. That is why a distributed run's energy has to be folded from the workers' own
    ledgers rather than collected by the driver holding a scope open.

    Args:
        ledger: An existing ledger to append to, or `None` for a fresh one.

    Yields:
        The ledger being filled, so the caller can report from it after the block.
    """
    parent = _LEDGER.get()
    target = ledger if ledger is not None else EnergyLedger()
    token = _LEDGER.set(target)
    try:
        yield target
    finally:
        _LEDGER.reset(token)
        # A scope opened inside another folds into it on the way out, so a caller that
        # brackets a sub-pipeline still sees those stages in the outer run's total. The fold
        # is the same mergeable operation a distributed run uses on its workers' ledgers.
        if parent is not None and target is not parent and target.stages:
            parent.merge(target)


@dataclass
class StageMeter:
    """The handle a measured stage reports its output through.

    A stage knows what it produced; the meter knows what it cost. Rows and tokens are
    accumulated rather than assigned, so a stage that emits in batches can report each one.

    Attributes:
        stage: Stage identifier, conventionally `"Kind#id"`.
        accelerator_type: Device model the stage ran on, `""` for a CPU stage.
        device_count: Devices held for the stage's duration.
        rows: Rows emitted so far.
        tokens: Tokens generated so far.
    """

    stage: str
    accelerator_type: str = ""
    device_count: int = 0
    rows: int = 0
    tokens: int = 0

    def add_rows(self, count: int) -> None:
        """Add to the rows this stage has emitted.

        Args:
            count: Rows produced since the last call; non-positive values are ignored.
        """
        if count > 0:
            self.rows += count

    def add_tokens(self, count: int) -> None:
        """Add to the tokens this stage has generated.

        Args:
            count: Tokens produced since the last call; non-positive values are ignored.
        """
        if count > 0:
            self.tokens += count


#: The last telemetry reading and when it was taken, so a stage that runs many times a second
#: does not hammer NVML. Two readings per stage is nothing for a stage that runs for seconds;
#: it is the entire cost for one that runs for milliseconds, which the per-batch GPU kernels do.
_LAST_DRAW: list[tuple[float, tuple[float, float, bool]]] = []


def _sampled_draw() -> tuple[float, float, bool]:
    """`_draw()`, at most once per `accelerator.energy.telemetry_interval_s`.

    The cached reading is deliberately reused rather than re-read: within one interval a
    device's draw is close to constant, and a per-invocation NVML round trip on a kernel that
    runs in milliseconds costs more than the measurement is worth.
    """
    from batcher.config import active_config

    interval = active_config().accelerator.energy.telemetry_interval_s
    now = time.monotonic()
    if _LAST_DRAW and now - _LAST_DRAW[0][0] < interval:
        return _LAST_DRAW[0][1]
    reading = _draw()
    _LAST_DRAW[:] = [(now, reading)]
    return reading


#: The last energy-counter reading and when it was taken, cached on the same interval and for
#: the same reason as `_LAST_DRAW`.
_LAST_ENERGY: list[tuple[float, tuple]] = []


def _sampled_energy() -> tuple:
    """The devices' integrated joule counters, at most once per telemetry interval.

    Shares `_LAST_DRAW`'s caching discipline, which has a consequence worth stating: a stage
    shorter than the interval sees the *same* cached reading at both ends, the delta is zero,
    and the caller falls back to the sampled-power figure. That is the correct outcome rather
    than a limitation — a stage too short to span a sampling interval is also too short for two
    NVML round trips per device to be worth paying.
    """
    from batcher.config import active_config

    interval = active_config().accelerator.energy.telemetry_interval_s
    now = time.monotonic()
    if _LAST_ENERGY and now - _LAST_ENERGY[0][0] < interval:
        return _LAST_ENERGY[0][1]
    try:
        from batcher._internal.hardware.telemetry.energy import device_energy

        reading = device_energy()
    except Exception:
        reading = ()
    _LAST_ENERGY[:] = [(now, reading)]
    return reading


def _integrated_joules(before: tuple, after: tuple) -> float | None:
    """Exact joules across an interval from the hardware counters, or `None` when unavailable.

    Preferred over `watts * seconds` whenever it answers, because the sampled figure assumes
    the draw between the samples was the mean of them and a GPU stage is the workload that
    violates that hardest. `None` covers every reason it cannot answer — no counter on the part,
    no driver, a driver reload mid-stage, or a stage shorter than the sampling interval — and
    each of those means the caller keeps the sampled figure rather than recording nothing.
    """
    if not before or not after:
        return None
    try:
        from batcher._internal.hardware.telemetry.energy import interval_energy_joules

        joules = interval_energy_joules(before, after)
    except Exception:
        return None
    return joules if joules and joules > 0 else None


def reset_energy_sampling() -> None:
    """Forget the cached telemetry reading, so the next measurement re-reads the devices.

    The counterpart of `carbonite.memory.probe.reset_memory_sampling`. Needed whenever the
    sampling interval changes under a running process — a `config_context` that tightens it —
    and by any test that fakes a device draw.
    """
    _LAST_DRAW.clear()
    _LAST_ENERGY.clear()


def _draw() -> tuple[float, float, bool]:
    """`(watts, utilization, measured)` across this host's devices, right now.

    `measured` is False when NVML answered nothing, in which case the two figures are zero and
    the caller falls back to the datasheet model.
    """
    try:
        from batcher._internal.hardware.nvml import device_telemetry

        readings = device_telemetry()
    except Exception:
        return 0.0, 0.0, False
    if not readings:
        return 0.0, 0.0, False
    watts = sum(r.power_watts for r in readings)
    util = sum(r.sm_utilization for r in readings) / len(readings)
    return watts, util, watts > 0


@contextlib.contextmanager
def measure_stage(
    stage: str,
    *,
    accelerator_type: str = "",
    device_count: int = 0,
    utilization: float = 1.0,
) -> Iterator[StageMeter]:
    """Bracket a stage and record what it drew into the active ledger.

    A no-op when no `energy_scope` is open or when `accelerator.energy.accounting` is off, so
    the call is safe to leave on a path that usually runs without accounting.

    Args:
        stage: Stage identifier, conventionally `"Kind#id"`.
        accelerator_type: Device model the stage runs on, `""` for a CPU stage.
        device_count: Devices the stage holds.
        utilization: Utilization to assume when telemetry cannot measure it.

    Yields:
        A `StageMeter` the stage reports its rows and tokens through.
    """
    meter = StageMeter(stage, accelerator_type, device_count)
    ledger = active_ledger()
    if ledger is None:
        yield meter
        return
    from batcher.config import active_config

    if not active_config().accelerator.energy.accounting:
        yield meter
        return

    start_watts, start_util, start_ok = _sampled_draw()
    start_energy = _sampled_energy()
    started = time.perf_counter()
    try:
        yield meter
    finally:
        seconds = max(0.0, time.perf_counter() - started)
        end_watts, end_util, end_ok = _sampled_draw()
        exact = _integrated_joules(start_energy, _sampled_energy())
        measured = start_ok and end_ok
        if measured:
            watts = (start_watts + end_watts) / 2.0
            util = (start_util + end_util) / 2.0
        else:
            from batcher.plan.energy.power import device_power_watts

            util = min(1.0, max(0.0, utilization))
            watts = device_power_watts(accelerator_type, util, include_host=True) * max(
                0, device_count
            )
        ledger.record(
            StageEnergy(
                stage=stage,
                accelerator_type=accelerator_type,
                device_count=device_count,
                seconds=seconds,
                utilization=util,
                joules=exact if exact is not None else max(0.0, watts) * seconds,
                rows=meter.rows,
                tokens=meter.tokens,
                # The counter is a measurement whether or not the power sampling worked, so a
                # stage the driver metered exactly is `measured` even when `_draw` reported
                # nothing — which is the case on a part that publishes the energy counter and
                # refuses instantaneous power, and there are several.
                measured=measured or exact is not None,
                integrated=exact is not None,
            )
        )
