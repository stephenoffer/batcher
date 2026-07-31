"""One verdict per device: what is actually limiting it, and therefore what to change.

Every other module here reports a number. Numbers are not the thing a caller needs — the same
80% SM utilization means "this device is the constraint, buy more of them" on one workload and
"this device is spinning on a kernel that should have been one instead of ten" on another, and a
report that hands over the number and stops has moved the diagnosis onto the reader.

The diagnosis is mechanical once enough readings are in hand, and this is that mechanism. It
combines the SM duty cycle, the memory duty cycle, the PCIe link utilization, the clock clamp
state, and the fixed-function engine counters into one of a small set of verdicts, each of which
names a different fix:

| Verdict | What it means | What to change |
|---|---|---|
| `compute_bound` | SMs busy, memory and bus quiet | Nothing here; add devices |
| `memory_bound` | Memory busy, SMs waiting on it | A smaller working set, fewer passes |
| `transfer_bound` | The host link is saturated | Keep data resident; pinned staging |
| `starved` | Everything quiet, work outstanding | Deeper prefetch, more in flight |
| `throttled` | The driver is clamping the clocks | Cooling, or the power limit |
| `contended` | Another process holds the device | Fractional sizing, or elsewhere |
| `codec_bound` | A fixed-function engine is the ceiling | More devices, or another codec |

**Ordering is the whole design.** A device can be several of these at once, and the verdicts are
not equally actionable. A throttled device reports `throttled` even at 100% SM utilization,
because "compute bound" on a clamped device sends the reader off to buy hardware that would also
be clamped. Contention outranks starvation for the same reason: a starved device that is starved
because a neighbour has it is not fixed by prefetching harder.

**A verdict is never returned from one sample.** Every entry point here takes summaries from
`telemetry.sampler`, not instantaneous readings, because the single most common way to
mis-diagnose a GPU stage is to sample it at the moment it drained.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.telemetry.sampler import MetricSummary

__all__ = [
    "VERDICT_ADVICE",
    "Bottleneck",
    "classify_device",
    "fleet_verdict",
]

#: What to do about each verdict, in the words the fix is written in. Held here rather than in
#: the reporting layer so every surface that renders a verdict — the terminal report, the
#: dashboard, the query-optimizer explanation — says the same thing about it.
VERDICT_ADVICE: dict[str, str] = {
    "compute_bound": "the device is the constraint; add devices or reduce work per row",
    "memory_bound": "shrink the working set or fuse passes; more parallelism will not help",
    "transfer_bound": "keep data device-resident, stage from pinned host memory, or read direct",
    "starved": "deepen the prefetch or raise the in-flight batch count; the feed is the limit",
    "throttled": "the driver is clamping clocks; check cooling and the enforced power limit",
    "contended": "another process holds this device; size fractionally or place elsewhere",
    "codec_bound": "a fixed-function engine is saturated; spread across devices",
    "occupancy_limited": "the kernel holds too few warps per SM; more work per call will not help",
    "unknown": "not enough telemetry to say; sample for longer or check driver visibility",
}

#: Duty cycle at or above which a unit counts as the constraint.
_BUSY = 0.75

#: Duty cycle below which a unit counts as idle across the window.
_QUIET = 0.25


@dataclass(frozen=True, slots=True)
class Bottleneck:
    """What limited one device across a sampling window.

    Attributes:
        index: NVML device index.
        verdict: One of the keys of `VERDICT_ADVICE`. `"unknown"` when the window held too
            little to say, which is a real answer and not a fallback — it means the reader
            should sample longer, not that the device is fine.
        confidence: How separated the winning signal was from the runners-up, in [0, 1]. Low
            confidence on a real verdict means the device is close to a second limit and fixing
            the first will expose it, which is worth saying before someone spends a week on it.
        detail: A short phrase naming the specific measurement behind the verdict.
    """

    index: int
    verdict: str = "unknown"
    confidence: float = 0.0
    detail: str = ""

    @property
    def advice(self) -> str:
        """What to change, in one line, from `VERDICT_ADVICE`."""
        return VERDICT_ADVICE.get(self.verdict, VERDICT_ADVICE["unknown"])

    @property
    def actionable(self) -> bool:
        """Whether the verdict names something the caller can act on.

        False for `"unknown"` and for `"compute_bound"`, which is the one verdict that says the
        device is already doing its job — there is nothing to fix, only more hardware to buy.
        """
        return self.verdict not in ("unknown", "compute_bound")


def _confidence(winner: float, runner_up: float) -> float:
    """How separated a winning signal is from the next one, in [0, 1].

    A margin rather than a probability: it is the fraction of the winning signal not shared with
    the runner-up, so two signals at the same level report zero confidence and a clear winner
    reports close to one. That is exactly the quantity a reader needs — it says whether fixing
    the named limit will reveal a second one immediately.
    """
    if winner <= 0:
        return 0.0
    return max(0.0, min(1.0, (winner - runner_up) / winner))


def classify_device(
    index: int,
    sm: MetricSummary,
    memory: MetricSummary | None = None,
    pcie: MetricSummary | None = None,
    throttled: MetricSummary | None = None,
    shared: bool | None = None,
    codec: MetricSummary | None = None,
    occupancy: MetricSummary | None = None,
) -> Bottleneck:
    """Diagnose what limited one device across a sampling window.

    Args:
        index: NVML device index the summaries describe.
        sm: SM-utilization summary. The one required input; without it there is no verdict.
        memory: Memory-controller duty-cycle summary, or `None` when not sampled.
        pcie: PCIe link utilization summary, or `None` when not sampled.
        throttled: Summary of the throttled flag sampled as 0/1, or `None`. Its `mean` is the
            fraction of the window the device spent clamped.
        shared: Whether another process was active on the device, from
            `telemetry.processes.device_shared_with_others`. `None` means unknowable, which is
            treated as not shared — the alternative would report every containerized device as
            contended.
        codec: Fixed-function engine duty-cycle summary, or `None`.
        occupancy: Resident-warp occupancy summary from `telemetry.dcgm`, or `None` when DCGM is
            unavailable — which is most hosts. Supplying it is what turns a `compute_bound`
            verdict into an actionable one: a device whose SMs are busy while holding a fraction
            of the warps they could is limited by the kernel's shape, not by the device.

    Returns:
        The verdict, with `"unknown"` when `sm` holds no samples.
    """
    if sm.samples == 0:
        return Bottleneck(index=index, verdict="unknown", detail="no samples")
    bus = pcie.mean if pcie and pcie.samples else 0.0
    mem = memory.mean if memory and memory.samples else 0.0
    engine = codec.mean if codec and codec.samples else 0.0
    clamp = throttled.mean if throttled and throttled.samples else 0.0
    return _external_limit(index, sm, bus, mem, engine, clamp, shared) or _sm_limit(
        index, sm, bus, mem, engine, occupancy
    )


def _external_limit(
    index: int,
    sm: MetricSummary,
    bus: float,
    mem: float,
    engine: float,
    clamp: float,
    shared: bool | None,
) -> Bottleneck | None:
    """The verdicts caused by something other than the SMs, in precedence order, or `None`.

    All of these outrank every SM-side verdict, and the order between them is the order in which
    fixing one would reveal the next. A clamped device leads because "compute bound" on clamped
    hardware sends the reader to buy more hardware that would also be clamped.
    """
    if clamp > 0.1:
        return Bottleneck(
            index=index,
            verdict="throttled",
            confidence=min(1.0, clamp / 0.5),
            detail=f"clocks clamped for {clamp:.0%} of the window",
        )
    if shared and sm.mean < _BUSY:
        return Bottleneck(
            index=index,
            verdict="contended",
            confidence=_confidence(1.0 - sm.mean, sm.mean),
            detail="another process was active on the device",
        )
    if bus >= _BUSY and bus > sm.mean:
        return Bottleneck(
            index=index,
            verdict="transfer_bound",
            confidence=_confidence(bus, max(sm.mean, mem)),
            detail=f"host link at {bus:.0%} against SMs at {sm.mean:.0%}",
        )
    if engine >= _BUSY and engine > sm.mean:
        return Bottleneck(
            index=index,
            verdict="codec_bound",
            confidence=_confidence(engine, sm.mean),
            detail=f"fixed-function engine at {engine:.0%}",
        )
    if mem >= _BUSY and mem > sm.mean:
        return Bottleneck(
            index=index,
            verdict="memory_bound",
            confidence=_confidence(mem, sm.mean),
            detail=f"memory at {mem:.0%} against SMs at {sm.mean:.0%}",
        )
    return None


def _sm_limit(
    index: int,
    sm: MetricSummary,
    bus: float,
    mem: float,
    engine: float,
    occupancy: MetricSummary | None,
) -> Bottleneck:
    """The verdict from the SM signal alone, once nothing external is the limit."""
    if sm.mean >= _BUSY:
        warps = occupancy.mean if occupancy and occupancy.samples else 0.0
        if occupancy and occupancy.samples and warps < 0.3:
            # Busy SMs holding a third of the warps they could: the scheduler has nothing left
            # to place, and the limit is the kernel's register or shared-memory footprint. This
            # outranks `compute_bound` because the two prescribe opposite things — one says buy
            # devices, the other says the devices already bought are a third used.
            return Bottleneck(
                index=index,
                verdict="occupancy_limited",
                confidence=_confidence(sm.mean - warps, warps),
                detail=f"SMs at {sm.mean:.0%} with occupancy at {warps:.0%}",
            )
        return Bottleneck(
            index=index,
            verdict="compute_bound",
            confidence=_confidence(sm.mean, max(mem, bus)),
            detail=f"SMs at {sm.mean:.0%} for {sm.above_fraction:.0%} of the window",
        )
    if sm.mean <= _QUIET and max(mem, bus, engine) <= _QUIET:
        return Bottleneck(
            index=index,
            verdict="starved",
            confidence=_confidence(1.0 - sm.mean, sm.mean),
            detail=(
                f"SMs idle for {1.0 - sm.above_fraction:.0%} of the window with nothing else busy"
            ),
        )
    if sm.bursty and max(mem, bus, engine) < _BUSY:
        # Mid-range mean, but the device swung across most of its range: it reached saturation
        # and fell back to idle repeatedly, with nothing else busy to explain the gaps. That is
        # the feed, and it is the one case a mean alone cannot see — averaging a saturated
        # device and an idle one produces exactly the mid-range figure that falls through below.
        return Bottleneck(
            index=index,
            verdict="starved",
            confidence=_confidence(sm.peak - sm.trough, sm.mean),
            detail=f"SM utilization swung {sm.trough:.0%}-{sm.peak:.0%} across the window",
        )
    # Everything mid-range and nothing dominant. Reporting a verdict here would be picking the
    # largest of several similar numbers and dressing it as a finding, which is how a report
    # sends someone to optimize the wrong thing.
    return Bottleneck(
        index=index,
        verdict="unknown",
        confidence=0.0,
        detail="no unit dominated the window",
    )


def fleet_verdict(verdicts: tuple[Bottleneck, ...]) -> Bottleneck | None:
    """The one finding worth leading a report with, across a fleet.

    Not the most common verdict and not the worst device: the most *actionable* one, because a
    report exists to change what someone does. A fleet where seven devices are compute bound and
    one is throttled leads with the throttled one — the seven are working as intended and the
    eighth is silently costing a third of a node.

    Args:
        verdicts: Per-device verdicts.

    Returns:
        The highest-confidence actionable verdict, or `None` when every device is either
        compute bound or unclassifiable, which is the healthy fleet.
    """
    actionable = [v for v in verdicts if v.actionable]
    if not actionable:
        return None
    return max(actionable, key=lambda v: v.confidence)
