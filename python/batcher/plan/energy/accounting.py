"""Per-stage energy accounting — the ledger a run fills in and a report reads out.

A GPU-hour is the unit a datacenter sells; a joule is the unit it buys. Between them sits the
question this module answers: which stage of which query spent the energy, and what did it
produce for it. That is what makes an efficiency number actionable — a pipeline reported as
"4.1 MJ" tells an engineer nothing, while the same pipeline reported as "decode: 3.4 MJ at
0.31 rows per joule, embed: 0.7 MJ at 940 rows per joule" says exactly where to look.

The ledger is a plain accumulator in the neutral `plan` layer, so Core can fill it while
executing, Kyber can read the previous run's figures as learned statistics, and `observe` can
render it, without any of those layers importing another. It holds only completed records, so
it never has to be thread-safe on a hot path: a stage appends once, when it finishes.

**Efficiency is reported as work per joule, never joules per unit of work.** The inverse form
divides by a work count that is legitimately zero for a filtered-out partition, and a stage
that produced nothing is not infinitely inefficient — it is a stage with no efficiency figure,
which `None` says and a division cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["EnergyLedger", "StageEnergy", "merge_ledgers"]


@dataclass(frozen=True, slots=True)
class StageEnergy:
    """What one stage of one run drew, and what it produced for it.

    Attributes:
        stage: Stage identifier, conventionally `"Kind#id"` to match `FeasibilityVerdict`.
        accelerator_type: Device model the stage ran on, or `""` for a CPU stage.
        device_count: Devices held for the duration.
        seconds: Wall-clock duration the devices were held, including any time the stage was
            idle while holding them — that time is charged, because the datacenter charges it.
        utilization: Mean device utilization over the duration, in [0, 1].
        joules: Energy attributed to the stage, IT load only.
        rows: Rows the stage emitted, `0` when not counted.
        tokens: Tokens the stage generated, `0` for a non-generative stage.
        measured: Whether `joules` came from a device power reading rather than from the
            datasheet model. A cost figure that cannot be told apart from an estimate is
            worth less than either, so the distinction travels with the record.
        integrated: Whether `joules` came from the driver's own hardware energy counter rather
            than from power sampled at the ends of the stage. Both are `measured`; only this
            one is *exact*. The difference is not cosmetic — sampling assumes the draw between
            the samples was the mean of them, and a GPU stage alternating between a staged
            transfer at 60 W and a kernel at 700 W violates that by an order of magnitude. A
            chargeback or a carbon figure should say which it had.
    """

    stage: str
    accelerator_type: str = ""
    device_count: int = 0
    seconds: float = 0.0
    utilization: float = 0.0
    joules: float = 0.0
    rows: int = 0
    tokens: int = 0
    measured: bool = False
    integrated: bool = False

    @property
    def rows_per_joule(self) -> float | None:
        """Rows emitted per joule, or `None` when the stage emitted nothing or drew nothing."""
        if self.joules <= 0 or self.rows <= 0:
            return None
        return self.rows / self.joules

    @property
    def tokens_per_joule(self) -> float | None:
        """Tokens generated per joule, or `None` when either figure is absent.

        The headline efficiency number for a generative fleet, and the one that makes two
        deployments comparable across device generations: a newer device that draws 40% more
        power while generating 2.5x the tokens is the cheaper machine to run, and only this
        ratio says so.
        """
        if self.joules <= 0 or self.tokens <= 0:
            return None
        return self.tokens / self.joules

    @property
    def idle_joules(self) -> float:
        """Energy spent holding the devices rather than computing on them.

        The waste a scheduler can actually remove — by packing the stage tighter, by releasing
        the device across a CPU-bound step, or by partitioning it — as distinct from the energy
        the computation genuinely required.

        Measured against the same work compressed to full utilization and the device then
        released: that hypothetical draws `tdp * u * seconds`, the stage actually draws
        `P(u) * seconds`, and the difference is the idle floor held across the slack,
        `idle * (1 - u) * seconds`. A fully fed device therefore wastes nothing, and a device
        held at zero utilization wastes its whole idle draw.
        """
        from batcher.plan.energy.power import device_power_watts

        if self.device_count <= 0 or self.seconds <= 0:
            return 0.0
        idle = device_power_watts(self.accelerator_type, 0.0)
        if idle <= 0:
            return 0.0
        slack = 1.0 - min(1.0, max(0.0, self.utilization))
        return idle * slack * self.seconds * self.device_count


@dataclass
class EnergyLedger:
    """Accumulated `StageEnergy` records for one run, with the roll-ups a report needs.

    Attributes:
        stages: Completed stage records, in completion order.
    """

    stages: list[StageEnergy] = field(default_factory=list)

    def record(self, stage: StageEnergy) -> None:
        """Append a completed stage's energy record.

        Args:
            stage: The record; stages may repeat, and repeats accumulate rather than replace.
        """
        self.stages.append(stage)

    @property
    def total_joules(self) -> float:
        """Total IT-load energy across every recorded stage."""
        return sum(s.joules for s in self.stages)

    @property
    def total_idle_joules(self) -> float:
        """Total energy spent holding devices rather than computing on them."""
        return sum(s.idle_joules for s in self.stages)

    @property
    def integrated_fraction(self) -> float:
        """Fraction of the run's energy that came from a hardware counter, in [0, 1].

        The confidence qualifier a total needs before it is quoted. A run at `1.0` has an exact
        figure; a run at `0.0` has one assembled from power samples and datasheet models, which
        is fine for comparing two runs on the same hardware and not fine for a bill. Weighted by
        joules rather than by stage count, because one long stage dominating the total decides
        how trustworthy the total is regardless of how many short ones surround it.
        """
        total = self.total_joules
        if total <= 0:
            return 0.0
        return sum(s.joules for s in self.stages if s.integrated) / total

    @property
    def total_tokens(self) -> int:
        """Total tokens generated across every recorded stage."""
        return sum(s.tokens for s in self.stages)

    @property
    def total_rows(self) -> int:
        """Total rows emitted across every recorded stage."""
        return sum(s.rows for s in self.stages)

    @property
    def measured_joules(self) -> float:
        """Energy that came from a device reading rather than the datasheet model."""
        return sum(s.joules for s in self.stages if s.measured)

    def merge(self, other: EnergyLedger) -> EnergyLedger:
        """Fold another ledger's records into this one, returning self.

        Energy is mergeable in exactly the way the engine's stateful operators are: a run's
        total is the sum over its workers, and every roll-up here is a sum or a ratio of sums,
        so combining is associative and commutative and the merged figures equal what a
        single-node run would have reported. That is what lets a distributed run report one
        honest energy number instead of the driver's own share of it.

        Args:
            other: A ledger from another worker or another stage of the same run.

        Returns:
            This ledger, so merges chain over a list of per-worker ledgers.
        """
        self.stages.extend(other.stages)
        return self

    def by_device(self) -> dict[str, float]:
        """Energy grouped by accelerator model.

        Returns:
            Device model name to joules; a CPU stage is keyed by the empty string.
        """
        out: dict[str, float] = {}
        for s in self.stages:
            out[s.accelerator_type] = out.get(s.accelerator_type, 0.0) + s.joules
        return out

    def by_stage(self) -> list[StageEnergy]:
        """One record per stage name, with repeats rolled up, in first-seen order.

        A kernel that runs once per morsel appends a record per call, which is right for
        accounting and useless for reading: the report wants the stage, not its invocations.
        Durations and energy sum; utilization is averaged *weighted by duration*, because a
        plain mean would let a thousand idle microseconds outvote a second of real work.
        """
        order: list[str] = []
        acc: dict[str, dict] = {}
        for s in self.stages:
            if s.stage not in acc:
                order.append(s.stage)
                acc[s.stage] = {
                    "device": s.accelerator_type,
                    "devices": s.device_count,
                    "seconds": 0.0,
                    "util_seconds": 0.0,
                    "joules": 0.0,
                    "rows": 0,
                    "tokens": 0,
                    "measured": True,
                }
            entry = acc[s.stage]
            entry["seconds"] += s.seconds
            entry["util_seconds"] += s.utilization * s.seconds
            entry["joules"] += s.joules
            entry["rows"] += s.rows
            entry["tokens"] += s.tokens
            entry["devices"] = max(entry["devices"], s.device_count)
            entry["measured"] = entry["measured"] and s.measured
        out: list[StageEnergy] = []
        for name in order:
            e = acc[name]
            seconds = e["seconds"]
            out.append(
                StageEnergy(
                    stage=name,
                    accelerator_type=e["device"],
                    device_count=e["devices"],
                    seconds=seconds,
                    utilization=(e["util_seconds"] / seconds) if seconds > 0 else 0.0,
                    joules=e["joules"],
                    rows=e["rows"],
                    tokens=e["tokens"],
                    measured=e["measured"],
                )
            )
        return out

    def hottest_stage(self) -> StageEnergy | None:
        """The single stage that drew the most energy, or `None` for an empty ledger.

        Where to look first: energy in a pipeline is almost always concentrated, so the
        largest consumer is nearly always the only one worth optimizing. Rolled up by stage,
        so a kernel that runs a thousand times is compared as one stage rather than as a
        thousand small ones that each lose to a single large stage.
        """
        return max(self.by_stage(), key=lambda s: s.joules, default=None)

    def tokens_per_joule(self) -> float | None:
        """Whole-run generative efficiency, or `None` when no tokens or no energy were recorded."""
        joules, tokens = self.total_joules, self.total_tokens
        if joules <= 0 or tokens <= 0:
            return None
        return tokens / joules

    def rows_per_joule(self) -> float | None:
        """Whole-run relational efficiency, or `None` when no rows or no energy were recorded."""
        joules, rows = self.total_joules, self.total_rows
        if joules <= 0 or rows <= 0:
            return None
        return rows / joules

    def idle_fraction(self) -> float:
        """Share of total energy spent holding idle devices, in [0, 1].

        The single number that says whether a fleet is under-fed. Above roughly a third, the
        pipeline is starving its devices and the fix is upstream — more prefetch, larger
        batches, or fewer devices — not a faster kernel.
        """
        total = self.total_joules
        return self.total_idle_joules / total if total > 0 else 0.0

    def summary(self) -> dict[str, float]:
        """A flat, JSON-safe roll-up for logs, metrics sinks, and the dashboard.

        Returns:
            Total energy, idle energy and fraction, the share of it that was measured
            rather than modelled, row and token counts, and the two efficiency ratios
            (omitted when undefined rather than reported as zero).
        """
        out: dict[str, float] = {
            "joules": self.total_joules,
            "idle_joules": self.total_idle_joules,
            "idle_fraction": self.idle_fraction(),
            "rows": float(self.total_rows),
            "tokens": float(self.total_tokens),
            "stages": float(len(self.stages)),
        }
        if self.total_joules > 0:
            out["measured_fraction"] = self.measured_joules / self.total_joules
        tpj, rpj = self.tokens_per_joule(), self.rows_per_joule()
        if tpj is not None:
            out["tokens_per_joule"] = tpj
        if rpj is not None:
            out["rows_per_joule"] = rpj
        return out


def merge_ledgers(ledgers: list[EnergyLedger] | tuple[EnergyLedger, ...]) -> EnergyLedger:
    """Combine per-worker ledgers into the one a distributed run reports.

    The `combine` half of the same mergeable shape the stateful operators use: partial ledgers
    from any number of workers fold into a single result that equals the single-node one, in
    any order.

    Args:
        ledgers: Per-worker ledgers; an empty sequence yields an empty ledger.

    Returns:
        A new ledger holding every record, leaving the inputs unmodified.
    """
    out = EnergyLedger()
    for ledger in ledgers:
        out.stages.extend(ledger.stages)
    return out
