"""Collecting what a pipeline drew, and folding the measurement back into what is learned.

The conductor's half of the energy loop: Core measures each stage, this scope collects
them, and a *measured* stage is folded into the hub on the way out so the next plan is
ranked against what this fleet actually does rather than against a datasheet.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from batcher.plan.energy import EnergyLedger

__all__ = ["measure_energy"]


@contextlib.contextmanager
def measure_energy() -> Iterator[EnergyLedger]:
    """Collect the energy every accelerator stage inside the block drew.

    A GPU-hour is what a fleet is billed; joules are what it buys. This is how a pipeline
    reports the second: each accelerator stage that runs inside the block records what it
    drew, measured from device readings where NVML is available and modelled from the device
    table where it is not, and the ledger tells the two apart.

    The ledger is filled as the block runs, so read it after the block. Render it with
    :func:`batcher.observe.format_energy_report`, or take the ratios off it directly.
    Recording is skipped entirely when `accelerator.energy.accounting` is off.

    On the way out, every *measured* stage is folded into the learned statistics, so the next
    run's device choice is made against what this fleet delivers rather than against a
    datasheet ratio. Modelled stages are not: learning from them would teach the optimizer its
    own assumptions back.

    Returns:
        A context manager yielding the `EnergyLedger` the block's stages record into.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> with bt.measure_energy() as energy:
            ...     _ = bt.from_pydict({"x": [1, 2, 3]}).to_pydict()
            >>> energy.total_joules >= 0.0
            True
    """
    from batcher.core.energy import energy_scope

    with energy_scope() as ledger:
        try:
            yield ledger
        finally:
            _learn_from(ledger)


def _learn_from(ledger: EnergyLedger) -> None:
    """Fold a completed run's measured efficiency into the learned statistics.

    The conductor's half of the loop the architecture describes: Core measured what each stage
    drew, and this is where that measurement reaches Kyber, so the next run's device choice is
    made against what this fleet actually delivers rather than against a datasheet ratio.
    Only *measured* records are folded — a modelled figure is the datasheet restated, and
    learning from it would teach the optimizer its own assumptions.

    Best-effort: a missing hub, an unreadable backend, or a failed write is skipped rather
    than raised, because a learning path must never fail a query.
    """
    if not ledger.stages:
        return
    try:
        from batcher.core.runtime import default_hub
        from batcher.kyber.gpu import record_measured_efficiency

        hub = default_hub()
        for stage in ledger.stages:
            if not stage.measured or not stage.accelerator_type:
                continue
            if stage.tokens > 0:
                record_measured_efficiency(
                    hub, stage.accelerator_type, stage.joules, stage.tokens, kind="tokens"
                )
            elif stage.rows > 0:
                record_measured_efficiency(
                    hub, stage.accelerator_type, stage.joules, stage.rows, kind="rows"
                )
    except Exception as exc:  # pragma: no cover - learning must never break a query
        from batcher._internal.logging import note_suppressed

        note_suppressed("api", "record measured efficiency", exc)
