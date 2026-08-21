"""What the spill device *measured*, against what its class claimed.

`storage_cost.spill_device_factor` prices a spilled byte from the device class read off
`/sys/block`. That is a structural identification, not a throughput measurement, and it is
systematically biased in one direction: `device_class` resolves a composite device to the
**slowest** class underneath it, and treats an NVMe-oF namespace and an iSCSI LUN as network
storage on the strength of a transport string. Both are the right call when they are right,
and both are unfalsifiable without a measurement — so a local RAID0 of four NVMe reported as
`rotational` prices a spilled byte thirty times too high, and every out-of-core plan on that
machine is contorted to avoid a spill the device would have absorbed comfortably.

This is the falsification. Core already records `spill_bytes` and `t_op_ms` for every operator
that spilled, so the fleet's own history says what the device delivered. Kyber reads it here
and consumes it; it records nothing (`core` measures, `kyber` decides).

## Why the correction is one-directional, and why that is the useful direction

The only clock on the record is `t_op_ms`, the operator's **whole** wall time, which includes
its compute. So `spill_bytes / t_op_ms` is not the device's throughput — it is a *lower bound*
on it, and the bound can be arbitrarily loose when an operator is compute-heavy.

A lower bound supports exactly one inference, and refuses the other:

* measured rate **high** — the device demonstrably moved that many bytes per millisecond, so
  it is at least that fast. A class claiming it is slower than that is wrong, and the factor
  may safely come down.
* measured rate **low** — says nothing. The operator may simply have been compute-bound. The
  factor is left alone.

That asymmetry lines up with the bias it is correcting: `device_class` errs toward pessimism
by construction, so the direction this can prove is the direction that is wrong in practice.
A device that is genuinely slower than its class claims is not detectable from this signal and
is not guessed at.

The result is shrunk toward the class factor by how much evidence there is (the same estimator
`calibration` uses) and never falls below `1.0`, so a cold store, a handful of samples, or a
single anomalous run cannot move a plan.
"""

from __future__ import annotations

import weakref
from statistics import median

from batcher._internal.hardware.storage import FLASH_SPILL_MBPS
from batcher._internal.logging import note_suppressed
from batcher.config import Config, active_config
from batcher.kyber.calibration import shrink
from batcher.metadata import MetadataHub

__all__ = ["learned_spill_factor", "measured_spill_mbps"]

#: How much the device class is worth, in pseudo-samples, against the measurement. Higher than
#: the cost coefficients' prior strength on purpose: a coefficient that drifts costs plan
#: quality, while this decides whether a plan may spill at all, and the class is a real
#: reading of a real device rather than a shipped guess.
_PRIOR_STRENGTH = 24.0

#: A spilling operator that moved less than this is not a throughput measurement — it is
#: mostly fixed overhead (file creation, a header, an fsync), and its implied rate understates
#: the device by an arbitrary amount. One morsel's worth of spill is the floor for a sample.
_MIN_SAMPLE_BYTES = 4 * 1024 * 1024

#: Per-hub memo, keyed weakly so a dropped hub evicts its entry. Value is
#: `(hub.version_bucket, storage_class, machine class, factor)`. Mirrors `calibration` and
#: `cpu_shares`: planning must not re-scan the whole op-stats history on every query.
#:
#: The **machine class** is in the key for the reason it is in theirs. One hub serves several
#: across a session — a driver planning for its workers, then for itself — and this factor is
#: fitted from spills that happened on one specific device. Keyed without it, whichever class
#: was asked first answered for every other, so a driver on a network volume could hand its
#: workers on local NVMe a thirtyfold spill price, or the reverse: a plan allowed out-of-core
#: on the strength of a device it will never touch.
_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, str, str, float | None]] = (
    weakref.WeakKeyDictionary()
)

#: Recompute only after this many new feedback rows, so per-query planning cost stays flat.
_REFRESH_AFTER = 64


def measured_spill_mbps(
    hub: MetadataHub | None,
    hw_fingerprint: str | None = None,
    min_samples: int = 1,
) -> float | None:
    """The median lower bound on spill throughput this machine class has demonstrated, in MB/s.

    A *lower* bound, not an estimate: the divisor is the operator's whole wall time, so a
    compute-heavy operator reads far below what the device can do. See the module docstring
    for what may and may not be concluded from that.

    Args:
        hub: The metadata hub holding the measured history, or `None`.
        hw_fingerprint: The machine class whose spills to read. `None` reads this process's
            own class, which is right single-node; a driver planning for workers of a
            different class must pass theirs, or it reads spills that happened on the wrong
            device entirely.
        min_samples: How many qualifying spills are needed before a median is returned.
            Below it the answer is `None` — one operator that happened to be IO-bound is not
            evidence about a device.

    Returns:
        The median measured rate in MB/s, or `None` when too few spills have been recorded.
    """
    if hub is None:
        return None
    try:
        by_kind = hub.op_stats_by_kind(hw_fingerprint)
    except Exception as exc:  # pragma: no cover - learning must never break a query
        note_suppressed("kyber", "read spill history", exc)
        return None
    rates: list[float] = []
    for rows in by_kind.values():
        for row in rows:
            spilled = float(row.get("spill_bytes") or 0.0)
            millis = float(row.get("t_op_ms") or 0.0)
            if spilled >= _MIN_SAMPLE_BYTES and millis > 0.0:
                # bytes/ms is MB/s to within the 1e6-vs-2^20 convention, which is the same
                # convention `FLASH_SPILL_MBPS` is stated in, so the ratio below is consistent.
                rates.append(spilled / millis / 1e3)
    if len(rates) < max(1, min_samples):
        return None
    return median(rates)


def learned_spill_factor(
    hub: MetadataHub | None,
    storage_class: str = "",
    config: Config | None = None,
    hw_fingerprint: str | None = None,
) -> float | None:
    """The spill-device factor corrected by what the device was measured to deliver.

    `None` whenever the measurement cannot improve on the class — no hub, too few spills, or
    a device already priced at or below what it demonstrated. A caller that gets `None` must
    keep `storage_cost.spill_device_factor`, which is what every existing call site does by
    construction.

    Args:
        hub: The metadata hub holding the measured history, or `None`.
        storage_class: The device class whose factor is being corrected, from
            `HardwareProfile.storage_class`. `""` resolves this process's own spill directory.
        config: Config supplying the sample-count gate; the active one when `None`.
        hw_fingerprint: The machine class whose spills to read; this process's when `None`.

    Returns:
        The corrected factor, at least `1.0`, or `None` to keep the class factor.
    """
    from batcher.kyber.storage_cost import spill_device_factor

    cfg = config or active_config()
    min_samples = max(1, cfg.optimizer.cost_calibration_min_samples)
    claimed = spill_device_factor(storage_class)
    if claimed <= 1.0:
        # Already at the floor: there is nothing a "the device is faster than you thought"
        # measurement could correct, so do not pay for the history scan.
        return None
    machine = hw_fingerprint or ""
    version_bucket = hub.version // _REFRESH_AFTER if hub is not None else 0
    key = (version_bucket, storage_class, machine)
    cached = _CACHE.get(hub) if hub is not None else None
    if cached is not None and cached[:3] == key:
        return cached[3]
    factor = _correct(hub, claimed, hw_fingerprint, min_samples)
    if hub is not None:
        _CACHE[hub] = (*key, factor)
    return factor


def _correct(
    hub: MetadataHub | None, claimed: float, hw_fingerprint: str | None, min_samples: int
) -> float | None:
    """The shrunk correction to `claimed`, or None when the evidence does not support one."""
    measured = measured_spill_mbps(hub, hw_fingerprint, min_samples)
    if measured is None or measured <= 0.0:
        return None
    # The factor the measurement *proves* is achievable. Flash is factor 1.0 by definition of
    # the table, so a device sustaining a fifth of flash proves a factor no worse than 5.
    implied = max(1.0, FLASH_SPILL_MBPS / measured)
    if implied >= claimed:
        return None  # the measurement is consistent with the class; nothing to correct
    n = _sample_count(hub, hw_fingerprint)
    corrected = shrink(implied, claimed, n, _PRIOR_STRENGTH)
    return max(1.0, min(claimed, corrected))


def _sample_count(hub: MetadataHub | None, hw_fingerprint: str | None) -> int:
    """How many spill samples backed the median — the evidence weight for shrinkage."""
    if hub is None:
        return 0
    try:
        by_kind = hub.op_stats_by_kind(hw_fingerprint)
    except Exception as exc:  # pragma: no cover
        note_suppressed("kyber", "count spill history", exc)
        return 0
    return sum(
        1
        for rows in by_kind.values()
        for row in rows
        if float(row.get("spill_bytes") or 0.0) >= _MIN_SAMPLE_BYTES
        and float(row.get("t_op_ms") or 0.0) > 0.0
    )
