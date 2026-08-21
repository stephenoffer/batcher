"""Scoping learned parameters to the machine that measured them.

Batcher learns from measurement: how many nanoseconds a join costs per probe row, how much
memory an aggregate holds per group, how large a UDF batch should be, how much VRAM a model
needs. Each of those is a property of *a workload on a machine*, and the machine half is not
optional. A per-row coefficient fitted on a 3 GHz AVX-512 core is wrong on a small ARM core by
several times over; a VRAM figure measured on an A100 is wrong on a T4 by five.

With one machine and one metadata store this never surfaced, because every measurement came
from the same hardware. It surfaces the moment a store is shared:

* a heterogeneous Ray cluster, where the driver and two node shapes all write the same store;
* an autoscaling group that mixes instance generations as capacity comes and goes;
* a laptop and CI sharing a checked-in store;
* one machine that simply changed — more RAM, a GPU added, a move from HDD to NVMe.

In each case the store blends measurements from unlike machines into a single fitted model
that is wrong for every one of them, with no error raised and nothing about the result that
looks off. `scoped` prevents the blend by putting the hardware fingerprint in the namespace,
so unlike machines write to different places and alike ones still share.

## What to scope, and what never to

The dividing line is what the stored value describes:

* **Scope anything measured in machine units** — nanoseconds, bytes of RAM or VRAM, device
  throughput, core counts, batch sizes chosen against them. These are statements about
  hardware and do not transfer.
* **Never scope a statement about data** — distinct counts, quantiles, column widths, join
  selectivities, skew. A column has the same distinct count whatever machine reads it, and
  scoping those would fragment the statistics that took the most work to collect, turning a
  well-calibrated fleet into N poorly-calibrated ones for no gain.

Getting this backwards in either direction is costly, so each call site says which side of the
line it is on.

## Why old values are dropped rather than migrated

A value already stored under an unscoped namespace has unknown provenance: nothing recorded
which machine measured it, and in the case this module exists to fix it is a blend of several.
Adopting it into whichever fingerprint asks first would hand one machine class a model built
from other machines' hardware, which is precisely the failure being removed. So the scoped
namespace starts empty and re-converges over the next few runs, which is fast because these
models are fitted from per-operator feedback that every query produces.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator

from batcher._internal.hardware import fingerprint

__all__ = ["local_or_planned_fingerprint", "planning_for", "scoped", "scoped_key"]

# The machine class the enclosing scope is planning *for*, when that is not this process.
#
# Ambient rather than threaded, and deliberately: the hazard this exists to remove is a
# **read and a write disagreeing**, and a parameter carried through twenty signatures is
# exactly how they come to disagree. The bandit's arm is written by `api.tuning.decisions`
# after a run and read by `kyber.rules.selection` while planning the next one; if those two
# resolve the class independently — say by asking the cluster twice, across an autoscale —
# the value is filed under a key nothing will ever read, and every learned quantity silently
# stops accruing. Inside one scope there is one answer by construction.
#
# `dist.executors.ray_runtime.scaling._TOPOLOGY` is the same pattern for the same reason.
_PLANNING_FOR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "batcher_planning_for_machine", default=""
)


@contextlib.contextmanager
def planning_for(hw_fingerprint: str) -> Iterator[None]:
    """Key machine-scoped learned state to `hw_fingerprint` for the enclosing scope.

    Wrap the span that both **plans** a run and **records** its outcome, so the two cannot
    key the same learned value differently. The conductor does this around a distributed
    run, naming the workers' machine class; single-node runs pass `""` and everything
    resolves to this process exactly as before.

    Args:
        hw_fingerprint: The class to key by, from `HardwareProfile.fingerprint`. `""` is a
            no-op, which is what a single-node run and an unprobeable fleet both pass.

    Yields:
        Nothing; the scope is the effect.
    """
    if not hw_fingerprint:
        yield
        return
    token = _PLANNING_FOR.set(hw_fingerprint)
    try:
        yield
    finally:
        _PLANNING_FOR.reset(token)


def local_or_planned_fingerprint() -> str:
    """The machine class in force here: the enclosing `planning_for` scope, else this process.

    The bare fingerprint that `scoped` and `scoped_key` embed in a name, for the one consumer
    that is not naming a namespace at all — `MetadataHub.op_stats_by_kind`, which buckets the
    feedback rows by fingerprint and needs to know which bucket to read. Exposed rather than
    re-derived there so all three resolve the class identically; a second resolution is exactly
    how a read and a write come to disagree, which is the hazard `planning_for` exists to
    remove.

    Returns:
        The fingerprint the enclosing scope names, or this process's own outside one.
    """
    return _PLANNING_FOR.get() or fingerprint()


# Separator between a namespace and its hardware fingerprint. `@` reads as "measured on" and
# appears in no existing namespace, so a scoped name can never collide with an unscoped one.
_SEPARATOR = "@"


def scoped(namespace: str, hw_fingerprint: str = "") -> str:
    """`namespace` qualified by a hardware fingerprint — by default this machine's.

    Use for any namespace whose stored values are measured in machine units — times, bytes,
    device capacities, or sizes chosen against them. Do not use for data statistics, which
    describe the data and transfer across machines unchanged.

    **Whose machine is not always this one.** A value written and read on the process that
    *executed* the work is correctly keyed by the local fingerprint, and that covers the UDF,
    autobatch and device-utilization loops. It does not cover the loops that run on the
    **driver** about work done on the **workers** — the join-strategy bandit, the broadcast and
    sort-merge crossovers, the build-side priors. Those are self-consistent under the local key
    (nothing is dropped, unlike the `op_stats` view this mirrors) but they name the wrong
    machine: a fleet that autoscales from one worker type to another files both under one key,
    and two drivers of different classes against identical workers fragment what should be one
    model. Such a caller passes the class it is planning *for*.

    Examples:
        .. doctest::

            >>> from batcher.metadata.hardware_scope import scoped
            >>> name = scoped("kyber.cost")
            >>> name.startswith("kyber.cost@") and len(name) == len("kyber.cost@") + 12
            True

    Args:
        namespace: The unscoped namespace name.
        hw_fingerprint: The machine class to key by, from `HardwareProfile.fingerprint`.
            `""` — the default, and what every caller measuring its own machine passes —
            falls back to the enclosing `planning_for` scope, then to this process's class.

    Returns:
        The namespace qualified with that machine class's fingerprint.
    """
    return f"{namespace}{_SEPARATOR}{hw_fingerprint or local_or_planned_fingerprint()}"


def scoped_key(key: str, hw_fingerprint: str = "") -> str:
    """`key` qualified by a hardware fingerprint — by default the one in force for this scope.

    The per-key counterpart of `scoped`, for a store whose namespace is already carrying
    another dimension and where splitting the namespace would fragment an index that other
    code walks whole. Prefer `scoped` where there is a choice: a scoped namespace keeps a
    machine's entries contiguous, which makes them cheap to load together and easy to drop
    when a machine class goes away.

    **Resolves the machine class exactly as `scoped` does**, through the enclosing
    `planning_for` scope before falling back to this process. It did not, and that was the one
    way the two spellings of the same idea could disagree: a value written with `scoped_key`
    inside a distributed run was filed under the *driver's* class while everything written with
    `scoped` in the same scope was filed under the *workers'*, so a read that used either
    spelling found nothing the other had stored. The whole reason `planning_for` is ambient
    rather than threaded is to make a read and a write agree by construction, and a second
    entry point that ignored it defeated that for its callers.

    Args:
        key: The unscoped key.
        hw_fingerprint: The machine class to key by, from `HardwareProfile.fingerprint`.
            `""` — the default — falls back to the enclosing `planning_for` scope, then to this
            process's class.

    Returns:
        The key qualified with that machine class's fingerprint.
    """
    return f"{key}{_SEPARATOR}{hw_fingerprint or local_or_planned_fingerprint()}"
