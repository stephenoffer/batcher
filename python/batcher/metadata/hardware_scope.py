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

from collections.abc import Mapping
from typing import Any

from batcher._internal.hardware import fingerprint

__all__ = ["measured_here", "scoped", "scoped_key"]

# Separator between a namespace and its hardware fingerprint. `@` reads as "measured on" and
# appears in no existing namespace, so a scoped name can never collide with an unscoped one.
_SEPARATOR = "@"


def scoped(namespace: str) -> str:
    """`namespace` qualified by this machine's hardware fingerprint.

    Use for any namespace whose stored values are measured in machine units — times, bytes,
    device capacities, or sizes chosen against them. Do not use for data statistics, which
    describe the data and transfer across machines unchanged.

    Examples:
        .. doctest::

            >>> from batcher.metadata.hardware_scope import scoped
            >>> name = scoped("kyber.cost")
            >>> name.startswith("kyber.cost@") and len(name) == len("kyber.cost@") + 12
            True

    Args:
        namespace: The unscoped namespace name.

    Returns:
        The namespace qualified with this machine class's fingerprint.
    """
    return f"{namespace}{_SEPARATOR}{fingerprint()}"


def scoped_key(key: str) -> str:
    """`key` qualified by this machine's hardware fingerprint.

    The per-key counterpart of `scoped`, for a store whose namespace is already carrying
    another dimension and where splitting the namespace would fragment an index that other
    code walks whole. Prefer `scoped` where there is a choice: a scoped namespace keeps a
    machine's entries contiguous, which makes them cheap to load together and easy to drop
    when a machine class goes away.

    Args:
        key: The unscoped key.

    Returns:
        The key qualified with this machine class's fingerprint.
    """
    return f"{key}{_SEPARATOR}{fingerprint()}"


def measured_here(row: Mapping[str, Any]) -> bool:
    """Whether a stored feedback row was measured on this machine class.

    The predicate behind the `op_stats_by_kind` filter. A row carries the fingerprint of the
    machine that measured it, which for a distributed worker's row is the *worker's* rather
    than the driver's.

    A row with no fingerprint — one written before the field existed — is **not** ours. That
    is deliberate: "measured on an unknown machine" is not evidence about this one, and
    adopting it would reinstate exactly the blend this module removes, on the first run after
    an upgrade. Such rows age out of the store and the models re-converge within a few runs,
    because every query contributes feedback.

    Args:
        row: A stored feedback row.

    Returns:
        `True` when this machine class measured it.
    """
    return row.get("hw_fingerprint", "") == fingerprint()
