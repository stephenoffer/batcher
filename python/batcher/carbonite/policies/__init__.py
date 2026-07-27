"""Carbonite's resource policies — admission, flow control, scheduling, and sizing.

Most modules here implement one `carbonite.base` policy `Protocol`. They are the seam's
default occupants: a real deployment replaces one by constructing a different
implementation in its place, and the manager never learns which it got.

`morsel` and `cpu_budget` are the exceptions, and deliberately so: both are pure sizing
arithmetic over measured conditions with no policy state to swap, one for how much data a
batch should carry and one for how many cores to ask for. They live here rather than in the
manager because that is where the manager reaches for "how big should this be", and because a
sizing rule with no state is exactly the thing that should be readable on its own.
"""

from __future__ import annotations

from batcher.carbonite.policies.admission import BudgetingAdmission
from batcher.carbonite.policies.cpu_budget import (
    compute_bound_core_budget,
    effective_core_budget,
    oversubscription_note,
)
from batcher.carbonite.policies.flow_control import (
    AIMDFlowControl,
    StaticCreditFlowControl,
    credit_ceiling,
    learned_channel_morsel_bytes,
    load_shuffle_window,
    record_shuffle_window,
)
from batcher.carbonite.policies.morsel import morsel_target
from batcher.carbonite.policies.scheduling import DefaultSchedulingPolicy
from batcher.carbonite.policies.spill_advice import SpillAdvisor

__all__ = [
    "AIMDFlowControl",
    "BudgetingAdmission",
    "DefaultSchedulingPolicy",
    "SpillAdvisor",
    "StaticCreditFlowControl",
    "compute_bound_core_budget",
    "credit_ceiling",
    "effective_core_budget",
    "learned_channel_morsel_bytes",
    "load_shuffle_window",
    "morsel_target",
    "oversubscription_note",
    "record_shuffle_window",
]
