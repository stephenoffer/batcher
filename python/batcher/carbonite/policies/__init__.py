"""Carbonite's resource policies — admission, flow control, and scheduling.

Each module here implements one `carbonite.base` policy `Protocol`. They are the seam's
default occupants: a real deployment replaces one by constructing a different
implementation in its place, and the manager never learns which it got.
"""

from __future__ import annotations

from batcher.carbonite.policies.admission import BudgetingAdmission
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
    "credit_ceiling",
    "learned_channel_morsel_bytes",
    "load_shuffle_window",
    "morsel_target",
    "record_shuffle_window",
]
