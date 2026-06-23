"""Stable identifiers used across plans and feedback."""

from __future__ import annotations

from typing import NewType

# A physical-operator identifier, unique within a plan. Used to correlate
# resource bounds (Kyber→Carbonite), allocations (Carbonite→Core), and execution
# feedback (Core→Kyber) for the same operator.
OpId = NewType("OpId", int)
