"""Carbonite data transfer: the standalone, locality-aware shuffle engine.

Groups the cross-worker movement layer Carbonite governs — the `ShuffleSession`
(credit-bounded, locality-aware), its Flight `FlightShuffleServer` endpoint and
`ShuffleTicket`, and the `TransferMode` selector. Re-exports only; the logic lives
in the sibling modules. This subpackage drives `batcher._native` (the bc-transport
data plane) — Carbonite as the transfer sublibrary, not glue inside the engine.
"""

from __future__ import annotations

from batcher.carbonite.transfer.locality import TransferMode, locality_ratio, select_mode
from batcher.carbonite.transfer.server import FlightShuffleServer, ShuffleTicket, fetch
from batcher.carbonite.transfer.session import ShuffleSession

__all__ = [
    "FlightShuffleServer",
    "ShuffleSession",
    "ShuffleTicket",
    "TransferMode",
    "fetch",
    "locality_ratio",
    "select_mode",
]
