"""Carbonite data transfer: the standalone, locality-aware shuffle engine.

Groups the cross-worker movement layer Carbonite governs — the `ShuffleSession`
(credit-bounded, locality-aware), its Flight `FlightShuffleServer` endpoint and
`ShuffleTicket`, and the `TransferMode` selector. Re-exports only; the logic lives
in the sibling modules. This subpackage drives `batcher._native` (the bc-transport
data plane) — Carbonite as the transfer sublibrary, not glue inside the engine.

Two of the siblings move bytes that never leave a node: `device_exchange` schedules a
redistribution between the devices of one host so the pairs do not serialize behind each
other, and `staging` sizes the host-side ring every transfer that does cross the host link
runs through. Both are pure planners over a described topology.
"""

from __future__ import annotations

from batcher.carbonite.transfer.device_exchange import (
    ExchangePlan,
    ExchangeStep,
    all_reduce_seconds,
    exchange_seconds,
    pairwise_rounds,
    plan_exchange,
    ring_bandwidth_gbps,
    ring_order,
    worth_device_exchange,
)
from batcher.carbonite.transfer.lifecycle import local_session
from batcher.carbonite.transfer.locality import (
    TransferMode,
    locality_ratio,
    select_device_mode,
    select_mode,
)
from batcher.carbonite.transfer.peers import (
    PeerTransfer,
    peer_transfers,
    reset_peer_transfers,
    straggler_peer,
)
from batcher.carbonite.transfer.server import FlightShuffleServer, ShuffleTicket, fetch
from batcher.carbonite.transfer.session import ShuffleSession
from batcher.carbonite.transfer.staging import (
    StagingPlan,
    chunk_bytes_for_link,
    effective_gbps,
    pipeline_depth,
    plan_staging,
    staging_seconds,
    worth_pinning,
)

__all__ = [
    "ExchangePlan",
    "ExchangeStep",
    "FlightShuffleServer",
    "PeerTransfer",
    "ShuffleSession",
    "ShuffleTicket",
    "StagingPlan",
    "TransferMode",
    "all_reduce_seconds",
    "chunk_bytes_for_link",
    "effective_gbps",
    "exchange_seconds",
    "fetch",
    "local_session",
    "locality_ratio",
    "pairwise_rounds",
    "peer_transfers",
    "pipeline_depth",
    "plan_exchange",
    "plan_staging",
    "reset_peer_transfers",
    "ring_bandwidth_gbps",
    "ring_order",
    "select_device_mode",
    "select_mode",
    "staging_seconds",
    "straggler_peer",
    "worth_device_exchange",
    "worth_pinning",
]
