"""Making a collective fail instead of hang.

A multi-GPU collective — a tensor-parallel model, an all-reduce inside a UDF, any NCCL or RCCL
group — has a default failure mode that is uniquely bad for a scheduler: **it waits forever.**
When one rank dies, or one device faults, or one node is preempted mid-all-reduce, the
surviving ranks do not raise. They sit in the collective, holding their GPUs, until something
outside kills them. From the orchestrator's side there is no failure at all: the task is
running, the actor is alive, the GPUs are allocated, and no progress is being made. Every
recovery mechanism in this package — the recompute loop, the fault ledger, the retry budget —
is downstream of a failure being *reported*, and none of them ever runs.

This is the single highest-impact stability setting on a multi-GPU cluster, and it is one
environment variable. With asynchronous error handling on, a rank that loses a peer aborts with
an error instead of blocking, which turns an indefinite hang into an ordinary task failure that
the rest of this package already knows how to survive.

**This module is the stability half only.** The *performance* half — which NIC each device
should use, whether peer-to-peer is worth enabling, how close a NIC must be for GPUDirect —
lives in `dist.gpu.fabric.collective_env`, which derives it from the node's measured topology.
The two sets of variables are disjoint by construction (`STABILITY_VARS` below shares no name
with that module's `COLLECTIVE_VARS`), so neither can overwrite the other's decision.

Two rules, the same two that module holds to. **Nothing is invented**: only settings whose
effect is documented and unambiguous are set. And **the operator always wins**: a variable
already present in the environment is never replaced, because a deployment that pinned it has
a reason no probe can see.

Carbonite owns it — turning a hang into a reported failure is protection, not scheduling.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

__all__ = [
    "STABILITY_VARS",
    "collective_findings",
    "stability_env",
]

#: Every variable this module may set. Disjoint from `dist.gpu.fabric.collective_env`'s
#: `COLLECTIVE_VARS` by construction — that module decides which wires to use, this one decides
#: what happens when they break — so a test can assert the two never contend for a name.
STABILITY_VARS = (
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_DEBUG",
)

#: What this module sets, and why each one is safe to set unconditionally.
#:
#: The two async-error spellings are both written because PyTorch renamed the variable: the
#: `TORCH_`-prefixed one is what current releases read, the bare one is what older releases
#: read, and each ignores the other's. Writing both is how one setting covers a fleet whose
#: workers do not all run the same torch — which is the normal state of a cluster mid-upgrade,
#: and exactly when a hang is most likely.
#:
#: `NCCL_DEBUG=WARN` is the low-noise level: it prints nothing on a healthy run and prints the
#: reason on a failure. Unset — the default — a collective that aborts leaves no explanation
#: anywhere, so the operator sees a task that died with no message and no way to tell a bad
#: cable from a bad rank.
_STABILITY_DEFAULTS: dict[str, str] = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_ASYNC_ERROR_HANDLING": "1",
    "NCCL_DEBUG": "WARN",
}


def stability_env(process_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The collective settings that turn a hang into a reported failure.

    Args:
        process_env: The environment to check for existing settings, or `None` for this
            process's own. A worker inherits the driver's, so this is what an operator's
            deliberate choice looks like from here.

    Returns:
        Variable to value, containing only the ones not already set. Empty when the operator
        has settled all of them, which is the intended outcome on a tuned fleet.
    """
    ambient = os.environ if process_env is None else process_env
    return {k: v for k, v in _STABILITY_DEFAULTS.items() if not ambient.get(k)}


def collective_findings(
    process_env: Mapping[str, str] | None = None,
    *,
    rdma_ports: int = -1,
) -> tuple[str, ...]:
    """Collective settings that will cost this fleet a hang or a silent slow path.

    Read-only: this reports, it does not change anything, so it is safe to call from a health
    report. Each finding is a complete sentence naming the condition and its consequence,
    because the reader is an operator deciding whether to change a launch script.

    Args:
        process_env: The environment to inspect, or `None` for this process's own.
        rdma_ports: How many RDMA ports this node has, or `-1` to read them live. `0` means
            the node genuinely has none, in which case disabling InfiniBand costs nothing and
            is not reported.

    Returns:
        The findings, most serious first. Empty on a well-configured node *and* on one whose
        environment says nothing — an unset variable is the library's default, not a fault.
    """
    ambient = os.environ if process_env is None else process_env
    out: list[str] = []
    if not (
        ambient.get("TORCH_NCCL_ASYNC_ERROR_HANDLING") or ambient.get("NCCL_ASYNC_ERROR_HANDLING")
    ):
        # First because it is the one that makes every other failure unrecoverable: without
        # it, a lost peer is not an error, it is silence, and nothing downstream ever runs.
        out.append(
            "collectives: asynchronous error handling is off, so a lost rank or a faulted "
            "device will hang the surviving ranks indefinitely instead of failing the task"
        )
    if _is_on(ambient.get("NCCL_IB_DISABLE")) and _rdma_ports(rdma_ports) > 0:
        out.append(
            "collectives: NCCL_IB_DISABLE is set on a node with RDMA ports, so collectives "
            "will fall back to TCP and leave the fabric idle"
        )
    if _is_on(ambient.get("NCCL_SHM_DISABLE")):
        out.append(
            "collectives: NCCL_SHM_DISABLE is set, so same-node ranks will not use shared "
            "memory and intra-node collectives will run over the network stack"
        )
    if not ambient.get("NCCL_DEBUG"):
        out.append("collectives: NCCL_DEBUG is unset, so a collective that aborts will not say why")
    return tuple(out)


def _is_on(value: str | None) -> bool:
    """Whether an environment flag is set to something the library reads as enabled."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _rdma_ports(given: int) -> int:
    """This node's RDMA port count, read live when the caller did not supply one.

    Returns `0` rather than raising when the fabric cannot be read, which makes the
    InfiniBand finding conditional on *knowing* there is a fabric to lose. Reporting it on a
    node whose `/sys` is not mounted would tell an operator to re-enable a transport their
    node may not have.
    """
    if given >= 0:
        return given
    try:
        from batcher._internal.hardware.fabric import rdma_summary

        return int(rdma_summary().get("ports", 0))
    except Exception:
        return 0
