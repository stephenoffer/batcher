"""Grow the cluster to the devices a plan wants, check one is free, and fan the work out.

Every entry point here has the same shape — ask the autoscaler, confirm admission, run — and
the same contract: a `None` means the fan-out does not apply and the caller has a slower path
behind it. What none of them may do is *silently* mean something else, which is what happened
when a moved autoscale helper made the whole file raise `ImportError` on its first line and the
handler read that as a decline. `note_gpu_failure` is the fix for the class, and
`tests/unit/test_gpu_backend_wiring.py` is the guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.api.terminal.gpu_backend.failure import note_gpu_failure
from batcher.api.terminal.routing import _ray_already_live

if TYPE_CHECKING:
    from batcher.io.source import Source


def _with_gpu_capacity(gpu_count: int, decision, run):
    """Run a fan-out with the cluster grown to the devices the plan wants, then released.

    Asks the autoscaler for `decision.desired_gpus` — enough to hold the working set in one
    wave — rather than for the devices the cluster already has. Asking for what is already
    there pins the floor against reclamation and can never grow the cluster, so a query that
    could use thirty-two devices ran on the four it happened to find. It then waits (bounded,
    and a no-op on a fixed cluster) before `run` sizes its shards, since sizing against the
    pre-scale topology is how a query asks for capacity and then declines to use it.

    `run` receives the device count the wait actually produced. Any failure inside is a
    `None` — every fan-out has a slower path behind it, and none of them change the answer.

    The three helpers come from the `ray_runtime` package façade, which is where they are
    exported. They used to be imported from `ray_runtime.scaling`, which defines none of them
    — so this function raised `ImportError` on its first line, every time, and the caller's
    blanket handler read that as "the fan-out does not apply". The result was that **no GPU
    query ever used more than one device**: every fan-out — aggregate, join and union alike —
    fell through to the single-dispatch path or to the CPU engine, correctly, silently, and at
    a fraction of the throughput the cluster had. This is the failure mode `.claude/rules`
    calls out by name — a moved symbol reading as a dispatch failure — and it is why
    `test_gpu_backend_imports_resolve` now asserts the import instead of trusting it.
    """
    from batcher.dist.executors.ray_runtime import (
        await_autoscale,
        release_autoscale,
        request_autoscale,
    )
    from batcher.dist.gpu.dispatch import await_gpu_admission

    wanted = max(gpu_count, int(decision.desired_gpus))
    request_autoscale(gpu_count, target_gpus=float(wanted))
    try:
        # The CPU figure is what makes the *wait* meaningful — a zero CPU target reads as "this
        # query wants nothing" and turns it into a no-op — while the GPU target is the one the
        # fleet is actually grown against. The shard tasks themselves request no core.
        await_autoscale(wanted, target_gpus=float(wanted))
        # ...and then whether any of that capacity is actually free. The wait above asks
        # whether the fleet is big *enough*, which it can be while every device of it is held
        # by another stage — and a GPU task that cannot be placed does not fail, it pends, and
        # `ray.get` on a pending task waits for as long as the query is willing to. Declining
        # here is what turns that into a slower answer instead of no answer.
        if not await_gpu_admission():
            return None
        return run(_cluster_gpu_count())
    except Exception as exc:
        note_gpu_failure("fan a GPU stage out across devices", exc)
        return None
    finally:
        release_autoscale()


def _try_sharded_aggregate(source: Source, ops: list[dict], gpu_count: int, decision):
    """Fan a chain with a mergeable reducer out across the cluster's GPUs, or `None`.

    `None` means the fan-out does not apply — the chain has no shardable split, the source is
    not splittable, or the cluster is unreadable — and the caller then uses the single-device
    dispatch. Every failure mode is a slower path, never a different answer: the fan-out is the
    mergeable decomposition of the same chain."""
    from batcher.dist.gpu import sharded_gpu_aggregate

    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_aggregate(
            source, ops, gpu_count=live, sharded=decision.distributed
        ),
    )


def _try_sharded_join(left, right, lops, rops, join_ir, ops, gpu_count: int, decision):
    """Fan a broadcast-safe join out across the cluster's GPUs, or `None`."""
    from batcher.dist.gpu import sharded_gpu_join

    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_join(
            left,
            right,
            lops,
            rops,
            join_ir,
            ops,
            gpu_count=live,
            sharded=decision.distributed,
            # Kyber's verdict, not a re-derivation. The two backends asking this question
            # separately is how they come to disagree, and the wrong answer is a device
            # out-of-memory on every worker at once. Omitting it entirely — which is what this
            # call did — raised `TypeError` inside a handler that reads every failure as "the
            # fan-out does not apply", so the multi-device join silently never ran.
            broadcast=decision.broadcast_join,
        ),
    )


def _try_sharded_union(usources, input_ops, distinct, ops, gpu_count: int, decision):
    """Fan a `UNION ALL` out across the cluster's GPUs, or `None`.

    A deduplicating union declines inside the fan-out rather than here, so the rule about why
    lives beside the algebra that cannot honour it."""
    from batcher.dist.gpu import sharded_gpu_union

    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_union(
            usources,
            input_ops,
            distinct,
            ops,
            gpu_count=live,
            sharded=decision.distributed,
        ),
    )


def _try_tree(spec: dict, sources: list[Source], gpu_count: int, decision):
    """Fan a plan tree out across the cluster's GPUs, or `None` when it does not apply."""
    from batcher.dist.gpu.tree import sharded_gpu_tree

    budget = _broadcast_budget_bytes()
    return _with_gpu_capacity(
        gpu_count,
        decision,
        lambda live: sharded_gpu_tree(
            spec,
            sources,
            gpu_count=live,
            sharded=decision.distributed,
            replicate_budget_bytes=budget,
        ),
    )


def _broadcast_budget_bytes() -> float:
    """How many bytes of replicated leaves one device may hold beside its own shard.

    Sized from the cluster's *binding* (smallest) device where the topology can report one,
    because a fan-out is only as safe as the device it runs worst on. A mixed fleet planned
    against its largest device is a fleet where the small ones all fail together.

    A fraction of the device's **usable** memory, not of its nameplate capacity. The replicated
    leaves sit beside this device's own shard, the joins built above them, and the CUDA context
    — so charging them a share of memory that includes the part nothing may allocate spends a
    budget twice. Both suppliers report a capacity, and `device_headroom()` is what turns one
    into the other, once.
    """
    from batcher._internal.device_share import device_headroom
    from batcher.config import active_config
    from batcher.dist.executors.ray_runtime.accelerators import cluster_gpu_memory_gb

    dc = active_config().distributed
    gpu_gb = cluster_gpu_memory_gb() or dc.resolved_gpu_memory_gb()
    fraction = min(0.9, max(0.0, float(dc.gpu_tree_broadcast_fraction)))
    return float(gpu_gb) * 1e9 * (1.0 - device_headroom()) * fraction


def _cluster_gpu_count() -> int:
    """Live GPU count across the cluster (or 1 for a GPU-equipped local process), else 0.

    The count — not just presence — so Kyber's policy can size the cluster's aggregate GPU
    memory and pick single-device vs sharded execution."""
    # Gated exactly as the `distributed="auto"` probe is, and for the same reason: this runs
    # on every terminal op, and `import ray` costs ~0.44 s the first time to answer a question
    # `sys.modules` already settles — a cluster cannot be initialized in a process that has
    # not imported Ray. See `routing._ray_already_live` for the full argument.
    if _ray_already_live():
        try:
            import ray

            if ray.is_initialized():
                from batcher.dist.executors.ray_runtime import cluster_topology

                return int(cluster_topology().get("gpus", 0))
        except Exception as exc:
            note_suppressed("api", "read cluster GPU topology", exc)
    from batcher.core.gpu_transform import gpu_available

    return 1 if gpu_available() else 0
