"""Bounded waits for a Ray cluster that is not ready yet.

Two waits with the same shape, and the same failure mode if they are unbounded. Connecting
to a head that is still starting: the driver and the head come up concurrently in every
orchestrated environment, so the first attach routinely fails against a cluster seconds
from ready. And waiting for the autoscaler to deliver capacity a query asked for, so the
query fills the cluster it triggered a scale-up for rather than the pre-scale one.

Both poll or retry against a deadline, both give up and let the caller degrade rather than
hang, and both are capped by the job's own lease (`config.deadline`) — because a wait that
outlives the process helps nobody, and on a leased allocation these are the longest things
between the query being submitted and any work happening.

Split from `scaling` (which *measures* the live cluster) and `autoscale_request` (which
*asks* for capacity) because this is the third side of the same concern: waiting for what
was asked for to arrive.
"""

from __future__ import annotations

import contextlib
import contextvars
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

from batcher._internal.errors import BackendError
from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.config.deadline import remaining_budget

__all__ = [
    "await_autoscale",
    "bring_up_outliving_caller",
    "resolve_transport",
    "verify_disk_reach",
]

_T = TypeVar("_T")


def _cluster_topology() -> dict:
    """The live topology, resolved through the `scaling` module object.

    Deliberately not `from .scaling import cluster_topology`: the tests for these waits patch
    `scaling.cluster_topology` to script a capacity series, and a name bound at import time
    would not see that patch — the wait would silently poll the real cluster while the test
    believed it was driving one. Also breaks what would otherwise be an import cycle, since
    `scaling.clamp_workers` delegates here.
    """
    from batcher.dist.executors.ray_runtime import scaling

    return scaling.cluster_topology()


# --- Connecting to a head that is still coming up -----------------------------------

#: Bring-ups waiting for the lasting thread, as `(context, fn, future)`; None until first use.
_bringup_calls: queue.SimpleQueue | None = None
_bringup_thread: threading.Thread | None = None
_bringup_guard = threading.Lock()


def bring_up_outliving_caller(ray, lock: threading.Lock, fn: Callable[[], _T]) -> _T:
    """Run the bring-up `fn` under `lock`, on a thread that outlives the caller if Ray is down.

    A `ray.init` that starts a local cluster spawns the GCS and the raylet with Ray's kernel
    fate-sharing, `PR_SET_PDEATHSIG = SIGKILL`, and Linux sends that signal when the parent
    *thread* exits, not the process. So a pipeline run from a worker thread -- two
    concurrent `collect(distributed=True)` calls, a web handler, an executor pool -- that
    happened to bring Ray up took the cluster down when its thread finished: the driver lost
    the GCS seconds after the job registered, every other query on it stalled, and 60 s
    later Ray's watchdog terminated the whole driver with exit 1 and no traceback. Measured:
    `test_two_real_pipelines_run_at_once_and_both_are_correct` died that way in 6 of 8 runs
    and passes 8 of 8 through this function, and a bare `ray.init()` on a thread that then
    exits leaves a driver that cannot run a task.

    Only a cold bring-up starts processes, so with Ray already up `fn` runs here.
    """

    def guarded() -> _T:
        with lock:
            return fn()

    return guarded() if ray.is_initialized() else _on_a_lasting_thread(guarded)


def _on_a_lasting_thread(fn: Callable[[], _T]) -> _T:
    """Run `fn` on a thread that lives as long as the process, and return what it returns.

    The main thread already outlives everything Ray starts, so it runs `fn` directly, as
    does the lasting thread itself. Every other caller hands `fn` over with its
    `contextvars` (the active config and deadline are context-scoped) and waits.
    """
    global _bringup_calls, _bringup_thread
    current = threading.current_thread()
    if current is threading.main_thread() or current is _bringup_thread:
        return fn()
    with _bringup_guard:
        if _bringup_calls is None:
            _bringup_calls = queue.SimpleQueue()
            _bringup_thread = threading.Thread(
                target=_serve_bringups,
                args=(_bringup_calls,),
                name="batcher-ray-bringup",
                daemon=True,  # never joined at exit: it must not die before Ray's own atexit
            )
            _bringup_thread.start()
        calls = _bringup_calls
    done: Future = Future()
    calls.put((contextvars.copy_context(), fn, done))
    return done.result()


def _serve_bringups(calls: queue.SimpleQueue) -> None:
    """The lasting thread's loop: run each bring-up in its caller's context, forever."""
    while True:
        context, fn, done = calls.get()
        try:
            done.set_result(context.run(fn))
        except BaseException as exc:  # handed back to the waiting caller, who re-raises it
            done.set_exception(exc)


def _explicit_cluster_address() -> str | None:
    """The cluster address the *user* named, or `None` when one was merely detected.

    The distinction decides what an unreachable cluster means. A detected address (a
    KubeRay/Anyscale marker in the environment) is a hint, and falling back to a local Ray
    when it does not answer is a reasonable degradation. An address the user configured is
    an instruction, and quietly running single-node instead is a wrong answer rather than a
    degraded one — the job reports success having used one machine of the cluster they
    named, which is indistinguishable from working.
    """
    import os

    dc = active_config().distributed
    return dc.ray_address or os.environ.get("RAY_ADDRESS") or None


def _attach_with_retry(ray, **init_kwargs) -> bool:
    """Attach to a running cluster, retrying a not-yet-answering head. True when attached.

    The head and the driver come up concurrently in every orchestrated environment, so the
    first attach routinely fails against a cluster that is seconds from ready: a KubeRay
    driver pod is admitted before the head passes readiness, and a Slurm job's `ray start
    --head` races the step that runs the query. Retrying with exponential backoff turns
    that race into a pause instead of a silent single-node run.

    Bounded by `cluster_connect_timeout_s` and, under a lease, by the time actually left —
    waiting past the job's own deadline for a cluster to appear helps nobody. Returns False
    when the window is exhausted, leaving the caller to decide between raising (an explicit
    address) and falling back to local (a detected one).
    """
    dc = active_config().distributed
    budget = remaining_budget(max(0.0, dc.cluster_connect_timeout_s), reserve_s=dc.drain_lead_s)
    deadline = time.monotonic() + budget
    backoff = 0.5
    while True:
        try:
            ray.init(**init_kwargs)
            return True
        except ConnectionError:
            if ray.is_initialized():
                return True  # someone else won the race; we are attached either way
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(backoff, max(0.0, deadline - time.monotonic())))
            backoff = min(backoff * 2, 5.0)


def _connect_or_fall_back(ray, workers: int) -> None:
    """Retry the attach, then either raise (explicit address) or start a local Ray.

    Splitting the two cases is the point. Falling back to local for a *detected* address
    keeps a dev run inside a workspace whose cluster is down working, which is why the
    fallback exists. Doing the same for an address the user configured turns "connect to
    this cluster" into "run on this laptop" with no error anywhere — the job succeeds, on
    one machine, and nothing says the cluster was never reached.
    """
    from batcher.dist.executors.ray_runtime.lifecycle import _ray_init_kwargs

    address = _explicit_cluster_address()
    if _attach_with_retry(ray, **_ray_init_kwargs(workers)):
        return
    if address is not None:
        raise BackendError(
            f"could not connect to the Ray cluster at {address!r} within "
            f"{active_config().distributed.cluster_connect_timeout_s}s. "
            "Batcher will not silently run single-node against a cluster address you set: "
            "check the address and that the head is reachable, raise "
            "distributed.cluster_connect_timeout_s if the head is still starting, or pass "
            "distributed=False to run on this machine deliberately."
        )
    # Only a *detected* address gets here: the environment hinted at a cluster that turned
    # out not to be reachable. Degrade to a local single-node Ray rather than fail a job the
    # user never pointed at a specific cluster — but say so at WARNING. This is the stranded
    # case the Ray integration guide describes. The job succeeds on one machine while the
    # cluster it was billed for sits idle, and the only other trace of it is the INFO-level
    # attachment line.
    from batcher._internal.logging import get_logger

    get_logger("dist").warning(
        "a managed Ray cluster was detected from the environment but did not answer within "
        "%ss; running on a LOCAL single-node Ray instead. Set RAY_ADDRESS or "
        "distributed.ray_address to fail rather than fall back, or raise "
        "distributed.cluster_connect_timeout_s if the head is still starting.",
        active_config().distributed.cluster_connect_timeout_s,
    )
    ray.init(**_ray_init_kwargs(workers, force_local=True))


# --- Waiting for the autoscaler to deliver capacity ---------------------------------

# The capacity a wait *confirmed* the autoscaler won't exceed (set when a wait stalls below
# target): a later query asking for more skips the wait instead of re-discovering the same
# ceiling, so a fixed-at-max cluster pays the startup grace ONCE, not per cold query. A wait
# that grows the cluster lifts it (`_note_reached`), so real scale-up is never pinned stale.
_reachable_ceiling: float = float("inf")
#: The same learned bound for devices. Separate from the CPU one because the two are learned
#: from different evidence and one must not stand in for the other: a CPU-only fleet stalling
#: at 48 cores says nothing about its GPUs, and a GPU fleet at its maximum says nothing about
#: how many cores the autoscaler would add.
_reachable_gpu_ceiling: float = float("inf")
_ceiling_lock = threading.Lock()


def _note_ceiling(best_cpus: int) -> None:
    """Record that the autoscaler stalled at `best_cpus` — the cluster will not exceed it."""
    global _reachable_ceiling
    with _ceiling_lock:
        _reachable_ceiling = min(_reachable_ceiling, float(best_cpus))


def _note_gpu_ceiling(best_gpus: float) -> None:
    """Record that the autoscaler stalled at `best_gpus` devices.

    **A zero is never recorded, but a positive stall is.** A fleet whose GPU node has not
    registered yet reports 0 devices, and capping future
    requests at 0 would disable the accelerator for the life of the driver on exactly the
    cluster that was about to have one. A positive stall is different evidence entirely — the
    fleet showed its devices and stopped there — and refusing to learn from it would make the
    promise "a fixed cluster pays the startup grace once, not per query" false for every GPU
    stage: a six-device fleet asked for eight would pay the full 12 s grace on *every* query,
    having already proved on the first one that the eighth device is not coming.
    """
    global _reachable_gpu_ceiling
    if best_gpus <= 0:
        return
    with _ceiling_lock:
        _reachable_gpu_ceiling = min(_reachable_gpu_ceiling, float(best_gpus))


def _note_reached(cpus: int) -> None:
    """Lift a stale ceiling once capacity has climbed past it (the cluster grew/recovered)."""
    global _reachable_ceiling
    with _ceiling_lock:
        if cpus > _reachable_ceiling:
            _reachable_ceiling = float("inf")


def _note_gpus_reached(gpus: float) -> None:
    """Lift a stale device ceiling once the fleet has grown past it."""
    global _reachable_gpu_ceiling
    with _ceiling_lock:
        if gpus > _reachable_gpu_ceiling:
            _reachable_gpu_ceiling = float("inf")


def _reset_capacity_ceiling() -> None:
    """Forget the learned ceilings (tests; and any caller that wants a fresh probe)."""
    global _reachable_ceiling, _reachable_gpu_ceiling
    with _ceiling_lock:
        _reachable_ceiling = float("inf")
        _reachable_gpu_ceiling = float("inf")


def await_autoscale(target_cpus: int, target_gpus: float = 0.0) -> None:
    """Block (bounded, growth-detected) until the autoscaler grows the cluster toward
    `target_cpus` cores (and `target_gpus` GPUs).

    Called *before* the fan-out is sized to the cluster, so a query that triggered a scale-up
    (`request_autoscale`) fills the SCALED-UP cluster rather than the pre-scale one — without
    it the worker-per-node fill reads the current (small) topology and the query never uses
    the nodes it asked for. A no-op when the wait is disabled, Ray is down, the cluster already
    covers the target, or a previous wait learned it will not reach the target
    (`_reachable_ceiling`) — so a fixed cluster pays the startup grace once, not per query.
    Pure scheduling — the result is identical whether it waits or not.
    """
    if active_config().distributed.autoscale_wait_s <= 0 or target_cpus <= 0:
        return
    import ray

    if not ray.is_initialized():
        return
    topo = _cluster_topology()
    avail = int(topo["cpus"])
    # Read current capacity BEFORE the ceiling short-circuit, so a cluster grown since the
    # ceiling was learned re-probes: covering the target returns satisfied (lifting the stale
    # ceiling); merely exceeding it drops the bound and waits for the rest.
    if avail >= target_cpus and float(topo["gpus"]) >= target_gpus:
        if target_gpus <= 0:
            _note_reached(avail)
        return
    gpus = float(topo["gpus"])
    with _ceiling_lock:
        ceiling, gpu_ceiling = _reachable_ceiling, _reachable_gpu_ceiling
    if avail > ceiling:
        _note_reached(avail)  # capacity climbed past the old ceiling — it is stale
        ceiling = float("inf")
    if gpus > gpu_ceiling:
        _note_gpus_reached(gpus)
        gpu_ceiling = float("inf")
    # Short-circuit only when every target this wait is *still short of* has been proven
    # unreachable. Phrased against what is unsatisfied rather than against the targets, because
    # a GPU stage passes its device count as `target_cpus` too — so on a 48-core, 6-device fleet
    # asked for 8 devices the CPU target is already met and `target_cpus > ceiling` is false,
    # which would leave the whole condition false and the poll loop re-entered on every query.
    # That is the exact form of the bug being fixed, one level down.
    cpu_short = avail < target_cpus
    gpu_short = gpus < target_gpus
    cpu_hopeless = not cpu_short or target_cpus > ceiling
    gpu_hopeless = not gpu_short or target_gpus > gpu_ceiling
    if cpu_hopeless and gpu_hopeless:
        return  # a prior wait proved this is unreachable — don't re-discover it
    _await_autoscale(target_cpus, avail, target_gpus, gpus)


def _await_autoscale(
    target_cpus: int, avail: int, target_gpus: float = 0.0, avail_gpus: float = 0.0
) -> int:
    """Wait (bounded) for the cluster to grow to `target_cpus` (and `target_gpus`), returning
    observed CPUs.

    Polls the live CPU/GPU counts every `autoscale_poll_s` until both cover their targets or
    `autoscale_wait_s` elapses, then returns the CPU count. A GPU stage waits for the GPUs
    too, not just the cores (else it clamps to the 0 GPUs visible before the GPU node boots).
    A no-op (returns `avail`) when the wait is disabled or the cluster already fits; stops
    early via the grace windows below when capacity goes flat.
    """
    dc = active_config().distributed
    if dc.autoscale_wait_s <= 0 or (avail >= target_cpus and avail_gpus >= target_gpus):
        return avail
    import time

    from batcher.config.deadline import remaining_budget

    # Under a wall-clock lease, wait only as long as the job will still be alive to use the
    # nodes — minus the drain lead, so the fleet that does arrive has time to publish and
    # migrate its output. This is the longest wait in the scheduling path (180 s by default
    # on an autoscaling cluster), so it is the one that most often consumes a short
    # allocation entirely: a Slurm job with 90 seconds left would otherwise spend all of it
    # waiting for capacity that arrives after the kill, and compute nothing. Shrinking it
    # only makes the poll loop give up sooner and run on the capacity already present, which
    # is exactly what it does for a stalled autoscaler.
    budget = remaining_budget(dc.autoscale_wait_s, reserve_s=dc.drain_lead_s)
    if budget <= 0:
        return avail  # no time to wait for nodes; run on what is here now
    deadline = time.monotonic() + budget
    poll = max(0.1, dc.autoscale_poll_s)
    # Give up early once capacity has been flat for the grace window — the autoscaler is done
    # (fixed cluster) or cannot satisfy the request (spot unavailable), so the rest of the
    # budget would block on nodes that will not arrive; any gain resets the window. Two
    # regimes: until the FIRST growth a short `startup_grace` applies — an infeasible request
    # (a fixed cluster already at max, the common case where a large aggregate's fan-out
    # exceeds the node count) grows zero from the start, and the query already runs on current
    # capacity, so it must not eat the full 90 s stall for nodes that never come. Once any
    # growth appears the cluster is genuinely scaling and the longer `autoscale_stall_s`
    # governs. `startup_grace` sits above a couple of polls so nodes registering within a few
    # seconds still cross into the growing regime.
    stall_grace = max(dc.autoscale_stall_s, poll * 2)
    startup_grace = max(dc.autoscale_startup_grace_s, poll * 2)
    best = (avail, avail_gpus)
    saw_growth = False
    reached = False
    stalled = False
    last_growth = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        topo = _cluster_topology()
        avail = int(topo["cpus"])
        avail_gpus = float(topo["gpus"])
        if avail >= target_cpus and avail_gpus >= target_gpus:
            reached = True
            break
        if (avail, avail_gpus) > best:
            best = (avail, avail_gpus)
            saw_growth = True
            last_growth = time.monotonic()
        elif time.monotonic() - last_growth >= (stall_grace if saw_growth else startup_grace):
            stalled = True
            break  # nothing is coming (never started, or grew then stopped)
    # A CPU-only wait that stalled below its target has learned a ceiling; one that reached
    # (or grew past a stale ceiling) lifts it. The device half is learned separately below,
    # under the one rule that makes it safe: a **zero** is never recorded, because a 0-GPU
    # snapshot before a GPU node boots must not cap future GPU requests. Excluding GPU waits
    # entirely, which is what this did, made the docstring's promise ("a fixed cluster pays the
    # startup grace once, not per query") false for every GPU stage.
    #
    # A wait cut short by the *lease* has learned nothing about the cluster. It ran out of
    # time, which is a fact about this job, not about how far the autoscaler will go — and
    # recording it as a ceiling would tell every later query in the process that capacity it
    # never probed is unreachable. Only a genuine stall, or running the full requested
    # budget, is evidence about the autoscaler.
    truncated = budget < dc.autoscale_wait_s and not stalled
    if target_gpus <= 0:
        if reached or avail >= target_cpus:
            _note_reached(avail)
        elif not truncated:
            _note_ceiling(int(best[0]))
    # The device half, learned on the same evidence and with the same truncation rule. A zero
    # is never recorded (see `_note_gpu_ceiling`), which is what makes learning here safe on the
    # cluster the old GPU exclusion was protecting: one whose GPU node has not registered yet.
    if target_gpus > 0:
        if reached or avail_gpus >= target_gpus:
            _note_gpus_reached(avail_gpus)
        elif not truncated:
            _note_gpu_ceiling(best[1])
    return avail


def resolve_transport(transport: str, workers: int) -> str:
    """Resolve `transport == "auto"` to a concrete shuffle transport.

    Flight (Carbonite) on a genuine multi-node cluster — the disk shuffle writes to
    a driver-local `work_dir` that worker nodes can't reach, so disk is correct only
    on a single node or a configured shared filesystem. Explicit `"flight"`/`"disk"`
    pass through unchanged.
    """
    if transport == "auto":
        if active_config().distributed.shared_filesystem:
            transport = "disk"
        else:
            from .lifecycle import _ensure_ray

            _ensure_ray(workers)
            transport = "flight" if _cluster_topology()["nodes"] > 1 else "disk"
    if transport == "disk":
        verify_disk_reach()
    return transport


#: Seconds a node has to read the disk shuffle's visibility sentinel. A node that misses it
#: is logged rather than failed (see `shuffle_io.verify_shared_scratch`).
_REACH_PROBE_TIMEOUT_S = 30.0


def verify_disk_reach() -> None:
    """Prove the disk shuffle's scratch base is the same directory on every worker node.

    Only nodes other than the driver's are asked: the driver wrote the sentinel, so its own
    node reads it by construction, and a single-node cluster therefore pays nothing. A
    `ConfigError` from the probe propagates, since running the shuffle anyway fails later
    and less clearly; any other failure to *run* the probe is noted and the query proceeds.
    """
    from batcher._internal.errors import ConfigError
    from batcher.dist.shuffle_io import verify_shared_scratch

    try:
        import ray

        if not ray.is_initialized():
            return
        from .scaling import _alive_nodes

        here = ray.get_runtime_context().get_node_id()
        others = [n["NodeID"] for n in _alive_nodes() if n.get("NodeID") not in (None, here)]
        verify_shared_scratch(others, _read_on_nodes)
    except ConfigError:
        raise
    except Exception as exc:  # a probe that cannot run must not fail a query
        note_suppressed("dist", "verify the disk shuffle's shared scratch", exc)


def _read_on_nodes(node_ids: list[str], path: str) -> dict:
    """Run `read_visibility_token(path)` pinned to each node; `...` for a node that is silent."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from batcher.dist.shuffle_io import read_visibility_token

    probe = ray.remote(num_cpus=0)(read_visibility_token)
    refs = {
        probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote(path): node_id
        for node_id in node_ids
    }
    ready, pending = ray.wait(list(refs), num_returns=len(refs), timeout=_REACH_PROBE_TIMEOUT_S)
    out: dict = {}
    for ref in ready:
        try:
            out[refs[ref]] = ray.get(ref)
        except Exception as exc:
            note_suppressed("dist", "read the disk shuffle sentinel on a node", exc)
    for ref in pending:
        with contextlib.suppress(Exception):
            ray.cancel(ref, force=True)
    return out
