"""Whether this cluster's GPU workers already have cuDF, and what to do when they do not.

A `pip` block in a Ray `runtime_env` is not free even when every package in it is already
present: Ray builds a separate virtualenv for that environment and resolves the requirements
into it. Measured on a fleet whose image already ships RAPIDS, that is **26 seconds** on the
first GPU task of a session and a further ~120 ms on every task after, charged to a query that
needed neither — and it is charged per distinct runtime_env, so a fan-out pays it per node.

So the pip block is added only when the cluster actually needs it, which means asking. The
asking is optimistic on purpose and self-correcting: an inconclusive probe answers "present",
and a task that then dies on the import records otherwise for every task after it. The reasons
are on `cluster_has_cudf` and `mark_cudf_missing`.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed

__all__ = ["cluster_has_cudf", "mark_cudf_missing", "reset_cudf_probe"]


#: Whether this cluster's GPU workers already import cuDF, or `None` before anything asked.
_cluster_cudf: bool | None = None


def cluster_has_cudf() -> bool:
    """Whether the GPU workers already have cuDF, so the pip spec would install nothing.

    A `pip` block in a `runtime_env` is not free even when every package in it is already
    present: Ray builds a separate virtualenv for that environment and resolves the
    requirements into it. Measured on a fleet whose image already ships RAPIDS, that is **26
    seconds** on the first GPU task of a session and a further ~120 ms on every task after,
    charged to a query that needed neither — and it is charged again for each distinct
    runtime_env, so a fan-out pays it per node.

    Asked once per driver process and cached, because the answer is a property of the image
    and cannot change under a running cluster. The probe takes a hundredth of a device rather
    than a whole one, so it neither blocks a real shard nor lands on a CPU-only node.

    An **inconclusive** probe answers "present", which is the opposite of what a best-effort
    check usually does and is deliberate. The probe is inconclusive exactly when the cluster is
    too busy to run a trivial task within a few seconds, and that is also when a 26-second
    environment build hurts most. Guessing wrong is cheap and self-correcting: the first real
    task fails with a cuDF import error, `mark_cudf_missing` records it, and every task after
    carries the pip block. Guessing the other way costs the build on every busy cluster,
    forever, and nothing ever discovers it was unnecessary.

    Returns:
        True when a GPU worker imported cuDF, and when the probe could not reach a conclusion.
        False only on a positive reading that cuDF is absent — from the probe itself, or from
        `mark_cudf_missing` after a real task failed on the import.
    """
    global _cluster_cudf
    if _cluster_cudf is None:
        _cluster_cudf = _probe_cluster_cudf()
    return _cluster_cudf


#: How long the probe waits for a trivial task before giving up and assuming cuDF is present.
#: Short on purpose: past a few seconds the cluster is busy rather than cuDF-less, and the
#: answer that follows from *that* is the one below.
_PROBE_TIMEOUT_S = 8.0


def _probe_cluster_cudf() -> bool:
    """Run one tiny task on a GPU node and report whether it could import cuDF."""
    from batcher.dist.executors.ray_runtime.scheduling import worker_runtime_env

    try:
        import ray

        if not ray.is_initialized():
            return True
        env = worker_runtime_env() or None
        # `num_cpus=0`, matching the GPU tasks this is probing on behalf of. Ray hands an
        # unspecified task one core, and the cluster this probe most needs to answer quickly is
        # the one whose cores are all inside somebody's placement group — so the probe pended,
        # spent its whole timeout, and reached the "inconclusive means present" branch by way
        # of an eight-second stall on every session. Asking for a core the probe does not use
        # can only delay it.
        options = {"num_gpus": 0.01, "num_cpus": 0, "max_retries": 0}
        if env:
            options["runtime_env"] = env
        ref = ray.remote(**options)(_import_cudf).remote()
        try:
            return bool(ray.get(ref, timeout=_PROBE_TIMEOUT_S))
        except Exception as exc:
            # A busy cluster, not a cuDF-less one. Cancel so the probe does not occupy a
            # device share behind the real work it was asked about.
            ray.cancel(ref, force=True)
            note_suppressed("dist", "probe the cluster for cuDF", exc)
            return True
    except Exception as exc:
        note_suppressed("dist", "probe the cluster for cuDF", exc)
        return True


def _import_cudf() -> bool:
    """On a worker: whether cuDF is importable here. The body of the cluster probe."""
    try:
        import cudf  # noqa: F401
    except Exception:
        return False
    return True


#: Substrings identifying a task that died because cuDF was not installed on its worker, as
#: opposed to any other import error. Matched on text because the exception reaches the driver
#: through Ray, which re-raises it as its own wrapper type and leaves only the message intact.
_CUDF_ABSENT_MARKERS = ("no module named 'cudf'", 'no module named "cudf"')


def mark_cudf_missing(exc: BaseException) -> bool:
    """Record that a worker had no cuDF, so later tasks carry the pip block.

    The correction half of the optimistic probe above: an inconclusive probe assumes cuDF is
    present, and this is what makes that assumption self-correcting rather than permanent.

    Args:
        exc: The error a GPU task raised.

    Returns:
        True when the error was a missing cuDF and the cache has been updated, so a caller may
        retry. False for every other failure, which this must not touch — a device out of
        memory is not evidence about what is installed.
    """
    global _cluster_cudf
    text = f"{type(exc).__name__} {exc}".lower()
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        text += f" {type(cause).__name__} {cause}".lower()
    if not any(marker in text for marker in _CUDF_ABSENT_MARKERS):
        return False
    _cluster_cudf = False
    return True


def reset_cudf_probe() -> None:
    """Forget the cached cuDF probe, so the next task asks the cluster again.

    For tests, and for a driver that reconnects to a different cluster in one process.
    """
    global _cluster_cudf
    _cluster_cudf = None
