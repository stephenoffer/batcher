"""Whether this cluster's GPU workers already have cuDF, and what to do when they do not.

A `pip` block in a Ray `runtime_env` is not free even when every package in it is already
present: Ray builds a separate virtualenv for that environment and resolves the requirements
into it. Measured on a fleet whose image already ships RAPIDS, that is **26 seconds** on the
first GPU task of a session and a further ~120 ms on every task after, charged to a query that
needed neither — and it is charged per distinct runtime_env, so a fan-out pays it per node.
Measured on a six-T4 fleet whose image ships *no* RAPIDS, where the block genuinely installs
something, it is **168 s on the first round** and ~0 after: 6 shards whose task bodies summed
to 1.7 s took 168 s, and 0.3 s on the round immediately following.

So the pip block is added only when the cluster actually needs it, which means asking. The
asking is optimistic on purpose and self-correcting: an inconclusive probe answers "present",
and a task that then dies on the import records otherwise for every task after it. The reasons
are on `cluster_has_cudf` and `mark_cudf_missing`.

And when the cluster does need it, there are two ways to deliver it and they are not close in
cost. A **shared mount** — which any fleet with an NFS/EFS/Lustre volume already has — carries
one staged RAPIDS tree that reaches every worker as a `PYTHONPATH` entry Ray propagates as an
ordinary environment variable and never resolves. Measured against the same fleet: **9.1 s**
for the first worker's cold import off NFS and nothing after, against 168 s. `rapids_env_path`
is that mechanism; `cudf_pip_spec` is the fallback for a fleet with no shared mount.
"""

from __future__ import annotations

import os

from batcher._internal.logging import note_suppressed

__all__ = [
    "cluster_has_cudf",
    "cudf_pip_spec",
    "mark_cudf_missing",
    "rapids_env_path",
    "reset_cudf_probe",
    "stage_rapids_env",
]


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


#: Site-packages entries a staged RAPIDS tree needs. Deliberately **not** numpy or pyarrow: a
#: GPU worker already carries both, at versions the driver was checked against, and a staged
#: copy of either shadows a working install with one whose bundled `.libs` directory was not
#: copied beside it. Measured: staging numpy this way makes every worker fail on
#: `libscipy_openblas64_-*.so: cannot open shared object file` before it reaches cuDF at all.
_RAPIDS_STAGE = (
    "cuda",
    "cudf",
    "libcudf",
    "libkvikio",
    "librmm",
    "numba",
    "numba_cuda",
    "nvidia",
    "rmm",
    "pylibcudf",
    "rapids_logger",
    "cachetools",
    "llvmlite",
    # NVML, which is not a RAPIDS dependency and is easy to leave out of a worker image — and
    # without it `gpu_inventory()` is empty, so every device-memory decision this engine makes
    # silently sizes against nothing: the RMM pool is not built, the frame cache's budget is
    # zero, and an overflowing shard is subdivided blind. Measured on this fleet, all four.
    "pynvml.py",
    "nvtx",
    "packaging",
    "fsspec",
    "pandas",
    "pytz",
    "dateutil",
    "six.py",
    "typing_extensions.py",
)


def rapids_env_path() -> str:
    """The shared-mount RAPIDS directory the workers should import cuDF from, or `""`.

    `""` — the default — means the mechanism is off and the pip fallback applies. A directory
    that is configured but absent also answers `""`, because a `PYTHONPATH` pointing at nothing
    would silently give the workers no cuDF at all while suppressing the pip block that would
    have.

    Returns:
        The directory, or `""` when the mechanism is off or the directory is not there.
    """
    from batcher.config import active_config

    path = str(active_config().distributed.gpu_rapids_path or "")
    if not path:
        return ""
    return path if os.path.isdir(path) else ""


def cudf_pip_spec() -> list[str]:
    """The pip requirements that put this driver's cuDF on a worker, for a fleet with no mount.

    Derived from the driver's own installation rather than hardcoded. A pinned
    `cudf-cu13==26.6.0` is wrong on every fleet that is not this one: a CUDA-12 image needs
    `cudf-cu12`, and a driver on a different RAPIDS release ships partials the workers cannot
    unpickle. Reading the version off the driver makes the two sides the same build by
    construction, which is the property the whole fan-out depends on.

    **numpy is deliberately not pinned.** It used to be, to `1.26.4`, and that pin was actively
    harmful: RAPIDS 25.04 and later support numpy 2, so the pin dragged a numpy-2 driver's
    workers *back* to numpy 1 — and Ray pickles arrays by module path, so every array the task
    returned then failed to unpickle on the driver with
    `ModuleNotFoundError: No module named 'numpy._core'`. It made the exact failure it was
    written to prevent, in the direction nobody tested.

    Returns:
        The requirement list, or `[]` when the driver has no cuDF to describe — in which case
        there is nothing this fleet could be told to install that is known to match.
    """
    try:
        import cudf
    except ImportError as exc:
        note_suppressed("dist", "read the driver's cuDF version for the worker pip spec", exc)
        return []
    version = str(getattr(cudf, "__version__", "")).strip()
    suffix = _cuda_wheel_suffix()
    if not version or not suffix:
        return []
    # cuDF reports `26.06.00`; the wheel is `26.6.0`. Normalizing rather than passing the
    # reported string through: `cudf-cu13==26.06.00` resolves to nothing on PyPI, so an
    # unnormalized pin turns a slow environment build into a failed one.
    normalized = ".".join(str(int(part)) for part in version.split(".")[:3] if part.isdigit())
    return [f"cudf-{suffix}=={normalized or version}"]


def _cuda_wheel_suffix() -> str:
    """`"cu13"` / `"cu12"` — the wheel variant matching the driver's CUDA major, or `""`.

    Read from the installed distribution's own name rather than from a CUDA runtime probe: the
    question is which wheel *this driver* has, and a head node with no device (the common
    shape) answers a runtime probe with nothing at all.
    """
    try:
        from importlib.metadata import distributions

        for dist in distributions():
            name = (dist.metadata["Name"] or "").lower()
            if name.startswith("cudf-cu"):
                return name.split("-", 1)[1]
    except Exception as exc:
        note_suppressed("dist", "read the driver's cuDF wheel variant", exc)
    return ""


def stage_rapids_env(dest: str = "", *, force: bool = False) -> str:
    """Copy the driver's RAPIDS install to a shared mount so workers import it for free.

    Idempotent: an already-populated directory is left alone, so calling this before every
    query costs one `isdir`.

    Args:
        dest: Where to stage. Defaults to `distributed.gpu_rapids_path`.
        force: Re-copy entries that are already there.

    Returns:
        The staged directory, or `""` when there is nothing to stage or nowhere to put it.
    """
    import shutil

    from batcher.config import active_config

    target = dest or str(active_config().distributed.gpu_rapids_path or "")
    if not target:
        return ""
    root = os.path.abspath(target)
    if not force and os.path.isdir(os.path.join(root, "cudf")):
        return root
    try:
        import cudf
    except ImportError as exc:
        note_suppressed("dist", "stage RAPIDS for the workers: the driver has no cuDF", exc)
        return ""
    site = os.path.dirname(os.path.dirname(os.path.abspath(cudf.__file__)))
    os.makedirs(root, exist_ok=True)
    for name in _RAPIDS_STAGE:
        src = os.path.join(site, name)
        dst = os.path.join(root, name)
        if not os.path.exists(src) or (os.path.exists(dst) and not force):
            continue
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        except OSError as exc:
            note_suppressed("dist", f"stage {name} for the workers", exc)
    _graft_numba_cuda(root)
    return root


def _graft_numba_cuda(root: str) -> None:
    """Put `numba_cuda`'s `numba.cuda` where a bare `sys.path` entry will find it.

    `numba-cuda` ships its package as `numba_cuda/numba/cuda` and grafts it onto `numba.cuda`
    through a `.pth` file the interpreter runs at start-up. A `PYTHONPATH` entry runs no `.pth`,
    so without this the workers import the *stub* `numba.cuda` that numba 0.64 ships and cuDF
    dies on `No module named 'numba.cuda.core'` — several imports deep, in a task, with a
    traceback that names neither numba-cuda nor the staging.
    """
    import shutil

    overlay = os.path.join(root, "numba_cuda", "numba", "cuda")
    target = os.path.join(root, "numba", "cuda")
    if not os.path.isdir(overlay) or os.path.isdir(os.path.join(target, "core")):
        return
    try:
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(overlay, target)
    except OSError as exc:
        note_suppressed("dist", "graft numba.cuda into the staged RAPIDS tree", exc)
