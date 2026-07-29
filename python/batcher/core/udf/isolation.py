"""What a UDF child process is allowed to see and consume.

A CPU-bound `map_batches` UDF runs in a `forkserver` child of the engine process. Children
inherit the parent's environment wholesale, and the engine's environment is where
credentials live: `bc-secrets` resolves `env:NAME` references precisely so a plan never has
to carry a secret inline, and `BATCHER_SECRET_COMMAND` names the operator's helper for
fetching arbitrary secrets from Vault, AWS, or GCP. So a two-line UDF could read
`AWS_SECRET_ACCESS_KEY` out of a *worker* process, or simply ask the helper for anything it
liked. That is what this module closes.

# Exactly which path this covers, and which it cannot

**It covers the process path only, and that limit is structural rather than an omission.**
`map_batches` runs a UDF on threads or on processes depending on how expensive the callable
measures (`core/udf/strategy.py`); only the process path forks a child, and only a child has
an environment separate from the engine's to scrub. A UDF on the thread path runs *inside
the engine process* and can read `os.environ` — along with anything else the engine holds —
because it **is** the engine process. No in-process mechanism changes that, and pretending
otherwise would be worse than the gap.

So the accurate claim is: a UDF can no longer harvest credentials *by being handed them in
a worker*. It is not: a UDF cannot obtain credentials.

# What this is, and what it is not

This is **environment isolation and resource limits**: defense in depth, labelled as such
deliberately. It is *not* a sandbox. A UDF is arbitrary Python in a process that can
`ctypes` its way to any syscall.

Batcher does not ship a syscall sandbox on purpose. An import allowlist is defeated by
`getattr(builtins, "__imp" + "ort__")` or a pickle `__reduce__`, so shipping one would
create a false claim rather than a defence. Real isolation needs seccomp-bpf or user
namespaces, which need privileges Batcher does not have and which break CUDA and torch —
i.e. they break the ML workloads that are the reason UDFs exist here. **Untrusted UDFs
belong in a container, not behind a config flag.** Run one process per trust domain.

What this *does* buy is real: a worker no longer carries the cluster's credentials, a
runaway allocation fails its own child instead of drawing the OOM killer onto the box, and
a wedged UDF raises instead of hanging the query forever.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

__all__ = [
    "DEFAULT_ENV_ALLOWLIST",
    "ResourceLimits",
    "child_initializer",
    "resolve_isolation",
    "shard_directory",
]

#: Environment variables a UDF child keeps. Everything else is dropped.
#:
#: An allowlist, not a denylist, and that direction is the whole point: a denylist has to
#: enumerate every secret-bearing name that exists now and every one any SDK invents later
#: (`AWS_*`, `GOOGLE_*`, `AZURE_*`, `HF_TOKEN`, `OPENAI_API_KEY`, a bespoke `ACME_DB_PASS`),
#: and it is wrong the first time it misses one. This list is instead the variables a UDF
#: legitimately *needs* — where to find binaries and temp space, what locale to use, and
#: the thread/device pinning that keeps a worker from oversubscribing the box.
DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "VIRTUAL_ENV",
    # Accelerator and thread pinning. Dropping these would not be a security win — it
    # would let every worker spin a full BLAS pool and oversubscribe the cores, which is
    # the regression `forkserver` was chosen to avoid in the first place.
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)

#: Prefixes scrubbed even if something adds them to the allowlist. `BATCHER_SECRET_COMMAND`
#: is the sharp one: it names a program that hands out secrets on request, so a child that
#: keeps it does not need to have inherited any particular credential to obtain one.
_ALWAYS_DROP_PREFIXES: tuple[str, ...] = ("BATCHER_",)


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Per-child resource ceilings, in the units `resource.setrlimit` wants.

    Zero means "leave the inherited limit alone" for every field, so the default
    configuration changes nothing about how a UDF runs.
    """

    #: Address-space ceiling. The honest answer to "can a UDF respect a memory quota":
    #: a Python-level check cannot stop `numpy.zeros(1 << 40)`, and `RLIMIT_AS` can —
    #: the allocation raises `MemoryError` in the child instead of the kernel's OOM
    #: killer choosing a victim, which on a shared box is often not the guilty process.
    memory_bytes: int = 0
    #: CPU-seconds before the kernel sends `SIGXCPU`. Catches an infinite loop that a
    #: wall-clock timeout would also catch, but without needing the driver to be watching.
    cpu_seconds: int = 0
    #: Open-file ceiling, against a UDF that leaks descriptors across many batches.
    nofile: int = 0
    #: Subprocess/thread ceiling, against a fork bomb — accidental or otherwise.
    nproc: int = 0
    #: Disable core dumps. A core file from a UDF child contains the whole address space,
    #: including anything the parent had in memory, written to disk under whatever umask
    #: the box has. Always applied, never opt-in.
    no_core: bool = True


def resolve_isolation(config) -> tuple[str, tuple[str, ...], ResourceLimits]:
    """Read the isolation mode, environment allowlist, and limits from `config.execution`.

    Args:
        config: The active `Config`.

    Returns:
        `(mode, allowed_env, limits)`, where mode is ``none``/``env``/``strict``.
    """
    execution = config.execution
    mode = getattr(execution, "udf_isolation", "env")
    extra = tuple(getattr(execution, "udf_env_allowlist", ()) or ())
    allowed = (*DEFAULT_ENV_ALLOWLIST, *extra)
    limits = ResourceLimits(
        memory_bytes=int(getattr(execution, "udf_memory_limit_bytes", 0) or 0),
        cpu_seconds=int(getattr(execution, "udf_cpu_limit_seconds", 0) or 0),
    )
    return mode, allowed, limits


def _apply_limits(limits: ResourceLimits) -> None:
    """Apply `limits` to this process. Best-effort, per limit."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return
    pairs = (
        (getattr(resource, "RLIMIT_AS", None), limits.memory_bytes),
        (getattr(resource, "RLIMIT_CPU", None), limits.cpu_seconds),
        (getattr(resource, "RLIMIT_NOFILE", None), limits.nofile),
        (getattr(resource, "RLIMIT_NPROC", None), limits.nproc),
    )
    for which, value in pairs:
        if which is None or value <= 0:
            continue
        try:
            _soft, hard = resource.getrlimit(which)
            # Never raise the hard limit: an unprivileged process cannot, and asking would
            # raise where lowering succeeds.
            ceiling = value if hard in (resource.RLIM_INFINITY, -1) else min(value, hard)
            resource.setrlimit(which, (ceiling, hard))
        except (ValueError, OSError):
            # One unsupported limit must not stop the others — dropping the *environment*
            # is the load-bearing half of this, and it has already happened by here.
            continue
    if limits.no_core:
        with _suppress_rlimit_errors():
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _suppress_rlimit_errors():
    """`contextlib.suppress` for the rlimit calls, imported lazily in the child."""
    import contextlib

    return contextlib.suppress(ValueError, OSError, AttributeError)


def child_initializer(allowed_env: tuple[str, ...], limits: ResourceLimits) -> None:
    """Run in each pool child before it takes any work: scrub, then constrain.

    Ordering matters. The environment is rebuilt **first**, so that even if applying a
    resource limit fails on some platform the credentials are already gone — the security
    property does not depend on the resource-limit code path succeeding.

    Args:
        allowed_env: Variable names to keep; every other is removed.
        limits: Ceilings to apply to this child.
    """
    keep = {name: os.environ[name] for name in allowed_env if name in os.environ}
    keep = {
        name: value
        for name, value in keep.items()
        if not any(name.startswith(p) for p in _ALWAYS_DROP_PREFIXES)
    }
    os.environ.clear()
    os.environ.update(keep)

    # Anything this child writes — a spill file, a model cache, a temp file inside the UDF
    # — is private to the running user. The parent sets its own umask for the shard files
    # it writes; this covers everything the UDF itself creates.
    if hasattr(os, "umask"):
        os.umask(0o077)

    _apply_limits(limits)


def shard_directory() -> str:
    """A private, per-process directory for the memory-mapped input shards.

    The shards are the *data itself*, written to `/dev/shm` — a world-writable directory —
    as `bcudf_<pid>_<n>_<g>.arrow` under the default umask, i.e. mode 0644. Any local user
    could read a query's batches while it ran, and on a multi-tenant box that is a
    cross-tenant data leak with no query involved at all.

    Putting them in a 0700 directory means the permission holds even for the window between
    `open` and any `chmod`, and it also gives the cleanup path a single thing to remove.

    Returns:
        Path to a directory that exists and is private to this user.
    """
    import tempfile

    root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    path = os.path.join(root, f"bcudf_{os.getpid()}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    if sys.platform != "win32":
        # `makedirs` honours the umask, so an existing or umask-widened directory is
        # tightened explicitly rather than assumed.
        with _suppress_rlimit_errors():
            os.chmod(path, 0o700)
    return path
