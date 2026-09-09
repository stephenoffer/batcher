"""Attach to the GPU cluster with cuDF reachable and **no Ray environment build**.

A `pip` block in a Ray `runtime_env` costs a per-node virtualenv resolve whether or not the
packages are already there. Measured on this six-T4 fleet, `cudf-cu13==26.6.0` costs **168 s on
the first round** and nothing on the second -- all of it environment build, none of it compute.
Paid per node, per distinct runtime_env, on every driver process.

A cluster with a shared mount does not have to pay it at all. RAPIDS is staged **once** onto
`/mnt/cluster_storage` and reaches the workers as a `PYTHONPATH` entry, which Ray propagates as
an ordinary environment variable and never resolves. Measured against the same fleet: **9.1 s**
for the first worker's cold NFS import and ~0 after, against 168 s.

The mechanism itself is the engine's -- `distributed.gpu_rapids_path` and
`dist.gpu.cudf_probe.stage_rapids_env` -- so this module only points the configuration at the
mount and connects. It was written here first and moved down once it worked; what is left is
the two lines a benchmark needs and not a second copy of the staging.
"""

from __future__ import annotations

import functools
import os
import sys

from _ray_env import strip_broken_runtime_env_hook

__all__ = ["RAPIDS_DIR", "init_gpu_cluster", "stage_rapids"]

print = functools.partial(print, flush=True)

#: Where the shared RAPIDS tree lives. Overridable so a fleet with a different shared mount --
#: or none -- can point this elsewhere, and `BENCH_RAPIDS_DIR=` (empty) disables the mechanism
#: entirely and falls back to whatever the cluster image ships.
RAPIDS_DIR = os.environ.get("BENCH_RAPIDS_DIR", "/mnt/cluster_storage/rapids_env")


def stage_rapids(force: bool = False) -> str:
    """Copy the driver's RAPIDS install onto the shared mount, once. Returns the directory."""
    from batcher.dist.gpu.cudf_probe import stage_rapids_env

    return stage_rapids_env(RAPIDS_DIR, force=force) if RAPIDS_DIR else ""


def init_gpu_cluster(
    env_vars: dict[str, str] | None = None, pip: list[str] | None = None, **kwargs
) -> None:
    """Attach to the running cluster with Batcher shipped and cuDF on the workers' path.

    Args:
        env_vars: Extra environment variables for the workers, merged over the RAPIDS path.
        pip: Packages the workers must install, or `None` for none. Reserved for an engine
            the cluster image does not carry — Daft's Ray-runner actors cannot import `daft`
            here, so its arm needs it. The install is charged to *every* worker in the job, so
            a caller must only pass it for a sweep where that engine runs alone; adding it to a
            sweep with other engines taxes engines that do not need it and turns their numbers
            into a measurement of a pip install.
        **kwargs: Forwarded to `ray.init`.
    """
    strip_broken_runtime_env_hook(unconditional=True)
    import ray

    if ray.is_initialized():
        return
    staged = stage_rapids()
    import batcher
    from batcher.config import active_config, set_config

    if staged:
        # The engine's own knob, so the GPU tasks it submits carry the path too — this module
        # only decides *where* the mount is.
        import dataclasses

        distributed = dataclasses.replace(active_config().distributed, gpu_rapids_path=staged)
        # From the *active* config, not a fresh `Config()`: replacing a section on a default
        # instance reverts every other section, so this would discard whatever the caller had
        # already configured. Harmless while this runs first and nothing else is set; not
        # harmless as a pattern, and the identical line in `cluster_suite` was a real defect.
        set_config(active_config().replace(distributed=distributed))

    env = {"PYTHONPATH": staged} if staged else {}
    env.update(env_vars or {})
    runtime_env: dict = {"py_modules": [os.path.dirname(os.path.abspath(batcher.__file__))]}
    if env:
        runtime_env["env_vars"] = env
    if pip:
        runtime_env["pip"] = list(pip)
    ray.init(
        address="auto",
        runtime_env=runtime_env,
        logging_level="ERROR",
        log_to_driver=False,
        **kwargs,
    )
    if staged:
        # The workers' path already carries RAPIDS, so the pip block `gpu_backend_cudf` reaches
        # for would install nothing at a cost of ~168 s a node. Re-probe rather than assume.
        from batcher.dist.gpu.cudf_probe import reset_cudf_probe

        reset_cudf_probe()


if __name__ == "__main__":
    print(stage_rapids(force="--force" in sys.argv))
