"""Ray Data adapter — the distributed-streaming comparator.

Ray Data has no SQL surface, so it sits out the standard SQL suites and carries the
operator-mix (filter/groupby/aggregate/sort) on the native ``ray.data.Dataset``
handle. A controlled local Ray is initialized on first use (no dashboard, quiet) so
runs stay reproducible.
"""

from __future__ import annotations

import importlib.util
import logging

import pyarrow as pa
import pyarrow.fs as pafs

from .base import Engine


def _neutralize_broken_runtime_env_hook() -> None:
    """Drop a ``RAY_RUNTIME_ENV_HOOK``/``RAY_RUNTIME_ENV_PLUGINS`` whose module is missing.

    A managed host env (e.g. a ``cgroup_runtime_plugin``) may export a runtime-env
    hook Ray imports during ``ray.init``; outside that runtime the module is absent
    and init crashes. A hook pointing at an unimportable module is broken regardless,
    so removing it is strictly safer — and a no-op where the module is present.
    """
    import importlib.util
    import os

    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        value = os.environ.get(var)
        if not value:
            continue
        head = value.lstrip("[{\"' ").split(".")[0].split("[")[0]
        if head and importlib.util.find_spec(head) is None:
            os.environ.pop(var, None)


def _worker_runtime_env() -> dict:
    """Drop an unresolvable local editable (``batcher-engine``) from the inherited pip env.

    Some managed platforms inject the workspace's ``requirements.txt`` as the default runtime-env
    ``pip`` block, inherited by every task/actor. When that list contains the local
    editable ``batcher-engine`` (not on any index), the per-worker pip build hard-fails
    and Ray Data cannot launch a single task. Ray Data's own dependencies already live
    in the cluster's base env, so nulling ``pip`` for the comparison is both correct and
    the representative setup (workers run the stock Ray Data image). Mirrors
    ``_neutralize_broken_runtime_env_hook`` — a broken inherited env is a no-op to strip.
    """
    return {"pip": None}


def _ensure_ray() -> None:
    import os

    import ray

    if not ray.is_initialized():
        _neutralize_broken_runtime_env_hook()
        # Attach to the existing cluster (a running Ray head). Ray Data is a
        # distributed engine; benchmarking it on the real multi-node cluster it is built
        # for is the representative comparison. ``BENCH_RAY_ADDRESS`` overrides the
        # target; the default "auto" discovers the local head. We do NOT spin up an
        # isolated local cluster — the data plane comparison must run on Ray's home turf.
        address = os.environ.get("BENCH_RAY_ADDRESS", "auto")
        ray.init(
            address=address,
            ignore_reinit_error=True,
            configure_logging=False,
            log_to_driver=False,
            runtime_env=_worker_runtime_env(),
        )
        # Silence Ray Data's per-dataset progress/execution logging so the benchmark
        # output stays readable (these are INFO logs, not part of the measured work).
        import ray.data

        ctx = ray.data.DataContext.get_current()
        ctx.enable_progress_bars = False
        ctx.execution_options.verbose_progress = False
        logging.getLogger("ray.data").setLevel(logging.WARNING)


class RayEngine(Engine):
    name = "ray"
    tier = "multi"
    supports_sql = False

    @classmethod
    def available(cls) -> bool:
        # ray.data needs pandas for the Arrow<->block bridge used by the cases.
        return all(importlib.util.find_spec(m) is not None for m in ("ray", "pandas"))

    def handle(self, table: pa.Table):
        import ray.data

        _ensure_ray()
        return ray.data.from_arrow(table)

    def read_parquet(self, uri: str):
        import ray.data

        _ensure_ray()
        return ray.data.read_parquet(uri)

    def scan_handle(self, filesystem: pafs.FileSystem, paths: list[str]):
        import ray.data

        _ensure_ray()
        # Ray Data takes an explicit path list (no glob), and reuses the filesystem the
        # scan suite already resolved rather than re-inferring one per path.
        return ray.data.read_parquet(paths, filesystem=filesystem)
