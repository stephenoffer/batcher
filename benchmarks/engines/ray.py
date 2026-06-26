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

from .base import Engine


def _neutralize_broken_runtime_env_hook() -> None:
    """Drop a ``RAY_RUNTIME_ENV_HOOK``/``RAY_RUNTIME_ENV_PLUGINS`` whose module is missing.

    A host env (e.g. Anyscale's ``cgroup_runtime_plugin``) may export a runtime-env
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


def _ensure_ray() -> None:
    import os

    import ray

    if not ray.is_initialized():
        _neutralize_broken_runtime_env_hook()
        # Start a fresh, isolated local cluster — never attach to a host cluster that
        # may be a different Ray version (a benchmark must be reproducible in-process).
        ray.init(
            address="local",
            include_dashboard=False,
            ignore_reinit_error=True,
            configure_logging=False,
            log_to_driver=False,
            num_cpus=int(os.environ.get("BENCH_RAY_CPUS", "0")) or None,
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
