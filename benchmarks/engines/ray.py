"""Ray Data adapter — the distributed-streaming comparator.

Ray Data has no SQL surface, so it sits out the standard SQL suites and carries the
operator-mix (filter/groupby/aggregate/sort) on the native ``ray.data.Dataset``
handle. A controlled local Ray is initialized on first use (no dashboard, quiet) so
runs stay reproducible.
"""

from __future__ import annotations

import importlib.util

import pyarrow as pa

from .base import Engine


def _ensure_ray() -> None:
    import ray

    if not ray.is_initialized():
        ray.init(
            include_dashboard=False,
            ignore_reinit_error=True,
            configure_logging=False,
            log_to_driver=False,
        )


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
