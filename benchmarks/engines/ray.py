"""Ray Data adapter — the distributed-streaming comparator.

Ray Data has no SQL surface, so it sits out the standard SQL suites and carries the
operator-mix (filter/groupby/aggregate/sort) and the TPC-H pipelines
(``suites/standard/tpch_ray``) on the native ``ray.data.Dataset`` handle. The
benchmark attaches to the running Ray cluster on first use so the comparison happens
on Ray's own turf.

**Tables are registered through Parquet, not ``from_arrow``.** This mirrors the Spark
adapter, and for the same reason: ``ray.data.from_arrow(table)`` makes exactly *one
block* per table, and a Ray Data block is the unit of parallelism. A one-block
Dataset runs every downstream ``map_batches``, ``groupby``, and ``join`` as a single
task on a single core, so on a 96-core box Ray Data was being measured
single-threaded — 6M-row ``lineitem``, one block, one CPU. That is not a Ray Data
limitation, it is a harness bug, and it is what made the join queries look
"impractically slow" and emit ``Cluster resources are not enough to run any task``.

Writing the normalized Arrow table to Parquet once (untimed setup) and reading it
back with ``ray.data.read_parquet`` is Ray Data's real ingest path, and it blocks the
data the way Ray Data itself would. Row groups are sized to Ray's own
``DataContext.target_max_block_size`` so the block count follows Ray's documented
target rather than a number tuned to flatter the comparison.
"""

from __future__ import annotations

import atexit
import importlib.util
import logging
import os
import shutil
import tempfile
from functools import lru_cache

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .base import Engine

# Floor on row-group rows: tiny dimension tables (nation=25, region=5) must not be
# split into single-row groups, which would cost more in task overhead than the scan.
_MIN_ROW_GROUP_ROWS = 8192


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
    """Drop an unresolvable local editable from the pip env, and put the suite on the path.

    Some managed platforms inject the workspace's ``requirements.txt`` as the default runtime-env
    ``pip`` block, inherited by every task/actor. When that list contains the local
    editable ``batcher-engine`` (not on any index), the per-worker pip build hard-fails
    and Ray Data cannot launch a single task. Ray Data's own dependencies already live
    in the cluster's base env, so nulling ``pip`` for the comparison is both correct and
    the representative setup (workers run the stock Ray Data image). Mirrors
    ``_neutralize_broken_runtime_env_hook`` — a broken inherited env is a no-op to strip.

    ``PYTHONPATH`` carries the ``benchmarks/`` directory to the workers. The TPC-H
    pipelines live in ``suites.standard.tpch_ray``, and cloudpickle serializes their
    ``map_batches`` callables *by reference* because they belong to an importable
    module -- so a worker that cannot import ``suites`` dies with
    ``ModuleNotFoundError: No module named 'suites'`` before running a single batch.
    The driver gets that directory from ``sys.path[0]``; workers are separate processes
    and inherit nothing, so it has to be passed explicitly.
    """
    benchmarks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inherited = os.environ.get("PYTHONPATH", "")
    path = f"{benchmarks_dir}{os.pathsep}{inherited}" if inherited else benchmarks_dir
    return {"pip": None, "env_vars": {"PYTHONPATH": path}}


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


@lru_cache(maxsize=1)
def _scratch() -> str:
    path = tempfile.mkdtemp(prefix="batcher-bench-ray-")
    atexit.register(shutil.rmtree, path, True)
    return path


def _row_group_rows(table: pa.Table) -> int:
    """Rows per row group, following Ray Data's own two read defaults.

    Ray Data reads Parquet at row-group granularity and *cannot split below a row
    group*, so the row-group size is what caps the block count — and therefore how
    many cores the query can use. Two Ray defaults bound it, and both must hold:

    * ``target_max_block_size`` (128 MiB) is a **ceiling** on block bytes, not a
      parallelism target. Sizing to it alone gave ``lineitem`` five row groups, so
      Ray Data ran a 96-core box five-wide and still looked pathologically slow.
    * Ray's default read parallelism is **2x the available CPUs**. That is the number
      of blocks ``read_parquet`` tries to produce when the data permits.

    Taking the smaller of the two gives Ray the parallelism its own defaults ask for
    without ever exceeding its own block-size ceiling. This is Ray's configuration,
    not a constant tuned to flatter the result.
    """
    import ray
    import ray.data

    if not table.num_rows:
        return _MIN_ROW_GROUP_ROWS
    bytes_per_row = max(1, table.nbytes // table.num_rows)
    size_cap = ray.data.DataContext.get_current().target_max_block_size // bytes_per_row
    cpus = int(ray.cluster_resources().get("CPU", 1)) or 1
    parallel_target = -(-table.num_rows // (2 * cpus))  # ceil
    return max(_MIN_ROW_GROUP_ROWS, min(size_cap, parallel_target))


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
        # Parquet round-trip rather than `from_arrow`: see the module docstring. A
        # `from_arrow` handle is one block, which pins every downstream operator to a
        # single core and is what made Ray Data's TPC-H numbers meaningless.
        path = os.path.join(_scratch(), f"handle-{id(table):x}.parquet")
        if not os.path.exists(path):
            pq.write_table(table, path, row_group_size=_row_group_rows(table))
        return ray.data.read_parquet(path)

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
