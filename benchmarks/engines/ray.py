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

The mechanism lives in ``engines/partitioned.py`` rather than here, because Daft has the
identical defect (``daft.from_arrow`` is also one partition) and it survived unfixed for
as long as this explanation lived only in *this* file — a 6.2x handicap charged to a
comparator that ships in the default lineup. A shared function propagates; a docstring
does not.
"""

from __future__ import annotations

import importlib.util
import logging
import os

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .base import Engine
from .partitioned import row_group_rows, scratch_dir


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
    """Ship the suite and Batcher to the workers, and leave the cluster's pip env alone.

    Three things have to be importable in a Ray Data worker for the TPC-H pipelines to run,
    and on a multi-node cluster none of them was:

    * **the suite** — the pipelines are module-level functions, so cloudpickle sends them by
      *reference* and the worker imports ``suites.standard.tpch_ray``. `working_dir` uploads
      the directory and unpacks it per node. See the note on that below.
    * **batcher** — importing ``suites`` pulls in the engine adapters, which import
      ``batcher``. Shipped as a ``py_modules`` entry, the same mechanism Batcher's own
      distributed path uses.
    * **the cluster's pip packages** — ``duckdb`` above all, which the reference side of
      several suites imports.

    This used to pass ``pip: None``, to drop an unresolvable local editable
    (``batcher-engine``) from a platform-injected ``requirements.txt``. That nulls the
    **whole** inherited pip block, not just the bad entry — so it also removed ``duckdb``
    from every worker, and importing ``suites`` then died with
    ``ModuleNotFoundError: No module named 'duckdb'``. Measured here: with the block
    inherited, workers import ``duckdb``, ``pandas``, ``pyarrow`` and ``numpy`` fine
    (``pip_check`` is false on this cluster's block, so the unresolvable marker entries do
    not fail the build). Nulling it was solving a problem this cluster does not have, at the
    cost of one it does.

    ``working_dir`` **uploads** the ``benchmarks/`` directory to the workers, which is what
    makes `suites` importable there. The TPC-H pipelines live in ``suites.standard.tpch_ray``,
    and cloudpickle serializes their ``map_batches`` callables *by reference* because they
    belong to an importable module -- so a worker that cannot import ``suites`` dies with
    ``ModuleNotFoundError: No module named 'suites'`` before running a single batch.

    It used to pass that directory as an absolute ``PYTHONPATH`` instead, and that is only a
    fix on a single node. The path names a location on the **driver's** filesystem; a worker
    on another host has nothing there, so the entry resolves to nothing and the import fails
    exactly as before. The tell is which nodes failed: tasks that happened to land on the
    head node succeeded, and every task on a worker node raised. Ray uploads a ``working_dir``
    to the object store and unpacks it per node, so it is the same directory everywhere --
    the mechanism Batcher already uses for its own package, applied to the suite.

    3.8 MB, uploaded once per session.
    """
    from batcher._internal.paths import package_dir

    benchmarks_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {"working_dir": benchmarks_dir, "py_modules": [package_dir()]}


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


def _row_group_rows(table: pa.Table) -> int:
    """Rows per row group, following Ray Data's own two read defaults.

    Ray Data reads Parquet at row-group granularity and *cannot split below a row group*,
    so the row-group size is what caps the block count — and therefore how many cores the
    query can use. Two Ray defaults bound it, and both must hold: `target_max_block_size`
    (128 MiB) is a **ceiling** on block bytes, not a parallelism target, and Ray's default
    read parallelism is **2x the available CPUs**. `partitioned.row_group_rows` takes the
    smaller of the two, so Ray gets the parallelism its own defaults ask for without ever
    exceeding its own block-size ceiling. This is Ray's configuration, not a constant tuned
    to flatter the result.
    """
    import ray.data

    cap = ray.data.DataContext.get_current().target_max_block_size
    return row_group_rows(table, max_group_bytes=cap)


class RayEngine(Engine):
    name = "ray"
    tier = "multi"
    supports_sql = False

    @classmethod
    def available(cls) -> bool:
        # ray.data needs pandas for the Arrow<->block bridge used by the cases.
        return all(importlib.util.find_spec(m) is not None for m in ("ray", "pandas"))

    def prepare(self) -> None:
        """Attach to the cluster now, so this adapter is the one that sets the job env.

        See `Engine.prepare`: the worker `PYTHONPATH` this engine needs can only be attached
        by whoever calls `ray.init`, and Batcher leads the multi-node lineup.
        """
        _ensure_ray()

    def handle(self, table: pa.Table):
        import ray.data

        _ensure_ray()
        # Parquet round-trip rather than `from_arrow`: see the module docstring. A
        # `from_arrow` handle is one block, which pins every downstream operator to a
        # single core and is what made Ray Data's TPC-H numbers meaningless.
        path = os.path.join(scratch_dir("ray"), f"ray-{id(table):x}.parquet")
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
