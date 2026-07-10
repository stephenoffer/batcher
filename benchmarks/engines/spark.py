"""PySpark adapter — opt-in JVM comparator (single-node local mode and cluster).

Spark is heavy (JVM startup, serialization overhead on small in-memory data), so it
is off by default and enabled only via ``--engines spark``. A local ``SparkSession``
is created lazily and reused. The harness's warm-up run amortizes the first-query
JIT/compile cost before timing.

Two environment realities the adapter handles rather than crashing on:

* **A JVM must exist**, not just the ``pyspark`` wheel. Without one, `available` reports
  False and the suite omits Spark — instead of letting ``getOrCreate()`` raise
  ``JAVA_GATEWAY_EXITED`` on the first query and take the whole run down. Install one
  with ``python -c "import jdk; print(jdk.install('17', jre=True))"``.
* **Tables are registered through Parquet, not ``parallelize``.** `createDataFrame` on a
  pandas frame serializes every row through the driver JVM, which OOMs its 1 GB default
  heap on a 6M-row `lineitem` — and is not how anyone feeds Spark at scale anyway.
  Writing the shared Arrow table to a temp Parquet file once and reading it back with
  ``spark.read.parquet`` measures Spark's real ingest path.
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import shutil
import tempfile
from functools import lru_cache

import pyarrow as pa
import pyarrow.parquet as pq

from .base import Engine, SqlRunner

# Driver heap for local mode. Must be set before the JVM launches, so it goes through
# `PYSPARK_SUBMIT_ARGS` rather than a `SparkConf` entry (which the driver reads too late).
_DRIVER_MEMORY = os.environ.get("BENCH_SPARK_DRIVER_MEMORY", "32g")


def _java_home() -> str | None:
    """A usable ``JAVA_HOME``: the environment's, or a JRE installed under ``~/.jre``."""
    env = os.environ.get("JAVA_HOME")
    if env and os.path.isfile(os.path.join(env, "bin", "java")):
        return env
    if shutil.which("java"):
        return env or ""  # java is on PATH; Spark finds it without JAVA_HOME
    jre_root = os.path.expanduser("~/.jre")
    if os.path.isdir(jre_root):
        for entry in sorted(os.listdir(jre_root)):
            candidate = os.path.join(jre_root, entry)
            if os.path.isfile(os.path.join(candidate, "bin", "java")):
                return candidate
    return None


@lru_cache(maxsize=1)
def _scratch() -> str:
    """A temp directory for the Parquet copies Spark reads, removed at exit."""
    path = tempfile.mkdtemp(prefix="batcher-bench-spark-")
    atexit.register(shutil.rmtree, path, True)
    return path


@lru_cache(maxsize=1)
def _session():
    home = _java_home()
    if home:
        os.environ.setdefault("JAVA_HOME", home)
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", f"--driver-memory {_DRIVER_MEMORY} pyspark-shell")
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[*]")
        .appName("batcher-bench")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def _to_arrow(sdf) -> pa.Table:
    """Materialize a Spark DataFrame as an Arrow table (pandas bridge for portability)."""
    return pa.Table.from_pandas(sdf.toPandas(), preserve_index=False)


def _register_parquet(spark, name: str, table: pa.Table) -> None:
    """Write `table` to a temp Parquet file and expose it to Spark SQL as `name`."""
    path = os.path.join(_scratch(), f"{name}.parquet")
    if not os.path.exists(path):
        pq.write_table(table, path)
    spark.read.parquet(path).createOrReplaceTempView(name)


class SparkEngine(Engine):
    name = "spark"
    tier = "both"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("pyspark") is not None and _java_home() is not None

    def handle(self, table: pa.Table):
        path = os.path.join(_scratch(), f"handle-{id(table):x}.parquet")
        if not os.path.exists(path):
            pq.write_table(table, path)
        return _session().read.parquet(path)

    def read_parquet(self, uri: str):
        return _session().read.parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        spark = _session()
        for name, tbl in tables.items():
            _register_parquet(spark, name, tbl)
        return lambda query: _to_arrow(spark.sql(query))

    def sql_runner_scan(self, uris: dict[str, str]) -> SqlRunner:
        spark = _session()
        for name, uri in uris.items():
            spark.read.parquet(uri).createOrReplaceTempView(name)
        return lambda query: _to_arrow(spark.sql(query))
