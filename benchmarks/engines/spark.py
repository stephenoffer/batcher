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

# S3 region for the S3A connector (mirrors the DuckDB loader's BENCH_S3_REGION).
_S3_REGION = os.environ.get("BENCH_S3_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


@lru_cache(maxsize=1)
def _hadoop_aws_version() -> str:
    """The `hadoop-aws` version matching Spark's bundled Hadoop client (they must agree).

    Stock PySpark ships the Hadoop client but not the S3A connector, so reading `s3://`
    fails with `No FileSystem for scheme`. The connector JAR's version must match the
    bundled `hadoop-client-*` exactly or class-loading breaks, so it is read off that JAR
    rather than hard-coded. Falls back to a recent version if the JAR name is unexpected.
    """
    import glob

    import pyspark

    jars = glob.glob(os.path.join(os.path.dirname(pyspark.__file__), "jars", "hadoop-client-*.jar"))
    for jar in jars:
        # hadoop-client-api-3.4.2.jar -> 3.4.2
        stem = os.path.basename(jar).removesuffix(".jar")
        version = stem.rsplit("-", 1)[-1]
        if version and version[0].isdigit():
            return version
    return "3.4.2"


def _s3a(uri: str) -> str:
    """Rewrite an ``s3://`` URI to the ``s3a://`` scheme Spark's Hadoop connector uses."""
    return "s3a://" + uri[len("s3://") :] if uri.startswith("s3://") else uri


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
    # Pull the S3A connector (matching Spark's Hadoop version) from Maven at launch, so
    # Spark can read the `s3://` benchmark data directly — without it, `s3a://` reads fail
    # with `No FileSystem for scheme`. The download happens once (cached under ~/.ivy2).
    packages = f"org.apache.hadoop:hadoop-aws:{_hadoop_aws_version()}"
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS",
        f"--driver-memory {_DRIVER_MEMORY} --packages {packages} pyspark-shell",
    )
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[*]")
        .appName("batcher-bench")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        # S3A: pin the region and let the connector's *default* credential chain resolve
        # creds (env vars + instance profile), same as the AWS CLI / DuckDB loader. The
        # provider is deliberately left unset: hadoop-aws 3.4 uses AWS SDK v2, whose
        # default chain is correct here — naming the old v1 `com.amazonaws...` provider
        # raises `ClassNotFoundException` against the v2 bundle.
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint.region", _S3_REGION)
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
        return _session().read.parquet(_s3a(uri))

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        spark = _session()
        for name, tbl in tables.items():
            _register_parquet(spark, name, tbl)
        return lambda query: _to_arrow(spark.sql(query))

    def sql_runner_scan(self, uris: dict[str, str]) -> SqlRunner:
        spark = _session()
        for name, uri in uris.items():
            spark.read.parquet(_s3a(uri)).createOrReplaceTempView(name)
        return lambda query: _to_arrow(spark.sql(query))

    def scan_sql_runner(self, glob: str) -> SqlRunner:
        spark = _session()
        path = _s3a(glob)

        def run(query: str) -> pa.Table:
            spark.read.parquet(path).createOrReplaceTempView("t")
            return _to_arrow(spark.sql(query))

        return run
