"""PySpark adapter — opt-in JVM comparator (single-node local mode and cluster).

Spark is heavy (JVM startup, serialization overhead on small in-memory data), so it
is off by default and enabled only via ``--engines spark``. A local ``SparkSession``
is created lazily and reused. The harness's warm-up run amortizes the first-query
JIT/compile cost before timing.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache

import pyarrow as pa

from .base import Engine, SqlRunner


@lru_cache(maxsize=1)
def _session():
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


class SparkEngine(Engine):
    name = "spark"
    tier = "both"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("pyspark") is not None

    def handle(self, table: pa.Table):
        return _session().createDataFrame(table.to_pandas())

    def read_parquet(self, uri: str):
        return _session().read.parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        spark = _session()
        for name, tbl in tables.items():
            spark.createDataFrame(tbl.to_pandas()).createOrReplaceTempView(name)
        return lambda query: _to_arrow(spark.sql(query))
