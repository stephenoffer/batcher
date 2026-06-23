"""Contract invariants for predicate-pushdown-capable sources.

Any source that advertises ``supports_predicate = True`` MUST accept a
``predicate`` keyword on both ``read`` and ``iter_batches`` (the engine pushes the
predicate through both the batch and streaming paths). This guards against a new
connector setting the flag but forgetting to thread the predicate.
"""

from __future__ import annotations

from inspect import signature

from batcher.io.formats.base import SOURCES


def _capable() -> list[str]:
    return sorted(
        n for n in SOURCES.names() if getattr(SOURCES.get(n), "supports_predicate", False)
    )


def test_predicate_capable_sources_present():
    # The columnar/lakehouse/SQL/NoSQL connectors that have a real backend filter.
    caps = set(_capable())
    expected = {
        "parquet",
        "parquet_dataset",
        "orc",
        "lance",
        "delta",
        "iceberg",
        "hudi",
        "delta_sharing",
        "adbc",
        "connectorx",
        "snowflake",
        "bigquery",
        "clickhouse",
        "odbc",
        "databricks",
        "mongo",
        "elasticsearch",
        "dynamodb",
        "cassandra",
        "couchbase",
    }
    assert expected <= caps, f"missing predicate pushdown: {expected - caps}"


def test_capable_sources_accept_predicate_on_read_and_iter():
    for name in _capable():
        cls = SOURCES.get(name)
        for method in ("read", "iter_batches"):
            params = signature(getattr(cls, method)).parameters
            assert "predicate" in params, f"{name}.{method} is missing a `predicate` parameter"
