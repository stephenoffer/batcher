"""Structural check: every predicate-aware source streams predicates too.

Each source below already pushes a predicate into its ``read`` and declares
``supports_predicate = True``; the streaming ``iter_batches`` must accept the
same ``predicate=`` parameter so the engine can push down on the streaming path
(it keeps a ``Filter`` re-check, so a partial/None translation is always safe).

These are signature/structural assertions only — no live backend is contacted,
and no optional driver is imported, so the test always runs.
"""

from __future__ import annotations

import inspect

import pytest

from batcher.io.formats.lakehouse.iceberg import IcebergSource
from batcher.io.formats.nosql.mongo import MongoSource
from batcher.io.formats.sql.adbc import ADBCSource
from batcher.io.formats.sql.bigquery import BigQuerySource
from batcher.io.formats.sql.clickhouse import ClickHouseSource
from batcher.io.formats.sql.connectorx import ConnectorXSource
from batcher.io.formats.sql.odbc import ODBCSource
from batcher.io.formats.sql.snowflake import SnowflakeSource

pytestmark = pytest.mark.unit

_SOURCES = [
    ADBCSource,
    SnowflakeSource,
    ClickHouseSource,
    ConnectorXSource,
    ODBCSource,
    BigQuerySource,
    IcebergSource,
    MongoSource,
]


@pytest.mark.parametrize("source", _SOURCES, ids=lambda s: s.__name__)
def test_iter_batches_accepts_predicate(source: type) -> None:
    sig = inspect.signature(source.iter_batches)
    assert "predicate" in sig.parameters, (
        f"{source.__name__}.iter_batches must accept a predicate= parameter"
    )
    assert sig.parameters["predicate"].default is None


@pytest.mark.parametrize("source", _SOURCES, ids=lambda s: s.__name__)
def test_source_supports_predicate(source: type) -> None:
    assert source.supports_predicate is True
