"""Established public benchmark data — the only data the suite reads.

The benchmarks never generate data. This package is the one place a public source is
named, fetched, and normalized to a stable cross-engine schema, so every engine sees
byte-identical inputs — the parity the harness's correctness gate depends on. It splits
by what a benchmark measures:

- :mod:`sources.tables` — TPC-H / TPC-DS / ClickBench tables, materialized into shared
  Arrow (or bound as lazy native scans via ``table_uris`` at large scale).
- :mod:`sources.corpora` — the file corpora whose *reading* is the measurement: the
  three parquet scan layouts and the JPEG image corpus.

Import from ``sources`` directly; the split is an implementation detail.
"""

from __future__ import annotations

from sources.corpora import (
    IMAGE_COUNTS,
    IMAGE_NATIVE_SIZE,
    IMAGES_BASE,
    SCAN_BASE,
    SCAN_LAYOUTS,
    SCAN_SIZES,
    ImageCorpus,
    ScanCorpus,
    image_corpus,
    scan_corpora,
)
from sources.tables import (
    CLICKBENCH_BASE,
    CLICKBENCH_PARTS,
    TPCDS_BASE,
    TPCDS_TABLES,
    TPCH_BASE,
    TPCH_COLUMNS,
    TPCH_TABLES,
    load_tables,
    scan_rename,
    table_uris,
)

__all__ = [
    "CLICKBENCH_BASE",
    "CLICKBENCH_PARTS",
    "IMAGES_BASE",
    "IMAGE_COUNTS",
    "IMAGE_NATIVE_SIZE",
    "SCAN_BASE",
    "SCAN_LAYOUTS",
    "SCAN_SIZES",
    "TPCDS_BASE",
    "TPCDS_TABLES",
    "TPCH_BASE",
    "TPCH_COLUMNS",
    "TPCH_TABLES",
    "ImageCorpus",
    "ScanCorpus",
    "image_corpus",
    "load_tables",
    "scan_corpora",
    "scan_rename",
    "table_uris",
]
