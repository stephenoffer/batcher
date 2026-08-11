"""Generated benchmark datasets — the one place the suite builds data instead of reading it.

Most datasets are read from canonical public parquet (``sources``). Two are built here, for
the same reason in both cases: there is nothing published to read.

- The semistructured suite has no public nested-JSON corpus to point at, so it generates one
  deterministically and shares the byte-identical Arrow table across every engine (the same
  parity the loaders give).
- The H2O.ai db-benchmark ships no data at all — every published run generates its own from
  the benchmark's two R scripts, so ``h2o_tables`` follows that spec, exactly as
  ``sources.tables`` runs TPC-DS's own ``dsdgen``.
"""

from __future__ import annotations

from datagen.h2o_tables import build_groupby, build_join
from datagen.json_events import build_events

__all__ = ["build_events", "build_groupby", "build_join"]
