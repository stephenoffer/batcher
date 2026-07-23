"""The one place a `Source` becomes a `Dataset`.

Every session constructor funnels through `_scan`, which is what makes the
governance rewrite unbypassable: a scan that skipped it would read ungoverned
rows. Keep new constructors going through here rather than building a `Scan`
node themselves.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.api.dataset import Dataset
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Scan
from batcher.plan.schema import SchemaRef


def _scan(source: Source) -> Dataset:
    """Build the `Dataset` for `source`, governed by the active security policy.

    The single place a source becomes a plan, and therefore the single place governance
    has to be applied for it to be unbypassable — see `api.security`.
    """
    from batcher.api.security import govern_scan

    plan: LogicalPlan = Scan(source_id=0, schema=SchemaRef.from_arrow(source.schema()))
    return Dataset(govern_scan(plan, source), sources=[source])


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    """A zero-row RecordBatch carrying `schema` (so empty inputs keep their types)."""
    return pa.RecordBatch.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
